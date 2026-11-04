"""Claude for ShinkaEvolve (ADR 0001's shim).

Shinka's own Anthropic call sends a temperature and a thinking budget, which Claude 5.x rejects
with HTTP 400; retries a 400 for up to 6000 s; sends no prompt caching; ignores cache tokens and
refusals when it prices a call; and drops the cost of an empty reply. ``install`` replaces that
call with ``query_anthropic_async`` here, which:

- sends what Claude 5.x accepts: adaptive thinking (the default), ``output_config.effort``, no
  temperature, streamed so a long, high-effort reply cannot time out;
- caches the system prompt up to the end of the task message (``cache_prefix``);
- turns a refusal or a reply cut at ``max_tokens`` into an empty reply, which Shinka counts as a
  failed proposal, and a 400 into an error Shinka does not retry;
- retries rate limits, timeouts and server errors itself, with backoff, after turning off the SDK's
  hidden retries, so every failed attempt reaches the ledger;
- appends every call to a JSONL ledger with all four token counts and its cost, and refuses new
  calls once the ledger reaches ``max_usd``. The ledger, not Shinka's tally, is the run's cost.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

# $ per million tokens: input, output, cache write (5-minute TTL, 1.25x input), cache read
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 2.50, 0.20),
    "claude-opus-5-5": (4.00, 20.00, 5.00, 0.20),
}
EFFORTS = ("low", "medium", "high", "xhigh", "max")
NOT_RETRYABLE = (anthropic.BadRequestError, anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                 anthropic.NotFoundError, anthropic.UnprocessableEntityError)


@dataclass(frozen=True)
class Settings:
    ledger: Path
    effort: str = "medium"
    max_usd: float = 7.0
    max_tokens: int = 32000
    cache_prefix: str = ""      # system text up to its end gets a cache breakpoint
    retries: int = 4            # extra attempts after a rate limit, timeout or server error

    def __post_init__(self):
        if self.effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, not {self.effort!r}")
        if self.max_usd <= 0:
            raise ValueError("max_usd must be positive")


_settings: Settings | None = None
_sleep = asyncio.sleep          # replaced in tests


def backoff(error: anthropic.APIError, attempt: int) -> float:
    """Seconds to wait before retrying: the server's ``retry-after`` when it sends one, else 2, 4,
    8, ... capped at 60."""
    response = getattr(error, "response", None)
    try:
        return min(60.0, float(response.headers["retry-after"]))
    except (AttributeError, KeyError, TypeError, ValueError):
        return min(60.0, 2.0 ** (attempt + 1))


def cost(model: str, usage) -> float:
    """Dollars for one call from its ``usage``; thinking tokens are billed as output."""
    p_in, p_out, p_write, p_read = PRICES[model]
    return (usage.input_tokens * p_in + usage.output_tokens * p_out
            + (usage.cache_creation_input_tokens or 0) * p_write
            + (usage.cache_read_input_tokens or 0) * p_read) / 1e6


def spent(ledger: Path) -> float:
    """Dollars recorded in the ledger so far."""
    if not Path(ledger).exists():
        return 0.0
    with Path(ledger).open() as handle:
        return sum(json.loads(line)["cost"] for line in handle if line.strip())


def system_blocks(system_msg: str, prefix: str) -> list[dict]:
    """The system prompt as content blocks, with a cache breakpoint after ``prefix`` when the
    prompt starts with it (Shinka appends the edit format after the task message)."""
    if prefix and system_msg.startswith(prefix):
        blocks = [{"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}}]
        rest = system_msg[len(prefix):]
        return blocks + ([{"type": "text", "text": rest}] if rest else [])
    return [{"type": "text", "text": system_msg}]


def _record(ledger: Path, entry: dict) -> None:
    Path(ledger).parent.mkdir(parents=True, exist_ok=True)
    with Path(ledger).open("a") as handle:
        handle.write(json.dumps(entry) + "\n")


async def query_anthropic_async(client, model, msg, system_msg, msg_history, output_model,
                                model_posteriors=None, **kwargs):
    """Drop-in for ``shinka.llm.query.query_anthropic_async``; Shinka's sampled ``kwargs``
    (temperature, thinking budget, max_tokens) are ignored."""
    from shinka.llm.providers.anthropic import split_content_blocks
    from shinka.llm.providers.errors import NonRetryableLLMError
    from shinka.llm.providers.result import QueryResult

    s = _settings
    if s is None:
        raise NonRetryableLLMError("mevolve.search.claude.install() was not called")
    if output_model is not None:
        raise NonRetryableLLMError("structured output is not used by MEvolve")
    if model not in PRICES:
        raise NonRetryableLLMError(f"no price for {model}; add it to mevolve.search.claude.PRICES")
    messages = list(msg_history) + [{"role": "user", "content": [{"type": "text", "text": msg}]}]
    client = client.with_options(max_retries=0)     # retries happen below, where the ledger sees them
    for attempt in range(s.retries + 1):
        if spent(s.ledger) >= s.max_usd:      # before every attempt: other proposals spend while this one waits
            raise NonRetryableLLMError(f"the dollar cap (${s.max_usd:.2f}) is reached")
        t0 = time.time()
        try:
            async with client.messages.stream(model=model, max_tokens=s.max_tokens,
                                              system=system_blocks(system_msg, s.cache_prefix),
                                              messages=messages,
                                              output_config={"effort": s.effort}) as stream:
                response = await stream.get_final_message()
            break
        except NOT_RETRYABLE as e:
            _record(s.ledger, {"time": t0, "model": model, "effort": s.effort, "error": f"{type(e).__name__}: {e}",
                               "retryable": False, "cost": 0.0})
            raise NonRetryableLLMError(f"{type(e).__name__}: {e}") from e
        except anthropic.APIError as e:     # rate limits, timeouts, 5xx: retried here, each one in the ledger
            _record(s.ledger, {"time": t0, "model": model, "effort": s.effort, "error": f"{type(e).__name__}: {e}",
                               "retryable": True, "attempt": attempt + 1, "cost": 0.0})
            if attempt == s.retries:
                raise
            await _sleep(backoff(e, attempt))
    usage = response.usage
    dollars = cost(model, usage)
    content, thought = split_content_blocks(response)
    stop = response.stop_reason
    _record(s.ledger, {
        "time": t0, "seconds": time.time() - t0, "model": model, "served_model": response.model,
        "effort": s.effort, "stop_reason": stop,
        "refusal_category": getattr(getattr(response, "stop_details", None), "category", None),
        "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        "cost": dollars, "request_id": getattr(response, "_request_id", None), "id": response.id})
    if stop in ("refusal", "max_tokens"):       # a failed proposal: Shinka moves on, never retries it
        content = ""
    p_in, p_out, p_write, p_read = PRICES[model]
    input_cost = (usage.input_tokens * p_in + (usage.cache_creation_input_tokens or 0) * p_write
                  + (usage.cache_read_input_tokens or 0) * p_read) / 1e6
    history = messages + [{"role": "assistant", "content": [{"type": "text", "text": content}]}]
    return QueryResult(
        content=content, msg=msg, system_msg=system_msg, new_msg_history=history, model_name=model,
        kwargs={"effort": s.effort, "max_tokens": s.max_tokens},
        input_tokens=usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0),
        output_tokens=usage.output_tokens, cost=dollars, input_cost=input_cost,
        output_cost=usage.output_tokens * p_out / 1e6, thought=thought, model_posteriors=model_posteriors)


def _sync_query_refused(*args, **kwargs):
    from shinka.llm.providers.errors import NonRetryableLLMError
    raise NonRetryableLLMError("MEvolve routes every Claude call through the async shim")


def install(settings: Settings) -> None:
    """Route Shinka's Anthropic calls through this module. Shinka looks the function up at call
    time, so replacing the module attribute is enough."""
    global _settings
    import shinka.llm.query as shinka_query

    _settings = settings
    shinka_query.query_anthropic_async = query_anthropic_async
    shinka_query.query_anthropic = _sync_query_refused

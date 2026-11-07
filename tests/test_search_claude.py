"""The Claude shim, against a fake client: no API call is made."""
import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("shinka")
anthropic = pytest.importorskip("anthropic")
import httpx  # noqa: E402

from mevolve.search import claude  # noqa: E402

MODEL = "claude-sonnet-5-5"


def usage(inp=100, out=50, write=0, read=0):
    return SimpleNamespace(input_tokens=inp, output_tokens=out, cache_creation_input_tokens=write,
                           cache_read_input_tokens=read)


def message(text="<NAME>x</NAME> code", stop="end_turn", u=None, category=None, thinking=""):
    blocks = [SimpleNamespace(type="thinking", thinking=thinking), SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=blocks, stop_reason=stop, usage=u or usage(), model=MODEL, id="msg_1",
                           stop_details=SimpleNamespace(category=category) if category else None)


class FakeStream:
    def __init__(self, result, calls, kwargs):
        self.result, self.calls, self.kwargs = result, calls, kwargs

    async def __aenter__(self):
        self.calls.append(self.kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.result


class FakeClient:
    """Returns ``results`` in turn, one per request (the last one repeats)."""

    def __init__(self, *results):
        self.calls, self.options, self.results = [], [], list(results)
        self.messages = SimpleNamespace(stream=lambda **kw: FakeStream(self._next(), self.calls, kw))

    def _next(self):
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]

    def with_options(self, **kw):
        self.options.append(kw)
        return self


def run(client, settings, system="TASK\nformat", **kw):
    claude.install(settings)
    return asyncio.run(claude.query_anthropic_async(client, MODEL, "the parent program", system, [], None,
                                                    temperature=0.7, thinking={"type": "enabled"}, **kw))


def settings(tmp_path, **kw):
    return claude.Settings(ledger=tmp_path / "ledger.jsonl", cache_prefix="TASK\n", **kw)


def test_request_is_what_claude_5_accepts(tmp_path):
    client = FakeClient(message())
    result = run(client, settings(tmp_path, effort="high"))
    sent = client.calls[0]
    assert "temperature" not in sent                # Shinka's sampled temperature and thinking budget are dropped
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}     # the reasoning comes back (#74)
    assert sent["output_config"] == {"effort": "high"} and sent["model"] == MODEL
    assert sent["system"][0] == {"type": "text", "text": "TASK\n", "cache_control": {"type": "ephemeral"}}
    assert sent["system"][1] == {"type": "text", "text": "format"}
    assert result.content == "<NAME>x</NAME> code"


def test_cost_counts_cache_tokens_and_lands_in_the_ledger(tmp_path):
    u = usage(inp=1000, out=2000, write=10_000, read=100_000)
    result = run(FakeClient(message(u=u)), settings(tmp_path))
    expected = (1000 * 2.0 + 2000 * 10.0 + 10_000 * 2.5 + 100_000 * 0.2) / 1e6
    assert result.cost == pytest.approx(expected)
    entry = json.loads((tmp_path / "ledger.jsonl").read_text())
    assert entry["cost"] == pytest.approx(expected) and entry["cache_read_input_tokens"] == 100_000
    assert claude.spent(tmp_path / "ledger.jsonl") == pytest.approx(expected)


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_refusal_and_cutoff_are_failed_proposals_but_still_cost(tmp_path, stop):
    result = run(FakeClient(message(stop=stop, category="cyber" if stop == "refusal" else None)), settings(tmp_path))
    assert result.content == ""
    entry = json.loads((tmp_path / "ledger.jsonl").read_text())
    assert entry["stop_reason"] == stop and entry["cost"] > 0


def test_a_400_is_not_retried(tmp_path):
    from shinka.llm.providers.errors import NonRetryableLLMError
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.BadRequestError("temperature is not supported", response=httpx.Response(400, request=request),
                                      body=None)
    client = FakeClient(error)
    with pytest.raises(NonRetryableLLMError):
        run(client, settings(tmp_path))
    assert len(client.calls) == 1
    assert "BadRequestError" in json.loads((tmp_path / "ledger.jsonl").read_text())["error"]


def test_the_cap_stops_new_calls(tmp_path):
    from shinka.llm.providers.errors import NonRetryableLLMError
    s = settings(tmp_path, max_usd=0.01)
    s.ledger.write_text(json.dumps({"cost": 0.02}) + "\n")
    client = FakeClient(message())
    with pytest.raises(NonRetryableLLMError, match="cap"):
        run(client, s)
    assert client.calls == []


def test_unknown_model_and_bad_effort_are_refused(tmp_path):
    from shinka.llm.providers.errors import NonRetryableLLMError
    claude.install(settings(tmp_path))
    with pytest.raises(NonRetryableLLMError, match="no price"):
        asyncio.run(claude.query_anthropic_async(FakeClient(message()), "claude-x", "m", "s", [], None))
    with pytest.raises(ValueError):
        claude.Settings(ledger=tmp_path / "l", effort="extreme")


def test_install_replaces_shinkas_call(tmp_path):
    import shinka.llm.query as q
    claude.install(settings(tmp_path))
    assert q.query_anthropic_async is claude.query_anthropic_async


def test_system_without_the_prefix_is_sent_whole():
    assert claude.system_blocks("other", "TASK") == [{"type": "text", "text": "other"}]


def rate_limit(retry_after=None):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    headers = {"retry-after": retry_after} if retry_after else {}
    return anthropic.RateLimitError("slow down", response=httpx.Response(429, request=request, headers=headers),
                                    body=None)


@pytest.fixture
def waits(monkeypatch):
    seconds = []

    async def sleep(s):
        seconds.append(s)
    monkeypatch.setattr(claude, "_sleep", sleep)
    return seconds


def test_retryable_errors_are_recorded_and_re_raised(tmp_path, waits):
    with pytest.raises(anthropic.RateLimitError):
        run(FakeClient(rate_limit()), settings(tmp_path, retries=2))
    entries = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [e["attempt"] for e in entries] == [1, 2, 3]
    assert all(e["retryable"] and "RateLimitError" in e["error"] for e in entries)
    assert waits == [2.0, 4.0]


def test_a_retry_then_success_leaves_both_in_the_ledger(tmp_path, waits):
    client = FakeClient(rate_limit(retry_after="7"), message())
    result = run(client, settings(tmp_path))
    assert client.options == [{"max_retries": 0}]       # the SDK's own retries would hide the 429
    entries = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [("error" in e) for e in entries] == [True, False]
    assert waits == [7.0] and result.content == "<NAME>x</NAME> code"


def test_the_cap_is_rechecked_before_a_retry(tmp_path, monkeypatch):
    s = settings(tmp_path, max_usd=0.01)

    async def other_proposals_spend(seconds):      # while this call waits, others reach the cap
        claude._record(s.ledger, {"time": 0, "cost": 0.02})
    monkeypatch.setattr(claude, "_sleep", other_proposals_spend)
    from shinka.llm.providers.errors import NonRetryableLLMError
    client = FakeClient(rate_limit(), message())
    with pytest.raises(NonRetryableLLMError, match="cap"):
        run(client, s)
    assert len(client.calls) == 1


def test_the_thinking_summary_reaches_shinka(tmp_path):
    client = FakeClient(message(thinking="The parent's reward rate collapsed, so add forgetting."))
    result = run(client, settings(tmp_path))
    assert result.thought == "The parent's reward rate collapsed, so add forgetting."
    assert result.content == "<NAME>x</NAME> code"          # the summary never enters the proposal

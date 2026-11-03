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


def message(text="<NAME>x</NAME> code", stop="end_turn", u=None, category=None):
    blocks = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
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
    def __init__(self, result):
        self.calls = []
        self.messages = SimpleNamespace(stream=lambda **kw: FakeStream(result, self.calls, kw))


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
    assert "temperature" not in sent and "thinking" not in sent
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


def test_retryable_errors_are_recorded_and_re_raised(tmp_path):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)
    with pytest.raises(anthropic.RateLimitError):
        run(FakeClient(error), settings(tmp_path))
    entry = json.loads((tmp_path / "ledger.jsonl").read_text())
    assert entry["retryable"] is True and "RateLimitError" in entry["error"]

"""The whole pipeline over real HTTP: the CLI, three LLM seats, tool calls, feed, trace.

A local server speaks the OpenAI chat-completions protocol that OpenRouter uses.
Its three behaviours cover the three ways a game can end:
  ok     reads the state, then discards: the game is played to its end
  down   always answers 503: an infrastructure fault, nobody to blame
  chatty never calls a tool: a model fault, that seat is to blame
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from hanabi_llm.__main__ import build_parser, run
from hanabi_llm.faults import FaultKind, GameStatus
from hanabi_llm.feed import read_feed
from hanabi_llm.fold import fold

ASSISTANT = "choices"


def reply(message: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", **message}}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def tool_call(name: str, args: dict[str, Any], n: int) -> dict[str, Any]:
    return reply({"content": None, "tool_calls": [{"id": f"call_{n}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})


class FakeProvider:
    """Records every request and answers by model id."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def answer(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        model, messages = body["model"], body["messages"]
        if model.endswith("down"):
            return 503, {"error": "overloaded"}
        if model.endswith("chatty"):
            return 200, reply({"content": "Let me think about the best move..."})
        if messages[-1]["role"] == "tool":                       # it has read the state: now act
            return 200, tool_call("discard_card", {"slot": 0}, len(messages))
        return 200, tool_call("get_state", {"section": "tokens"}, len(messages))


@pytest.fixture
def provider() -> Iterator[tuple[FakeProvider, str]]:
    fake = FakeProvider()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append(body)
            fake.headers.append({k.lower(): v for k, v in self.headers.items()})
            status, payload = fake.answer(body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield fake, f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
    server.shutdown()
    server.server_close()


def play(tmp_path: Path, url: str, models: str, monkeypatch: pytest.MonkeyPatch, *extra: str):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    args = build_parser().parse_args(["--models", models, "--seed", "5", "--api-url", url, "--log", str(tmp_path / "g.jsonl"),
                                      "--trace", str(tmp_path / "t.jsonl"), *extra])
    return asyncio.run(run(args))


def test_three_llm_seats_play_a_whole_game_through_tool_calls(provider, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake, url = provider
    result = play(tmp_path, url, "fake/ok-a,fake/ok-b,fake/ok-c", monkeypatch)
    assert result.status is GameStatus.COMPLETED and result.fault is None and result.counts_for_ranking
    assert "final round" in result.reason                              # discard-only play runs the deck out

    # what the provider was sent
    assert {h["authorization"] for h in fake.headers} == {"Bearer test-key"}
    first = fake.requests[0]
    assert first["model"] == "fake/ok-a" and first["tool_choice"] == "auto" and first["max_tokens"] == 4096
    assert [t["function"]["name"] for t in first["tools"]] == ["discard_card", "give_hint", "play_card", "get_state", "explain_rules"]
    assert first["messages"][0]["role"] == "system" and first["messages"][0]["content"].startswith("You are Alice.")
    assert "## hands" in first["messages"][1]["content"]               # the turn opens with the whole state
    by_model = {r["model"]: r["messages"][0]["content"] for r in fake.requests}
    assert [by_model[m].split(".")[0] for m in ("fake/ok-a", "fake/ok-b", "fake/ok-c")] == ["You are Alice", "You are Bob", "You are Carol"]
    assert "mechanic" not in " ".join(by_model.values()).lower()        # the baseline: no strategy anywhere

    # a turn is two model calls: read, then act; the tool's answer went back into the conversation
    assert len(fake.requests) == 2 * result.turns
    answer = next(r for r in fake.requests if r["messages"][-1]["role"] == "tool")["messages"][-1]["content"]
    assert answer.startswith("## tokens") and "Hint tokens: available 8" in answer

    # the artefacts
    feed = read_feed(tmp_path / "g.jsonl")
    assert feed[0]["seats"] == {"Alice": "fake/ok-a", "Bob": "fake/ok-b", "Carol": "fake/ok-c"} and feed[0]["config"]["knowledge"] == []
    assert fold(feed).score == result.score and feed[-1]["status"] == "completed"
    rows = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert len(rows) == result.turns and all(r["tokens"] == [200, 20] for r in rows)     # two calls x (100, 10)
    called = [m["tool_calls"][0]["function"]["name"] for m in rows[0]["messages"] if m.get("role") == "assistant"]
    assert called == ["get_state", "discard_card"]
    # timing: every line of the feed is stamped, every turn and model call is timed in the trace
    assert all(r["ts"].endswith("Z") for r in feed)
    assert all(r["started"].endswith("Z") and r["duration_s"] >= 0 for r in rows)
    calls = [m["call"] for m in rows[0]["messages"] if "call" in m]
    assert len(calls) == 2 and all(c["attempts"] == 1 and c["latency_s"] >= 0 and c["started"].endswith("Z") for c in calls)


def test_a_provider_outage_aborts_the_game_without_blaming_the_model(provider, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import hanabi_llm.llm as llm

    fake, url = provider
    real_sleep = asyncio.sleep

    async def fast(_seconds: float) -> None:
        await real_sleep(0)

    monkeypatch.setattr(llm.asyncio, "sleep", fast)                     # skip the retry backoff
    result = play(tmp_path, url, "fake/down,fake/ok-b,fake/ok-c", monkeypatch)
    assert result.status is GameStatus.ABORTED_INFRA_FAULT and not result.counts_for_ranking
    assert result.fault is not None and result.fault.kind is FaultKind.PROVIDER_ERROR and result.fault.player == "Alice"
    assert len(fake.requests) == 4                                      # the client retried 4 times before giving up
    assert read_feed(tmp_path / "g.jsonl")[-1]["fault"]["kind"] == "provider_error"


def test_a_model_that_never_calls_a_tool_is_at_fault(provider, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake, url = provider
    result = play(tmp_path, url, "fake/ok-a,fake/chatty,fake/ok-c", monkeypatch)
    assert result.status is GameStatus.ABORTED_MODEL_FAULT and not result.counts_for_ranking
    assert result.fault is not None and result.fault.player == "Bob" and result.fault.kind is FaultKind.INVALID_ACTIONS
    assert result.turns == 1                                            # Alice played her turn before Bob failed his
    assert len([r for r in fake.requests if r["model"].endswith("chatty")]) == 5       # the budget: 5 attempts, then the fault
    last = read_feed(tmp_path / "g.jsonl")[-1]
    assert last["status"] == "aborted_model_fault" and last["official_score"] is None
    assert last["fault"]["detail"].count('no tool call, it said: "Let me think about the best move...') == 5   # the feed says what happened

    # the faulted turn is in the trace too, with the whole conversation, so it can be read afterwards
    rows = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert [r["player"] for r in rows] == ["Alice", "Bob"] and rows[0].get("fault") is None
    assert rows[1]["fault"]["kind"] == "invalid_actions" and rows[1]["action"] is None
    assert sum(1 for m in rows[1]["messages"] if m.get("role") == "assistant") == 5

    from hanabi_llm.trace_report import format_report

    report = format_report(rows)
    assert "Bob" in report and report.count("no tool call, it said") == 5 and "FAULT  Bob  T2 invalid_actions" in report

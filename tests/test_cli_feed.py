"""The CLI writes a feed a viewer can follow, and a trace the feed does not need."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from hanabi_llm.__main__ import build_parser, run
from hanabi_llm.feed import read_feed
from hanabi_llm.fold import fold


def play(tmp_path: Path, models: str, seed: int, *extra: str) -> tuple[list[dict], Path, object]:  # type: ignore[type-arg]
    log, trace = tmp_path / "g.jsonl", tmp_path / "t.jsonl"
    args = build_parser().parse_args(["--models", models, "--seed", str(seed), "--log", str(log), "--trace", str(trace), *extra])
    result = asyncio.run(run(args))
    return read_feed(log), trace, result


def test_the_log_is_a_plain_json_feed_with_header_events_and_result(tmp_path: Path) -> None:
    feed, _, result = play(tmp_path, "random,random,random", 3)
    assert [r["type"] for r in feed[:1]] == ["header"] and feed[-1]["type"] == "result"
    assert {r["type"] for r in feed[1:-1]} == {"event"}
    header = feed[0]
    assert header["schema"] == 3 and header["seed"] == 3 and header["names"] == ["Alice", "Bob", "Carol"]
    assert header["seats"] == {"Alice": "random", "Bob": "random", "Carol": "random"}
    assert len(header["deal"]["hands"]) == 3 and header["deal"]["deck_size"] == 35
    assert all(isinstance(c["color"], str) and isinstance(c["rank"], int) for h in header["deal"]["hands"] for c in h)  # real JSON, no reprs
    assert all(set(c) == {"color", "rank"} for h in header["deal"]["hands"] for c in h)               # a card is a colour and a rank, nothing else
    assert '"uid"' not in json.dumps(feed)                                                              # no card id anywhere in the feed
    assert feed[-1]["status"] == "completed" and feed[-1]["score"] == result.score  # type: ignore[attr-defined]


def test_folding_the_written_file_gives_the_final_state_of_the_game(tmp_path: Path) -> None:
    feed, _, result = play(tmp_path, "random,random", 5)
    state = fold(feed)
    assert state.score == result.score and state.over_reason == result.reason  # type: ignore[attr-defined]


def test_the_trace_is_separate_and_holds_no_game_state(tmp_path: Path) -> None:
    feed, trace, _ = play(tmp_path, "random,random", 5)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert rows and {"turn", "player", "action", "messages", "tokens"} <= rows[0].keys()
    assert "messages" not in json.dumps(feed)  # conversations never enter the feed


def test_the_feed_is_reproducible_from_the_seed(tmp_path: Path) -> None:
    a, _, _ = play(tmp_path, "random,random", 9)
    (tmp_path / "again").mkdir()
    b, _, _ = play(tmp_path / "again", "random,random", 9)
    strip = lambda recs: [{k: v for k, v in r.items() if k != "ts"} for r in recs]  # the time of writing differs
    assert strip(a) == strip(b)

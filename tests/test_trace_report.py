"""The trace report: what each model did, and exactly what was rejected."""

from __future__ import annotations

import json

from hanabi_llm.trace_report import format_report, summarise


def assistant(*calls: tuple[str, str, str], content: str | None = None) -> dict:  # type: ignore[type-arg]
    msg: dict = {"role": "assistant", "content": content or ""}  # type: ignore[type-arg]
    if calls:
        msg["tool_calls"] = [{"id": cid, "type": "function", "function": {"name": name, "arguments": args}} for cid, name, args in calls]
    return msg


def tool(cid: str, content: str) -> dict:  # type: ignore[type-arg]
    return {"role": "tool", "tool_call_id": cid, "content": content}


ROWS = [
    {"turn": 1, "player": "Alice", "action": {"tool": "discard_card", "position": 0}, "tokens": [100, 10], "messages": [
        assistant(("c1", "get_state", "{}")), tool("c1", "## turn ..."),
        assistant(("c2", "discard_card", '{"position": 0}')), tool("c2", "Accepted: discard_card. Your turn is over.")]},
    {"turn": 3, "player": "Carol", "action": {"tool": "play_card", "position": 2}, "tokens": [300, 40], "messages": [
        assistant(("c1", "play_card", '{"position": 9}')), tool("c1", "Error: position 9 is not in your hand. Your hand has 5 cards, positions 0 to 4."),
        assistant(content="Hmm, let me reconsider."),
        assistant(("c2", "explain_rules", '{"topic": "nope"}')), tool("c2", "Error: Unknown topic 'nope'."),
        assistant(("c3", "play_card", '{"position": 2}')), tool("c3", "Accepted: play_card. Your turn is over.")]},
    {"turn": 9, "player": "Carol", "action": None, "tokens": None, "fault": {"kind": "invalid_actions", "detail": "5 rejected action attempts in one turn: ..."},
     "messages": [assistant(("c1", "give_hint", '{"player": "Carol", "hint": "red"}')), tool("c1", "Error: You cannot give a hint to yourself.")]},
]


def test_the_counts_per_player() -> None:
    stats, _ = summarise(ROWS)
    alice, carol = stats["Alice"], stats["Carol"]
    assert (alice.turns, alice.model_calls, dict(alice.tools), alice.rejected_actions) == (1, 2, {"get_state": 1, "discard_card": 1}, 0)
    assert (carol.turns, carol.model_calls) == (2, 5)                           # four model calls in turn 3, one in the faulted turn
    assert carol.no_tool_replies == 1 and carol.rejected_actions == 2 and carol.read_errors == 1
    assert (carol.tokens_in, carol.tokens_out) == (300, 40)                     # a faulted turn has no token count, and that is fine


def test_every_rejection_is_listed_in_words_in_order() -> None:
    _, incidents = summarise(ROWS)
    assert [(i.turn, i.player) for i in incidents] == [(3, "Carol"), (3, "Carol"), (9, "Carol")]
    assert incidents[0].text.startswith('play_card({"position": 9}) -> Error: position 9 is not in your hand')
    assert incidents[1].text == 'no tool call, it said: "Hmm, let me reconsider."'
    assert 'give_hint({"player": "Carol", "hint": "red"})' in incidents[2].text and "cannot give a hint to yourself" in incidents[2].text


def test_a_mistake_in_a_read_tool_is_counted_apart_from_rejected_actions() -> None:
    stats, incidents = summarise(ROWS)
    assert stats["Carol"].read_errors == 1 and not any("explain_rules" in i.text for i in incidents)


def test_the_report_reads_as_a_table_then_the_incidents_then_the_fault() -> None:
    report = format_report(ROWS)
    lines = report.splitlines()
    assert lines[0].split()[:3] == ["player", "turns", "calls"] and lines[1].startswith("Alice") and lines[2].startswith("Carol")
    assert "get_state x1, discard_card x1" in lines[1]
    assert "Rejected attempts and replies without a tool call:" in report and "  T3 Carol: no tool call" in report
    assert report.splitlines()[-1] == "FAULT  Carol  T9 invalid_actions: 5 rejected action attempts in one turn"   # not repeated in full


def test_a_clean_trace_says_so() -> None:
    assert "No rejected attempts" in format_report(ROWS[:1])


def test_it_reads_the_file_the_cli_writes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from hanabi_llm.trace_report import read_trace

    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in ROWS) + "\n\n")
    assert read_trace(path) == ROWS

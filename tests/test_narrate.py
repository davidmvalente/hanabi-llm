"""The wording of events: readable, copy-pasteable, and defined in one place."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conftest import B, R, Y, rig
from hanabi_llm.__main__ import build_parser, run
from hanabi_llm.engine import Game
from hanabi_llm.feed import read_feed
from hanabi_llm.model import DiscardCard, GiveHint, PlayCard
from hanabi_llm.narrate import segments, to_text
from hanabi_llm.viewer import frames
from hanabi_llm.views import build_view, describe_event, narrate_event

NAMES = ["Alice", "Bob", "Carol"]
CARD = {"color": "blue", "rank": 1}


def line(rec: dict, you: int | None = None) -> str:  # type: ignore[type-arg]
    return to_text(segments({"type": "event", **rec}), NAMES, you)


@pytest.mark.parametrize(("rec", "expected"), [
    # the hints from the review, in the new wording
    ({"turn": 7, "kind": "hint", "actor": 0, "receiver": 2, "attribute": "color", "value": "green", "targets": [3]},
     "T7 Alice hinted Carol: slot 3 is green"),
    ({"turn": 11, "kind": "hint", "actor": 1, "receiver": 2, "attribute": "rank", "value": 3, "targets": [3]},
     "T11 Bob hinted Carol: slot 3 is rank 3"),
    ({"turn": 9, "kind": "hint", "actor": 2, "receiver": 0, "attribute": "rank", "value": 2, "targets": [1, 3]},
     "T9 Carol hinted Alice: slots 1 and 3 are rank 2"),
    ({"turn": 1, "kind": "hint", "actor": 0, "receiver": 1, "attribute": "color", "value": "red", "targets": [0, 2, 4]},
     "T1 Alice hinted Bob: slots 0, 2 and 4 are red"),
    # the colour is in the TEXT of every card, so it survives copy and paste
    ({"turn": 8, "kind": "discard", "actor": 1, "slot": 1, "card": CARD}, "T8 Bob discarded blue 1 from slot 1"),
    ({"turn": 8, "kind": "draw", "actor": 1, "slot": 1, "card": {"color": "red", "rank": 2}}, "T8 Bob drew red 2 into slot 1"),
    ({"turn": 8, "kind": "draw", "actor": 1, "card": {"color": "red", "rank": 2}}, "T8 Bob drew red 2"),                 # an older feed: no slot
    ({"turn": 10, "kind": "play", "actor": 0, "slot": 2, "card": {"color": "blue", "rank": 2}, "success": False},
     "T10 Alice played blue 2 from slot 2 -> misfire (fuse lost)"),
    ({"turn": 5, "kind": "play", "actor": 0, "slot": 0, "card": CARD, "success": True}, "T5 Alice played blue 1 from slot 0 -> placed"),
    ({"turn": 65, "kind": "end", "reason": "all fuse tokens were used up", "score": 4}, "T65 Game over: all fuse tokens were used up (score 4)"),
])
def test_the_wording(rec: dict, expected: str) -> None:  # type: ignore[type-arg]
    assert line(rec) == expected


def test_a_hidden_draw_is_a_card_not_a_colour() -> None:
    hidden = {"turn": 8, "kind": "draw", "actor": 1, "slot": 1, "card": None}
    assert line(hidden) == "T8 Bob drew a card into slot 1"
    assert line({**hidden, "card": {"uid": 2, "color": None, "rank": None}}) == "T8 Bob drew a card into slot 1"   # an older feed's null identity still reads
    assert line({"turn": 3, "kind": "play", "actor": 0, "position": 2, "card": CARD, "success": True}) == "T3 Alice played blue 1 from slot 2 -> placed"   # and `position`


def test_no_line_carries_information_only_in_formatting() -> None:
    """Every card and every hint value is a word in the text; the turn label is followed by a space."""
    for rec in (
        {"turn": 8, "kind": "discard", "actor": 1, "slot": 1, "card": CARD},
        {"turn": 7, "kind": "hint", "actor": 0, "receiver": 2, "attribute": "color", "value": "green", "targets": [3]},
    ):
        text = line(rec)
        assert text.startswith("T") and text.split(" ")[0][1:].isdigit() and " " in text   # "T7 Alice", never "T7Alice"
    assert "blue 1" in line({"turn": 8, "kind": "discard", "actor": 1, "slot": 1, "card": CARD})


def test_the_prompt_history_uses_the_same_wording_and_marks_you() -> None:
    game = Game(NAMES, seed=4)
    rig(game, [[(R, 1)] * 5, [(B, 2)] * 5, [(Y, 3)] * 5], deck=[(R, 5)] * 10)
    game.apply(0, GiveHint(1, B))
    game.apply(1, PlayCard(0))
    game.apply(2, DiscardCard(0))
    view = build_view(game, 1)  # what Bob's model receives: the events are already redacted for him
    texts = [describe_event(e, view) for e in view.events]
    assert texts[0] == "T1 Alice hinted Bob (you): slots 0, 1, 2, 3 and 4 are blue"
    assert texts[1].startswith("T2 Bob (you) played blue 2 from slot 0 -> ")
    assert texts[2] == "T2 Bob (you) drew a card into slot 0"                          # you do not see what you drew, but where it went
    assert [narrate_event(e, NAMES) for e in game.state.events][3] == "T3 Carol discarded yellow 3 from slot 0"


def test_the_viewer_is_sent_the_same_wording(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    asyncio.run(run(build_parser().parse_args(["--models", "random,random,random", "--seed", "3", "--log", str(path)])))
    feed = read_feed(path)
    names = feed[0]["names"]
    got = list(frames(path, follow=False))
    for f, rec in zip(got, feed, strict=True):
        if rec["type"] == "event":
            assert to_text(f["segments"], names) == to_text(segments(rec), names)
        else:
            assert "segments" not in f


def test_every_event_kind_can_be_narrated(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    asyncio.run(run(build_parser().parse_args(["--models", "random,random,random", "--seed", "8", "--log", str(path)])))
    kinds = set()
    for rec in read_feed(path):
        if rec["type"] == "event":
            kinds.add(rec["kind"])
            assert to_text(segments(rec), ["A", "B", "C"]).startswith(f"T{rec['turn']} ")
    assert {"hint", "play", "discard", "draw", "end"} <= kinds

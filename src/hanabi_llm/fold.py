"""Reference fold: rebuild the state of a game from its feed.

    state(t) = fold(step, initial_state(header), events[:t])

``step`` is a pure transition on feed records, independent of the engine: it
sees only what is in the feed. Two facts make this the heart of the model:

  1. the fold of the omniscient feed equals the engine's true state, and
  2. the fold of a player's PROJECTED feed equals that player's view
     (``views.build_view``): view(state(t)) = fold(project(events[:t])).

Both are tested on whole games (tests/test_fold.py). The web viewer is served
the folded states of this module, so it contains no game logic of its own.

A card whose identity is hidden has ``color`` and ``rank`` of ``None``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .feed import Record


@dataclass(slots=True)
class FoldCard:
    uid: int
    color: str | None
    rank: int | None
    known_color: str | None = None          # what hints told the holder
    known_rank: int | None = None
    not_colors: list[str] = field(default_factory=list)
    not_ranks: list[int] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"uid": self.uid, "color": self.color, "rank": self.rank, "known_color": self.known_color,
                "known_rank": self.known_rank, "not_colors": sorted(self.not_colors), "not_ranks": sorted(self.not_ranks)}


@dataclass(slots=True)
class FoldState:
    names: list[str]
    colors: list[str]
    top_rank: int
    max_hints: int
    max_fuses: int
    hands: list[list[FoldCard]]
    stacks: dict[str, int]
    discard: list[dict[str, Any]]
    hints: int
    fuses: int
    deck_size: int
    turn: int = 1
    current: int = 0
    final_turns_left: int | None = None
    over_reason: str | None = None
    last_actor: int = 0

    @property
    def score(self) -> int:
        return sum(self.stacks.values())

    def to_json(self) -> dict[str, Any]:
        return {
            "names": self.names, "colors": self.colors, "top_rank": self.top_rank,
            "hands": [[c.to_json() for c in hand] for hand in self.hands],
            "stacks": self.stacks, "discard": self.discard,
            "hints": {"available": self.hints, "capacity": self.max_hints, "spent": self.max_hints - self.hints},
            "fuses": {"available": self.fuses, "capacity": self.max_fuses, "spent": self.max_fuses - self.fuses},
            "deck_size": self.deck_size, "turn": self.turn, "current": self.current,
            "final_turns_left": self.final_turns_left, "over_reason": self.over_reason, "score": self.score,
        }


def initial_state(header: Record) -> FoldState:
    rules = header["rules"]
    return FoldState(
        names=list(header["names"]), colors=list(rules["colors"]), top_rank=rules["top_rank"],
        max_hints=rules["max_hints"], max_fuses=rules["fuses"],
        hands=[[FoldCard(c["uid"], c["color"], c["rank"]) for c in hand] for hand in header["deal"]["hands"]],
        stacks={c: 0 for c in rules["colors"]}, discard=[], hints=rules["initial_hints"], fuses=rules["fuses"],
        deck_size=header["deal"]["deck_size"],
    )


def _gain_hint(s: FoldState) -> None:
    s.hints = min(s.max_hints, s.hints + 1)


def _begin_action(s: FoldState, rec: Record) -> None:
    """Common bookkeeping for the three turn-taking events: the final-round clock and the next player."""
    if s.final_turns_left is not None:
        s.final_turns_left -= 1
    s.last_actor = rec["actor"]
    s.turn = rec["turn"] + 1
    s.current = (rec["actor"] + 1) % len(s.names)


def step(s: FoldState, rec: Record) -> None:
    """Apply one feed record to the state, in place."""
    if rec["type"] != "event":
        return
    kind = rec["kind"]
    if kind == "hint":
        _begin_action(s, rec)
        s.hints -= 1
        targets, value = set(rec["targets"]), rec["value"]
        for pos, card in enumerate(s.hands[rec["receiver"]]):
            hit = pos in targets  # a hint is complete: the targets ARE the value, every other card is NOT
            if rec["attribute"] == "color":
                if hit:
                    card.known_color = value
                elif value not in card.not_colors:
                    card.not_colors.append(value)
            elif hit:
                card.known_rank = value
            elif value not in card.not_ranks:
                card.not_ranks.append(value)
    elif kind in ("play", "discard"):
        _begin_action(s, rec)
        gone = s.hands[rec["actor"]].pop(rec["position"])
        if gone.uid != rec["card"]["uid"]:
            raise ValueError(f"feed inconsistent: position {rec['position']} held card {gone.uid}, event says {rec['card']['uid']}")
        card = rec["card"]  # a played or discarded card is revealed to everyone
        if kind == "play" and rec["success"]:
            s.stacks[card["color"]] = card["rank"]
            if card["rank"] == s.top_rank:
                _gain_hint(s)
        else:
            s.discard.append(card)
            if kind == "play":
                s.fuses -= 1
            else:
                _gain_hint(s)
    elif kind == "draw":
        c = rec["card"]
        s.hands[rec["actor"]].append(FoldCard(c["uid"], c["color"], c["rank"]))
        s.deck_size -= 1
        if s.deck_size == 0 and s.final_turns_left is None:
            s.final_turns_left = len(s.names)  # the player who drew the last card also gets a final turn
    elif kind == "end":
        s.over_reason = rec["reason"]
        s.turn, s.current = rec["turn"], s.last_actor  # the engine does not advance the turn when the game ends
    else:
        raise ValueError(f"unknown event kind {kind!r}")


def fold(records: Iterable[Record]) -> FoldState:
    """Fold a feed (header first) into the state after its last record."""
    it = iter(records)
    header = next(it)
    if header["type"] != "header":
        raise ValueError("a feed starts with a header record")
    state = initial_state(header)
    for rec in it:
        step(state, rec)
    return state

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

This is also the engine's own state machine: ``Game.apply`` decides what
happens (validates, resolves a play, draws from the secret deck), turns that
into events, and applies them with ``step``. There is no second copy of what a
hint, a play, a discard or a draw does to the state. The rule functions the
ontology binds by name (refund, playable, game over, final round) live here too.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .feed import Record
from .registry import bind


@dataclass(slots=True)
class FoldCard:
    color: str | None  # None = the identity is hidden from whoever this state is for
    rank: int | None
    known_color: str | None = None          # what hints told the holder
    known_rank: int | None = None
    not_colors: list[str] = field(default_factory=list)
    not_ranks: list[int] = field(default_factory=list)

    @classmethod
    def from_record(cls, card: dict[str, Any] | None) -> FoldCard:
        """A card from a feed record: ``{"color", "rank"}``, or ``None`` / a null colour when hidden."""
        if card is None or card.get("color") is None:
            return cls(None, None)
        return cls(card["color"], card["rank"])

    def to_json(self) -> dict[str, Any]:
        return {"color": self.color, "rank": self.rank, "known_color": self.known_color,
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
    #: Cards placed on the fireworks, in play order (the truth must account for all of them).
    played: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def dealt(cls, names: list[str], colors: list[str], top_rank: int, max_hints: int, initial_hints: int, fuses: int,
              hands: list[list[FoldCard]], deck_size: int) -> FoldState:
        """The state right after the deal."""
        return cls(names=names, colors=colors, top_rank=top_rank, max_hints=max_hints, max_fuses=fuses, hands=hands,
                   stacks={c: 0 for c in colors}, discard=[], hints=initial_hints, fuses=fuses, deck_size=deck_size)

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
    return FoldState.dealt(
        names=list(header["names"]), colors=list(rules["colors"]), top_rank=rules["top_rank"], max_hints=rules["max_hints"],
        initial_hints=rules["initial_hints"], fuses=rules["fuses"],
        hands=[[FoldCard.from_record(c) for c in hand] for hand in header["deal"]["hands"]],
        deck_size=header["deal"]["deck_size"],
    )


# --- the rules the ontology binds by name ------------------------------------------------------------------


@bind("refund_hint")
def refund_hint(s: FoldState) -> None:
    """Discarding, or completing a stack with its top rank, returns a hint token (never above the maximum)."""
    s.hints = min(s.max_hints, s.hints + 1)


@bind("play_is_legal_or_misfire")
def is_playable(s: FoldState, color: str, rank: int) -> bool:
    """A card is placed if it is the next one its colour needs; otherwise it is a misfire."""
    return s.stacks[color] == rank - 1


@bind("final_round_done")
def final_round_done(s: FoldState) -> bool:
    return s.final_turns_left == 0


@bind("game_is_over")
def game_is_over(s: FoldState) -> str | None:
    """The reason the game has ended, or ``None`` if it continues."""
    if s.fuses <= 0:
        return "all fuse tokens were used up"
    if all(h == s.top_rank for h in s.stacks.values()):
        return "all fireworks are complete"
    if final_round_done(s):
        return "the final round after the deck ran out is over"
    return None


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
        shown = {"color": rec["card"]["color"], "rank": rec["card"]["rank"]}  # a played or discarded card is revealed to everyone
        if gone.color is not None and (gone.color, gone.rank) != (shown["color"], shown["rank"]):
            raise ValueError(f"feed inconsistent: position {rec['position']} held {gone.color} {gone.rank}, the event says {shown['color']} {shown['rank']}")
        if kind == "play" and rec["success"]:
            s.played.append(shown)
            s.stacks[shown["color"]] = shown["rank"]
            if shown["rank"] == s.top_rank:
                refund_hint(s)
        else:
            s.discard.append(shown)
            if kind == "play":
                s.fuses -= 1
            else:
                refund_hint(s)
    elif kind == "draw":
        s.hands[rec["actor"]].append(FoldCard.from_record(rec["card"]))
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

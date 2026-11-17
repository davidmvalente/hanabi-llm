"""Plain, immutable value types shared by every layer.

Nothing in here knows about RDF, HTTP or LLMs. The ontology (``hanabi.ttl``)
describes the same concepts declaratively; ``views.py`` is the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

#: Seat index of a player (0-based). Names are display-only.
Seat = int


class Color(StrEnum):
    """Firework colours. Values match ``hn:colorName`` in the ontology."""

    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    WHITE = "white"


@dataclass(frozen=True, slots=True)
class Card:
    """A physical card. ``uid`` is stable for the whole game."""

    uid: int
    color: Color
    rank: int

    def __str__(self) -> str:
        return f"{self.color.value} {self.rank}"


@dataclass(slots=True)
class CardKnowledge:
    """What is *publicly known* about one card, purely from hints given.

    Every player can compute this for every card, so it is common knowledge
    and safe to show to anyone (including the card's holder).
    """

    color: Color | None = None
    rank: int | None = None
    not_colors: set[Color] = field(default_factory=set)
    not_ranks: set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Actions (what a player can attempt). Each carries the ontology function name
# of the affordance that offers it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayCard:
    affordance: ClassVar[str] = "play_card"
    position: int


@dataclass(frozen=True, slots=True)
class DiscardCard:
    affordance: ClassVar[str] = "discard_card"
    position: int


@dataclass(frozen=True, slots=True)
class GiveHint:
    affordance: ClassVar[str] = "give_hint"
    target: Seat
    color: Color | None = None
    rank: int | None = None


Action = PlayCard | DiscardCard | GiveHint


# ---------------------------------------------------------------------------
# Events (what happened). The public log is a list of these.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayEvent:
    turn: int
    actor: Seat
    position: int
    card: Card
    success: bool


@dataclass(frozen=True, slots=True)
class DiscardEvent:
    turn: int
    actor: Seat
    position: int
    card: Card


@dataclass(frozen=True, slots=True)
class HintEvent:
    turn: int
    actor: Seat
    target: Seat
    color: Color | None
    rank: int | None
    touched: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DrawEvent:
    """``actor`` drew ``card`` into their last slot.

    The *drawer* must not see ``card``; views redact it for them.
    """

    turn: int
    actor: Seat
    card: Card


@dataclass(frozen=True, slots=True)
class EndEvent:
    turn: int
    reason: str
    score: int


Event = PlayEvent | DiscardEvent | HintEvent | DrawEvent | EndEvent


class IllegalAction(Exception):
    """Raised by the engine when an action violates a rule.

    The message is written for the model: it says what was wrong and why.
    """

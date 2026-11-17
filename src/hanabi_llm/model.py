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


#: What a hint names: a colour (``Color``) or a rank (``int``). The two kinds are disjoint,
#: so one field says both WHICH attribute and its value, and "neither" or "both" cannot be written.
HintValue = Color | int


def hint_attribute(value: HintValue) -> str:
    """"color" or "rank"."""
    return "color" if isinstance(value, Color) else "rank"


@dataclass(frozen=True, slots=True)
class GiveHint:
    """Tell ``receiver`` which cards of their hand are ``value``.

    The hint always covers EVERY card of the receiver's hand with that colour
    or rank, so what it targets is a subset of the hand (see ``HintEvent.targets``)
    and it also tells the receiver that all the other cards are not ``value``.
    """

    affordance: ClassVar[str] = "give_hint"
    receiver: Seat
    value: HintValue

    @property
    def attribute(self) -> str:
        return hint_attribute(self.value)


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
    """``actor`` told ``receiver`` that the cards at ``targets`` are ``value`` (and the rest are not).

    ``targets`` is the subset of the receiver's hand the hint covers, as positions
    at the moment of the hint: exactly the cards that have ``value``, no more, no fewer.
    """

    turn: int
    actor: Seat
    receiver: Seat
    value: HintValue
    targets: tuple[int, ...]

    @property
    def attribute(self) -> str:
        return hint_attribute(self.value)


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

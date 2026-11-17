"""Candidate actions as plain strings, for models that rank documents instead of calling tools.

A rerank model scores a list of documents against a query. Here the documents are the actions
the current player may take, written as short sentences:

    play the card at position 2
    discard the card at position 0
    hint Bob: green
    hint Carol: rank 3

Two candidate sets:

  legal  only actions the rules allow now (a hint must target at least one card, needs a token).
  all    every action the game has a name for: every position to play or discard, and every
         receiver x value combination of hint, whether or not it could be given. Choosing an
         impossible one is a rejected action.

The wording is the same in both sets and never says what a hint would target, so the text cannot
reveal which hints are legal. Nothing here reads hidden information: the candidates depend only
on the player's own hand SIZE, the other players' (visible) hands, and the token counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .model import Action, Color, DiscardCard, GiveHint, PlayCard
from .ontology import RuleSet
from .views import PlayerView

CANDIDATE_SETS = ("legal", "all")


def action_text(action: Action, names: Sequence[str]) -> str:
    if isinstance(action, PlayCard):
        return f"play the card at position {action.position}"
    if isinstance(action, DiscardCard):
        return f"discard the card at position {action.position}"
    value = action.value.value if isinstance(action.value, Color) else f"rank {action.value}"
    return f"hint {names[action.receiver]}: {value}"


def candidate_actions(view: PlayerView, rules: RuleSet, legal: Sequence[Action], which: str = "legal") -> list[Action]:
    """The candidates in a fixed canonical order: plays, discards, then hints by receiver, colour, rank."""
    if which not in CANDIDATE_SETS:
        raise ValueError(f"candidates must be one of {CANDIDATE_SETS}")
    own = len(view.hands[view.viewer])
    every: list[Action] = []
    every += [PlayCard(p) for p in range(own)]
    every += [DiscardCard(p) for p in range(own)]
    for receiver in range(len(view.names)):
        if receiver == view.viewer:
            continue
        every += [GiveHint(receiver, c) for c in rules.colors]
        every += [GiveHint(receiver, r) for r in sorted(rules.copies_by_rank)]
    if which == "all":
        return every
    allowed = set(legal)
    return [a for a in every if a in allowed]


@dataclass(frozen=True, slots=True)
class ActionSpace:
    """How many candidate actions a player faces, as a function of the number of players."""

    players: int
    hand_size: int
    plays_and_discards: int  # 2 x hand size
    hints_all: int           # every receiver x value
    total_all: int
    hints_max_legal: int     # the most hints that can touch a card
    total_max_legal: int


def action_space(rules: RuleSet, players: int) -> ActionSpace:
    """The size of the candidate set for ``players`` players (see docs/RERANK.md)."""
    h = rules.hand_size_by_players[players]
    values = len(rules.colors) + len(rules.copies_by_rank)               # 5 colours + 5 ranks
    # A hand of h cards shows at most min(h, colours) distinct colours and min(h, ranks) distinct ranks.
    legal_per_receiver = min(h, len(rules.colors)) + min(h, len(rules.copies_by_rank))
    return ActionSpace(
        players=players, hand_size=h, plays_and_discards=2 * h,
        hints_all=(players - 1) * values, total_all=2 * h + (players - 1) * values,
        hints_max_legal=(players - 1) * legal_per_receiver, total_max_legal=2 * h + (players - 1) * legal_per_receiver,
    )

"""Authoritative game engine.

The engine DECIDES; the fold APPLIES. ``Game.apply`` validates an action with the
precondition functions the ontology names, decides what happens (resolves a
play, draws from the secret deck, works out which cards a hint covers), turns
that into events, and applies them with ``fold.step``. There is exactly one
implementation of what a hint, a play, a discard or a draw does to the state
(``fold.py``), shared with the feed, the viewer and the tests.

What lives here and nowhere else:
  * the secret deck and its order (it is outside the fold: nobody may see it),
  * validation of actions, and the legal-action list,
  * the decisions (``decide_*``): a play succeeds if the card is the next one
    its colour needs; a hint covers every card of the hand with that value.

``game.state`` is a read-only typed snapshot of the fold, rebuilt after every
action. It cannot be mutated (tuples and read-only mappings), so a mistake
raises instead of silently diverging from the truth. To set up a scenario use
``Game.rig``.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from .feed import event_record
from .fold import FoldCard, FoldState, is_playable, step
from .model import (
    Action,
    Card,
    CardKnowledge,
    Color,
    DiscardCard,
    DiscardEvent,
    DrawEvent,
    EndEvent,
    Event,
    GiveHint,
    HintEvent,
    HintValue,
    IllegalAction,
    PlayCard,
    PlayEvent,
    Seat,
)
from .ontology import AffordanceSpec, Ontology, RuleSet, load_ontology, verify_bindings
from .registry import FUNCTIONS, bind


@dataclass(frozen=True, slots=True)
class GameState:
    """A read-only snapshot of everything about a game, hidden information included."""

    names: list[str]
    hands: tuple[tuple[Card, ...], ...]
    deck: tuple[Card, ...]  # the top of the deck is the last element
    discard: tuple[Card, ...]
    stacks: Mapping[Color, int]
    hints: int
    fuses: int
    current: Seat
    turn: int
    #: ``None`` until the deck runs out, then counts down remaining turns.
    final_turns_left: int | None
    #: Cards placed on the fireworks, in play order.
    played: tuple[Card, ...]
    #: What hints have told the holder about each card still in a hand.
    knowledge: Mapping[int, CardKnowledge]
    events: tuple[Event, ...]
    over_reason: str | None

    @property
    def score(self) -> int:
        """Sum of the stack heights."""
        return sum(self.stacks.values())

    @property
    def lost(self) -> bool:
        """True if the team blew all its fuses (official score is then 0)."""
        return self.over_reason == "all fuse tokens were used up"


def hint_targets(hand: Sequence[Card | FoldCard], value: HintValue) -> tuple[int, ...]:
    """The positions a hint of ``value`` covers: EVERY card of the hand with that colour or rank."""
    if isinstance(value, Color):
        return tuple(i for i, c in enumerate(hand) if c.color == value)
    return tuple(i for i, c in enumerate(hand) if c.rank == value)


# ---------------------------------------------------------------------------
# Preconditions. Signature: (state, rules, actor, action) -> None | raises.
# Names match hn:checkFunction in hanabi.ttl.
# ---------------------------------------------------------------------------


@bind("card_in_hand")
def _card_in_hand(state: GameState, rules: RuleSet, actor: Seat, action: Action) -> None:
    if isinstance(action, GiveHint):
        return
    size = len(state.hands[actor])
    if not 0 <= action.position < size:
        raise IllegalAction(
            f"position {action.position} is not in your hand. "
            f"Your hand has {size} cards, positions 0 to {size - 1}."
        )


@bind("hint_one_attribute")
def _hint_one_attribute(state: GameState, rules: RuleSet, actor: Seat, action: Action) -> None:
    """A hint names ONE attribute. The encoding (a single ``value``) already rules out "both" and "neither";
    what is left to check is that the value is a real colour or rank."""
    if not isinstance(action, GiveHint):
        return
    value = action.value
    if isinstance(value, Color):
        if value not in rules.colors:
            raise IllegalAction(f"Unknown colour {value!r}.")
    elif isinstance(value, bool) or value not in rules.copies_by_rank:
        raise IllegalAction(f"Unknown hint {value!r}: name a colour or a rank in {sorted(rules.copies_by_rank)}.")


@bind("hint_receiver_is_other")
def _hint_receiver_is_other(state: GameState, rules: RuleSet, actor: Seat, action: Action) -> None:
    if not isinstance(action, GiveHint):
        return
    if not 0 <= action.receiver < len(state.names):
        raise IllegalAction("Unknown player to hint.")
    if action.receiver == actor:
        raise IllegalAction("You cannot give a hint to yourself.")


@bind("hint_token_available")
def _hint_token_available(state: GameState, rules: RuleSet, actor: Seat, action: Action) -> None:
    if isinstance(action, GiveHint) and state.hints <= 0:
        raise IllegalAction("No hint tokens left. Play or discard a card instead; discarding returns a hint token.")


@bind("hint_targets_cards")
def _hint_targets_cards(state: GameState, rules: RuleSet, actor: Seat, action: Action) -> None:
    if not isinstance(action, GiveHint):
        return
    if not hint_targets(state.hands[action.receiver], action.value):
        what = action.value.value if isinstance(action.value, Color) else str(action.value)
        raise IllegalAction(
            f"{state.names[action.receiver]} holds no card matching '{what}', so this hint would reveal nothing. "
            "A hint must target at least one card."
        )


# ---------------------------------------------------------------------------
# Decisions. Signature: (game, actor, action) -> list[Event].
# They only DECIDE and build events (drawing from the secret deck is the one
# thing they do); ``Game.apply`` applies the events with ``fold.step``.
# ---------------------------------------------------------------------------


def _card(fc: FoldCard) -> Card:
    assert fc.color is not None and fc.rank is not None, "the engine's own state always knows every identity"
    return Card(fc.uid, Color(fc.color), fc.rank)


def _draw_event(game: Game, actor: Seat, turn: int) -> list[Event]:
    """A refill for the actor's hand, if the deck has a card; new cards always go into the LAST slot."""
    return [DrawEvent(turn, actor, game._deck.pop())] if game._deck else []


@bind("decide_play_card")
def _decide_play_card(game: Game, actor: Seat, action: Action) -> list[Event]:
    assert isinstance(action, PlayCard)
    fold = game._fold
    card = _card(fold.hands[actor][action.position])
    success = is_playable(fold, card.color, card.rank)  # the rule "play_is_legal_or_misfire"
    return [PlayEvent(fold.turn, actor, action.position, card, success), *_draw_event(game, actor, fold.turn)]


@bind("decide_discard_card")
def _decide_discard_card(game: Game, actor: Seat, action: Action) -> list[Event]:
    assert isinstance(action, DiscardCard)
    fold = game._fold
    card = _card(fold.hands[actor][action.position])
    return [DiscardEvent(fold.turn, actor, action.position, card), *_draw_event(game, actor, fold.turn)]


@bind("decide_give_hint")
def _decide_give_hint(game: Game, actor: Seat, action: Action) -> list[Event]:
    assert isinstance(action, GiveHint)
    targets = hint_targets(game._fold.hands[action.receiver], action.value)  # complete: every matching card
    return [HintEvent(game._fold.turn, actor, action.receiver, action.value, targets)]


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------


class Game:
    """One game of Hanabi. Not thread-safe; drive it from a single task."""

    def __init__(
        self,
        player_names: list[str],
        seed: int | None = None,
        ontology: Ontology | None = None,
    ) -> None:
        self.ontology = ontology or load_ontology()
        verify_bindings(self.ontology, FUNCTIONS)
        rules = self.ontology.rules
        n = len(player_names)
        if n not in rules.hand_size_by_players:
            raise ValueError(f"unsupported player count {n}; supported: {sorted(rules.hand_size_by_players)}")
        if len(set(player_names)) != n:
            raise ValueError("player names must be unique")

        # Shuffle FIRST, then number the cards in shuffled order. If ids were
        # assigned before shuffling they would be sorted by colour and rank,
        # and a model could read its own cards' identities off their ids.
        identities = [
            (color, rank)
            for color in rules.colors
            for rank, copies in sorted(rules.copies_by_rank.items())
            for _ in range(copies)
        ]
        random.Random(seed).shuffle(identities)
        deck = [Card(uid, color, rank) for uid, (color, rank) in enumerate(identities)]
        self.seed = seed

        hand_size = rules.hand_size_by_players[n]
        hands = [[deck.pop() for _ in range(hand_size)] for _ in range(n)]
        #: The secret: the cards still to be drawn. Not part of the fold, so no view can ever contain it.
        self._deck: list[Card] = deck
        #: The truth: the one place the game state lives, changed only by ``fold.step``.
        self._fold = FoldState.dealt(
            names=list(player_names), colors=[c.value for c in rules.colors], top_rank=rules.top_rank,
            max_hints=rules.max_hints, initial_hints=rules.initial_hints, fuses=rules.initial_fuses,
            hands=[[FoldCard(c.uid, c.color.value, c.rank) for c in hand] for hand in hands], deck_size=len(deck),
        )
        self._events: list[Event] = []
        self._snapshot: GameState | None = None

    @property
    def rules(self) -> RuleSet:
        return self.ontology.rules

    @property
    def is_over(self) -> bool:
        return self._fold.over_reason is not None

    @property
    def state(self) -> GameState:
        """A read-only typed snapshot of the truth (rebuilt only after the game changes)."""
        if self._snapshot is None:
            f = self._fold
            self._snapshot = GameState(
                names=list(f.names),
                hands=tuple(tuple(_card(c) for c in hand) for hand in f.hands),
                deck=tuple(self._deck),
                discard=tuple(Card(c["uid"], Color(c["color"]), c["rank"]) for c in f.discard),
                stacks=MappingProxyType({Color(c): h for c, h in f.stacks.items()}),
                hints=f.hints, fuses=f.fuses, current=f.current, turn=f.turn, final_turns_left=f.final_turns_left,
                played=tuple(Card(c["uid"], Color(c["color"]), c["rank"]) for c in f.played),
                knowledge=MappingProxyType({
                    c.uid: CardKnowledge(
                        color=Color(c.known_color) if c.known_color else None, rank=c.known_rank,
                        not_colors=frozenset(Color(x) for x in c.not_colors), not_ranks=frozenset(c.not_ranks))
                    for hand in f.hands for c in hand}),
                events=tuple(self._events), over_reason=f.over_reason,
            )
        return self._snapshot

    # -- scenario setup -----------------------------------------------------

    def rig(
        self,
        *,
        hands: Sequence[Sequence[Card]] | None = None,
        deck: Sequence[Card] | None = None,
        stacks: Mapping[Color, int] | None = None,
        hints: int | None = None,
        fuses: int | None = None,
        knowledge: Mapping[int, CardKnowledge] | None = None,
    ) -> None:
        """Set up a position directly, for tests and for putting a decision-maker on a fixed scenario.

        This bypasses the rules, so it can create positions the rules never would (for
        example hands that do not add up to a real deck); the SHACL audit rejects those.
        The top of ``deck`` is its last element. ``knowledge`` is applied to cards in hands.
        """
        f = self._fold
        if hands is not None:
            f.hands = [[FoldCard(c.uid, c.color.value, c.rank) for c in hand] for hand in hands]
        if deck is not None:
            self._deck = list(deck)
            f.deck_size = len(self._deck)
        for color, height in (stacks or {}).items():
            f.stacks[color.value] = height
        if hints is not None:
            f.hints = hints
        if fuses is not None:
            f.fuses = fuses
        if knowledge:
            held = {c.uid: c for hand in f.hands for c in hand}
            for uid, k in knowledge.items():
                c = held[uid]
                c.known_color, c.known_rank = (k.color.value if k.color else None), k.rank
                c.not_colors, c.not_ranks = [x.value for x in k.not_colors], list(k.not_ranks)
        self._snapshot = None

    # -- legality -----------------------------------------------------------

    def legal_actions(self, actor: Seat) -> list[Action]:
        """All actions ``actor`` could legally take right now."""
        st = self.state
        if self.is_over or actor != st.current:
            return []
        actions: list[Action] = []
        for pos in range(len(st.hands[actor])):
            actions.append(PlayCard(pos))
            actions.append(DiscardCard(pos))
        if st.hints > 0:
            for receiver, hand in enumerate(st.hands):
                if receiver == actor:
                    continue
                for color in self.rules.colors:
                    if hint_targets(hand, color):
                        actions.append(GiveHint(receiver, color))
                for rank in sorted(self.rules.copies_by_rank):
                    if hint_targets(hand, rank):
                        actions.append(GiveHint(receiver, rank))
        return actions

    def check(self, actor: Seat, action: Action) -> AffordanceSpec:
        """Validate ``action`` without changing anything.

        Runs the precondition functions named by the ontology for this
        affordance. Raises ``IllegalAction`` (message written for the model).
        Returns the affordance spec so ``apply`` need not look it up again.
        """
        st = self.state
        if self.is_over:
            raise IllegalAction("The game is over.")
        if actor != st.current:
            raise IllegalAction(f"It is {st.names[st.current]}'s turn, not yours.")
        spec = self.ontology.affordances[action.affordance]
        for check in spec.required_checks:
            FUNCTIONS[check](st, self.rules, actor, action)
        return spec

    # -- the one entry point that changes the game --------------------------

    def apply(self, actor: Seat, action: Action) -> list[Event]:
        """Validate ``action``, decide what happens, and fold the events into the state.

        Raises ``IllegalAction`` if illegal. On success returns the events produced.
        """
        spec = self.check(actor, action)
        events = cast(list[Event], FUNCTIONS[spec.effect_function](self, actor, action))
        turn = events[0].turn
        for ev in events:
            step(self._fold, event_record(ev))  # the ONLY place the state changes
        reason = FUNCTIONS["game_is_over"](self._fold)
        if reason is not None:
            end = EndEvent(turn, reason, self._fold.score)
            events.append(end)
            step(self._fold, event_record(end))
        self._events.extend(events)
        self._snapshot = None
        return events

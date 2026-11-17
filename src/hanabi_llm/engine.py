"""Authoritative game engine.

Responsibilities:
  * hold the one true game state (including every hidden card),
  * enumerate legal actions,
  * enforce rules by running the precondition functions *named by the
    ontology* for the chosen affordance, then its named effect function,
  * produce an immutable event log.

The engine knows nothing about LLMs or RDF views. It trusts nobody: any
action, from any source, goes through ``Game.apply``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import cast

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


@dataclass(slots=True)
class GameState:
    """Everything about a game, hidden information included."""

    names: list[str]
    hands: list[list[Card]]
    deck: list[Card]
    discard: list[Card]
    stacks: dict[Color, int]
    hints: int
    fuses: int
    current: Seat = 0
    turn: int = 1
    #: ``None`` until the deck runs out, then counts down remaining turns.
    final_turns_left: int | None = None
    #: Cards placed on the fireworks, in play order. Kept so the ground truth accounts for every card.
    played: list[Card] = field(default_factory=list)
    knowledge: dict[int, CardKnowledge] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    over_reason: str | None = None

    @property
    def score(self) -> int:
        """Sum of the stack heights."""
        return sum(self.stacks.values())

    @property
    def lost(self) -> bool:
        """True if the team blew all its fuses (official score is then 0)."""
        return self.over_reason == "all fuse tokens were used up"


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


def hint_targets(hand: list[Card], value: HintValue) -> tuple[int, ...]:
    """The positions a hint of ``value`` covers: EVERY card of the hand with that colour or rank."""
    if isinstance(value, Color):
        return tuple(i for i, c in enumerate(hand) if c.color == value)
    return tuple(i for i, c in enumerate(hand) if c.rank == value)


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
# Predicates and shared effects used by several affordances.
# ---------------------------------------------------------------------------


@bind("play_is_legal_or_misfire")
def is_playable(state: GameState, card: Card) -> bool:
    """True if ``card`` is the next one needed on its colour's stack."""
    return state.stacks[card.color] == card.rank - 1


@bind("refund_hint")
def _refund_hint(state: GameState, rules: RuleSet) -> None:
    state.hints = min(rules.max_hints, state.hints + 1)


@bind("final_round_done")
def _final_round_done(state: GameState) -> bool:
    return state.final_turns_left == 0


@bind("game_is_over")
def _game_is_over(state: GameState, rules: RuleSet) -> str | None:
    """Return the reason the game ended, or ``None`` if it continues."""
    if state.fuses <= 0:
        return "all fuse tokens were used up"
    if all(h == rules.top_rank for h in state.stacks.values()):
        return "all fireworks are complete"
    if _final_round_done(state):
        return "the final round after the deck ran out is over"
    return None


def _draw(state: GameState, actor: Seat) -> DrawEvent | None:
    """Refill the actor's hand. New cards always go into the *last* slot."""
    if not state.deck:
        return None
    card = state.deck.pop()
    state.hands[actor].append(card)
    state.knowledge[card.uid] = CardKnowledge()
    return DrawEvent(state.turn, actor, card)


# ---------------------------------------------------------------------------
# Effects. Signature: (state, rules, actor, action) -> list[Event].
# Effects mutate state but do NOT append to state.events; Game.apply does.
# ---------------------------------------------------------------------------


@bind("apply_play_card")
def _apply_play_card(state: GameState, rules: RuleSet, actor: Seat, action: Action) -> list[Event]:
    assert isinstance(action, PlayCard)
    card = state.hands[actor].pop(action.position)
    success = is_playable(state, card)
    if success:
        state.stacks[card.color] += 1
        state.played.append(card)
        if card.rank == rules.top_rank:
            FUNCTIONS["refund_hint"](state, rules)
    else:
        state.discard.append(card)
        state.fuses -= 1
    events: list[Event] = [PlayEvent(state.turn, actor, action.position, card, success)]
    if (draw := _draw(state, actor)) is not None:
        events.append(draw)
    return events


@bind("apply_discard_card")
def _apply_discard_card(state: GameState, rules: RuleSet, actor: Seat, action: Action) -> list[Event]:
    assert isinstance(action, DiscardCard)
    card = state.hands[actor].pop(action.position)
    state.discard.append(card)
    FUNCTIONS["refund_hint"](state, rules)
    events: list[Event] = [DiscardEvent(state.turn, actor, action.position, card)]
    if (draw := _draw(state, actor)) is not None:
        events.append(draw)
    return events


@bind("apply_give_hint")
def _apply_give_hint(state: GameState, rules: RuleSet, actor: Seat, action: Action) -> list[Event]:
    assert isinstance(action, GiveHint)
    state.hints -= 1
    hand = state.hands[action.receiver]
    targets = hint_targets(hand, action.value)
    value = action.value
    for position, card in enumerate(hand):
        know = state.knowledge[card.uid]
        if position in targets:  # the cards the hint covers ARE this colour / rank ...
            if isinstance(value, Color):
                know.color = value
            else:
                know.rank = value
        elif isinstance(value, Color):  # ... and, because a hint is complete, every other card is NOT
            know.not_colors.add(value)
        else:
            know.not_ranks.add(value)
    return [HintEvent(state.turn, actor, action.receiver, value, targets)]


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
        self.state = GameState(
            names=list(player_names),
            hands=hands,
            deck=deck,
            discard=[],
            stacks={c: 0 for c in rules.colors},
            hints=rules.initial_hints,
            fuses=rules.initial_fuses,
        )
        self.state.knowledge = {c.uid: CardKnowledge() for hand in hands for c in hand}

    @property
    def rules(self) -> RuleSet:
        return self.ontology.rules

    @property
    def is_over(self) -> bool:
        return self.state.over_reason is not None

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

    # -- the one entry point that mutates state -----------------------------

    def apply(self, actor: Seat, action: Action) -> list[Event]:
        """Validate and execute ``action``. Raises ``IllegalAction`` if illegal.

        On success returns the events produced (also appended to the log).
        """
        st = self.state
        spec = self.check(actor, action)

        was_final = st.final_turns_left is not None
        events = cast(list[Event], FUNCTIONS[spec.effect_function](st, self.rules, actor, action))

        # Deck-exhaustion clock: the turn that draws the last card starts a
        # final round in which every player (that drawer included) acts once.
        if was_final:
            assert st.final_turns_left is not None
            st.final_turns_left -= 1
        elif not st.deck:
            st.final_turns_left = len(st.names)

        st.over_reason = FUNCTIONS["game_is_over"](st, self.rules)
        if st.over_reason is not None:
            events.append(EndEvent(st.turn, st.over_reason, st.score))
        else:
            st.current = (st.current + 1) % len(st.names)
            st.turn += 1
        st.events.extend(events)
        return events

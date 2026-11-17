"""Helpers to build games with a rigged, known deal."""

from __future__ import annotations

import pytest

from hanabi_llm.engine import Game
from hanabi_llm.model import Card, CardKnowledge, Color

R, Y, G, B, W = Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE, Color.WHITE


def rig(game: Game, hands: list[list[tuple[Color, int]]], deck: list[tuple[Color, int]] | None = None) -> Game:
    """Overwrite hands (and optionally the deck; top of deck = last element)."""
    uid = 1000
    st = game.state
    st.knowledge.clear()
    new_hands: list[list[Card]] = []
    for hand in hands:
        cards = []
        for color, rank in hand:
            cards.append(Card(uid, color, rank))
            st.knowledge[uid] = CardKnowledge()
            uid += 1
        new_hands.append(cards)
    st.hands = new_hands
    if deck is not None:
        st.deck = [Card(uid + i, c, r) for i, (c, r) in enumerate(deck)]
    return game


@pytest.fixture
def game2() -> Game:
    return Game(["Ann", "Bob"], seed=0)

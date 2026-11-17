"""Helpers to build games with a rigged, known deal."""

from __future__ import annotations

import pytest

from hanabi_llm.engine import Game
from hanabi_llm.model import Card, Color

R, Y, G, B, W = Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE, Color.WHITE


def rig(game: Game, hands: list[list[tuple[Color, int]]], deck: list[tuple[Color, int]] | None = None) -> Game:
    """Overwrite hands (and optionally the deck; top of deck = last element)."""
    uid = 1000
    new_hands: list[list[Card]] = []
    for hand in hands:
        new_hands.append([Card(uid + i, color, rank) for i, (color, rank) in enumerate(hand)])
        uid += len(hand)
    game.rig(hands=new_hands, deck=None if deck is None else [Card(uid + i, c, r) for i, (c, r) in enumerate(deck)])
    return game


@pytest.fixture
def game2() -> Game:
    return Game(["Ann", "Bob"], seed=0)

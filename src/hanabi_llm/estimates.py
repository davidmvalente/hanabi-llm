"""Card-counting estimates computed purely from one player's own view.

This is the "collision rate" of the game: given what a player can see, how
likely is one of their own cards to be each possible identity? It uses only
information already in the ``PlayerView``, so it cannot leak anything.

Approximation: each card is treated independently (the joint constraints
between a player's own cards are ignored). Good enough to guide play, and it
is the same for every player.
"""

from __future__ import annotations

from collections import Counter

from .model import Color, IllegalAction
from .ontology import RuleSet
from .views import PlayerView


def unseen_counts(view: PlayerView, rules: RuleSet) -> Counter[tuple[Color, int]]:
    """Copies of each (colour, rank) the viewer cannot account for."""
    remaining: Counter[tuple[Color, int]] = Counter()
    for color in rules.colors:
        for rank, copies in rules.copies_by_rank.items():
            remaining[(color, rank)] = copies
    for card in view.discard:
        remaining[(card.color, card.rank)] -= 1
    for color, height in view.stacks.items():
        for rank in range(1, height + 1):
            remaining[(color, rank)] -= 1  # one copy of each played rank is on the stack
    for seat, hand in enumerate(view.hands):
        if seat == view.viewer:
            continue
        for cv in hand:
            assert cv.card is not None
            remaining[(cv.card.color, cv.card.rank)] -= 1
    return remaining


def estimate_card(view: PlayerView, rules: RuleSet, position: int) -> str:
    """Describe the distribution over identities for the viewer's card at ``position``."""
    hand = view.hands[view.viewer]
    if not 0 <= position < len(hand):
        raise IllegalAction(f"position {position} is not in your hand (positions 0 to {len(hand) - 1}).")
    k = hand[position].knowledge
    pool = unseen_counts(view, rules)

    candidates: dict[tuple[Color, int], int] = {}
    for (color, rank), n in pool.items():
        if n <= 0:
            continue
        if k.color is not None and color != k.color:
            continue
        if k.rank is not None and rank != k.rank:
            continue
        if color in k.not_colors or rank in k.not_ranks:
            continue
        candidates[(color, rank)] = n
    total = sum(candidates.values())
    if total == 0:
        return "No identity is consistent with what you know; the hints and the visible cards contradict each other."

    playable = sum(n for (c, r), n in candidates.items() if view.stacks[c] == r - 1)
    useless = sum(n for (c, r), n in candidates.items() if r <= view.stacks[c])
    top_rank = sum(n for (_, r), n in candidates.items() if r == rules.top_rank)
    ranked = sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0][0].value, kv[0][1]))
    lines = [
        f"Position {position}: {total} candidate copies unseen and consistent with your hints.",
        f"  P(playable now) = {playable / total:.0%}",
        f"  P(already played / useless) = {useless / total:.0%}",
        f"  P(is a {rules.top_rank}) = {top_rank / total:.0%}",
        "  Most likely identities:",
    ]
    for (color, rank), n in ranked[:8]:
        lines.append(f"    {color.value} {rank}: {n / total:.0%} ({n}/{total})")
    if len(ranked) > 8:
        lines.append(f"    ... and {len(ranked) - 8} more")
    return "\n".join(lines)

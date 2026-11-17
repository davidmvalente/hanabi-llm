"""The game state as ONE player may see it, in six sections that partition it.

    turn       whose turn, turn number, cards left in the deck, final round, game over
    tokens     hint tokens and fuse tokens, both as available / spent / capacity
    fireworks  the five stacks, the next card each needs, the score
    discard    the public discard pile
    hands      every hand: teammates' cards, and what each hand has been told
    history    the most recent events

Mutually exclusive and exhaustive: every field of ``PlayerView`` is rendered in
exactly one section, and the full text is just the sections in order. All of
it is computed from a ``PlayerView`` that is already redacted for the viewer,
so none of it can reveal a hidden card.

Wording: a token pool is always "available / spent / capacity" with
``spent = capacity - available``. Refunds need no bookkeeping: a refund simply
raises ``available`` again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .model import IllegalAction
from .ontology import RuleSet
from .views import PlayerView, describe_event, describe_knowledge

#: Display order. The ontology's ``hn:choices`` for ``get_state`` must list exactly these.
SECTIONS: Final = ("turn", "tokens", "fireworks", "discard", "hands", "history")
HISTORY_LENGTH: Final = 20


@dataclass(frozen=True, slots=True)
class TokenPool:
    """A limited pool of tokens. ``available + spent == capacity`` always."""

    available: int
    capacity: int

    @property
    def spent(self) -> int:
        return self.capacity - self.available


def hint_pool(view: PlayerView, rules: RuleSet) -> TokenPool:
    return TokenPool(view.hints, rules.max_hints)


def fuse_pool(view: PlayerView, rules: RuleSet) -> TokenPool:
    return TokenPool(view.fuses, rules.initial_fuses)


def _turn(view: PlayerView, rules: RuleSet) -> str:
    who = "YOUR turn" if view.current == view.viewer else f"{view.names[view.current]}'s turn"
    lines = [f"Turn {view.turn}: it is {who}.", f"Cards left in the deck: {view.deck_size}."]
    if view.final_turns_left is not None:
        lines.append(f"FINAL ROUND: the deck is empty; {view.final_turns_left} turn(s) remain, including this one.")
    if view.over_reason is not None:
        lines.append(f"GAME OVER: {view.over_reason}.")
    return "\n".join(lines)


def _tokens(view: PlayerView, rules: RuleSet) -> str:
    def row(label: str, pool: TokenPool) -> str:
        return f"{label}: available {pool.available}, spent {pool.spent}, capacity {pool.capacity}."

    return "\n".join([row("Hint tokens", hint_pool(view, rules)),
                      row("Fuse tokens", fuse_pool(view, rules)) + " The game ends when none is available."])


def _fireworks(view: PlayerView, rules: RuleSet) -> str:
    best = len(rules.colors) * rules.top_rank
    lines = [f"Score {view.score}/{best}."]
    for color, height in view.stacks.items():
        need = "complete" if height >= rules.top_rank else f"next needed: {color.value} {height + 1}"
        lines.append(f"  {color.value}: {height}, {need}")
    return "\n".join(lines)


def _discard(view: PlayerView, rules: RuleSet) -> str:
    if not view.discard:
        return "Discard pile: empty."
    return f"Discard pile ({len(view.discard)}): " + ", ".join(str(c) for c in view.discard)


def _hands(view: PlayerView, rules: RuleSet) -> str:
    lines: list[str] = []
    for seat, hand in enumerate(view.hands):
        hidden = any(cv.card is None for cv in hand)  # decided by the regime, in build_view
        if seat == view.viewer:
            who = f"{view.names[seat]} (you)."
            lines.append(f"{who} You cannot see your cards; this is what hints have told you:" if hidden
                         else f"{who} You can see your cards. 'You know' is what hints have told you:")
        else:
            lines.append(f"{view.names[seat]}. You can see their cards; they cannot. 'They know' is what hints told them:")
        for cv in hand:
            know = describe_knowledge(cv.knowledge)
            if cv.card is None:
                lines.append(f"  position {cv.position}: {know}")
            else:
                lines.append(f"  position {cv.position}: {cv.card}   ({'you' if seat == view.viewer else 'they'} know: {know})")
    lines.append("New cards enter at the last position; cards after a removed one shift down by one.")
    return "\n".join(lines)


def _history(view: PlayerView, rules: RuleSet) -> str:
    recent = view.events[-HISTORY_LENGTH:]
    if not recent:
        return "No events yet."
    return f"Last {len(recent)} event(s):\n" + "\n".join("  " + describe_event(e, view) for e in recent)


_RENDERERS = {"turn": _turn, "tokens": _tokens, "fireworks": _fireworks,
              "discard": _discard, "hands": _hands, "history": _history}
assert tuple(_RENDERERS) == SECTIONS


def state_text(view: PlayerView, rules: RuleSet, section: str | None = None) -> str:
    """One section, or (with ``section=None``) all of them in order."""
    if section is not None:
        key = section.strip().lower()
        if key not in _RENDERERS:
            raise IllegalAction(f"Unknown section {section!r}. Sections: {list(SECTIONS)}.")
        return f"## {key}\n{_RENDERERS[key](view, rules)}"
    return "\n\n".join(f"## {k}\n{r(view, rules)}" for k, r in _RENDERERS.items())

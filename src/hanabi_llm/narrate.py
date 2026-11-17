"""One wording for events, used by the prompt history, the CLI and the viewer.

An event is described as a list of SEGMENTS (small typed pieces) so that a
renderer can colour players, chip cards and mark outcomes without parsing
text; ``to_text`` flattens the same segments to plain text. Because the
wording lives only here, the three consumers cannot drift apart.

Plain text is written to survive copy and paste: a card is always "blue 1",
never a bare "1" with the colour only in a background, and every line starts
"T7 Alice ...". Examples:

    T7 Alice hinted Carol: position 3 is green
    T7 Alice hinted Carol: positions 1 and 3 are rank 2
    T8 Bob discarded blue 1 from position 1
    T8 Bob drew red 2
    T10 Alice played blue 2 from position 2 -> misfire (fuse lost)

A hint line states what the hint says. That every other card of the hand is
NOT that colour or rank is a rule (``hint_is_complete``), shown per card in the
state, so it is not repeated on every line.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

Record = dict[str, Any]
Segment = dict[str, Any]


def _join(items: Sequence[int]) -> str:
    """1 -> "1", 1 and 3 -> "1 and 3", 0, 2, 4 -> "0, 2 and 4"."""
    parts = [str(i) for i in items]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def card_hidden(card: Record) -> bool:
    """A card whose identity was removed (or is the redacted placeholder, rank 0)."""
    return card.get("color") is None or not card.get("rank")


def segments(rec: Record) -> list[Segment]:
    """Describe one ``event`` feed record."""
    turn: Segment = {"k": "turn", "v": rec["turn"]}
    kind = rec["kind"]
    if kind == "end":
        return [turn, {"k": "text", "v": f"Game over: {rec['reason']} (score {rec['score']})"}]
    actor: Segment = {"k": "player", "seat": rec["actor"]}
    if kind == "hint":
        targets = rec["targets"]
        value = {"k": "value", "attribute": rec["attribute"], "value": rec["value"]}
        return [turn, actor, {"k": "text", "v": " hinted "}, {"k": "player", "seat": rec["receiver"]},
                {"k": "text", "v": f": {'position' if len(targets) == 1 else 'positions'} {_join(targets)} {'is' if len(targets) == 1 else 'are'} "}, value]
    card: Segment = {"k": "card", "card": rec.get("card")}
    if kind == "play":
        mark = {"k": "mark", "tone": "ok", "v": "placed"} if rec["success"] else {"k": "mark", "tone": "bad", "v": "misfire (fuse lost)"}
        return [turn, actor, {"k": "text", "v": " played "}, card, {"k": "text", "v": f" from position {rec['position']} "}, mark]
    if kind == "discard":
        return [turn, actor, {"k": "text", "v": " discarded "}, card, {"k": "text", "v": f" from position {rec['position']}"}]
    if kind == "draw":
        # The drawer is the one who may not see it (a viewer that blinds that player hides it too).
        return [turn, actor, {"k": "text", "v": " drew "}, {**card, "secret_from": rec["actor"]}]
    raise ValueError(f"unknown event kind {kind!r}")


def to_text(segs: Sequence[Segment], names: Sequence[str], you: int | None = None) -> str:
    """Flatten segments to one line of plain text. ``you`` marks the viewer's own name."""
    out: list[str] = []
    for g in segs:
        k = g["k"]
        if k == "turn":
            out.append(f"T{g['v']} ")
        elif k == "player":
            out.append(names[g["seat"]] + (" (you)" if g["seat"] == you else ""))
        elif k == "text":
            out.append(g["v"])
        elif k == "value":
            out.append(str(g["value"]) if g["attribute"] == "color" else f"rank {g['value']}")
        elif k == "card":
            c = g["card"]
            out.append("a card" if card_hidden(c) else f"{c['color']} {c['rank']}")
        elif k == "mark":
            out.append(f"-> {g['v']}")
    return "".join(out)

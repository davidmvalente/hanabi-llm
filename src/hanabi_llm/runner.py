"""The referee: owns the game loop and the only write path to the engine.

Per turn:  project state -> view + toolbox for the active seat -> ask the
player (with a wall-clock guard) -> apply through ``Game.apply``.

If a player cannot produce a valid action the game is ABORTED, never
"patched up" with a substitute move (see ``faults.py``). The result says who
was at fault, or that nobody was (infrastructure), so the tournament layer
can disqualify that seat or simply re-run the game.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from .engine import Game
from .faults import FaultKind, GameStatus, JSON, PlayerFault
from .model import Action, Event, Seat
from .players import Player, TurnContext
from .tools import ToolBox
from .views import build_view


@dataclass(frozen=True, slots=True)
class Fault:
    seat: Seat
    player: str
    kind: FaultKind
    detail: str
    turn: int
    messages: tuple[JSON, ...] = ()


@dataclass(slots=True)
class TurnTrace:
    turn: int
    seat: Seat
    player: str
    action: Action
    events: list[Event]
    messages: list[JSON]
    prompt_tokens: int
    completion_tokens: int


@dataclass(slots=True)
class GameResult:
    status: GameStatus
    seed: int | None
    score: int
    lost: bool
    reason: str
    fault: Fault | None = None
    traces: list[TurnTrace] = field(default_factory=list)

    @property
    def counts_for_ranking(self) -> bool:
        """Only a game that was genuinely played to its end may be ranked."""
        return self.status is GameStatus.COMPLETED

    @property
    def official_score(self) -> int | None:
        """Standard rules: blowing all fuses scores zero. ``None`` if void."""
        if not self.counts_for_ranking:
            return None
        return 0 if self.lost else self.score

    @property
    def turns(self) -> int:
        return len(self.traces)

    @property
    def action_log(self) -> list[tuple[Seat, Action]]:
        """Seat/action pairs; with ``seed`` this replays the game exactly."""
        return [(t.seat, t.action) for t in self.traces]


TurnCallback = Callable[[TurnTrace], Awaitable[None] | None]


async def play_game(
    game: Game,
    players: list[Player],
    *,
    on_turn: TurnCallback | None = None,
    turn_timeout: float = 300.0,
) -> GameResult:
    """Run a game to completion or to the first fault.

    ``turn_timeout`` is only a safety net against hung connections. It is
    deliberately generous and, when it fires, is treated as an infrastructure
    fault, not blamed on the model: slow is not wrong.
    """
    if [p.name for p in players] != game.state.names:
        raise ValueError("player names must match the game's seat names, in order")
    notes: dict[Seat, str] = {}
    traces: list[TurnTrace] = []

    def finish(status: GameStatus, fault: Fault | None, reason: str) -> GameResult:
        return GameResult(status, game.seed, game.state.score, game.state.lost, reason, fault, traces)

    while not game.is_over:
        seat = game.state.current
        view = build_view(game, seat)
        ctx = TurnContext(
            seat=seat,
            view=view,
            toolbox=ToolBox(game, seat, view, notes),
            legal=tuple(game.legal_actions(seat)),
            notes=notes.get(seat, ""),
        )
        try:
            async with asyncio.timeout(turn_timeout):
                result = await players[seat].take_turn(ctx)
        except TimeoutError:
            fault = Fault(seat, players[seat].name, FaultKind.TIMEOUT, f"no answer within {turn_timeout:.0f}s", game.state.turn)
            return finish(GameStatus.ABORTED_INFRA_FAULT, fault, "aborted: turn timeout")
        except PlayerFault as pf:
            fault = Fault(seat, players[seat].name, pf.kind, pf.detail, game.state.turn, tuple(pf.trace))
            status = GameStatus.ABORTED_MODEL_FAULT if pf.kind.attributable else GameStatus.ABORTED_INFRA_FAULT
            return finish(status, fault, f"aborted: {pf.kind.value}")

        turn_no = game.state.turn
        events = game.apply(seat, result.action)  # re-validates; raises IllegalAction on a bug
        trace = TurnTrace(turn_no, seat, players[seat].name, result.action, events, result.trace,
                          result.prompt_tokens, result.completion_tokens)
        traces.append(trace)
        if on_turn is not None:
            maybe = on_turn(trace)
            if maybe is not None:
                await maybe

    assert game.state.over_reason is not None
    return finish(GameStatus.COMPLETED, None, game.state.over_reason)

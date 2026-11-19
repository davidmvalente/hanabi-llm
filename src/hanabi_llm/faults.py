"""Fault taxonomy: who is responsible when a game cannot finish?

Fairness principle: a model is judged only on what it controls. Anything
else (provider outage, rate limit, slow network) voids the game for
*everyone* instead of penalising, or silently altering, one player's game.

    MODEL faults (attributable)  -> the game is aborted, the seat is blamed.
    INFRA faults (not attributable) -> the game is aborted, nobody is blamed,
                                       and the game should be re-run.

A faulted game never reports a ranking score. There are no silent
substitute moves: a game's result is either genuinely played or marked void.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

JSON = dict[str, Any]


class FaultKind(StrEnum):
    INVALID_ACTIONS = "invalid_actions"  # exceeded the budget of rejected actions
    NO_ACTION = "no_action"              # never produced an action tool call in time
    PROVIDER_ERROR = "provider_error"    # provider failed after all retries
    TIMEOUT = "timeout"                  # wall-clock guard tripped

    @property
    def attributable(self) -> bool:
        """True if the fault is the model's own doing."""
        return self in (FaultKind.INVALID_ACTIONS, FaultKind.NO_ACTION)


class GameStatus(StrEnum):
    COMPLETED = "completed"
    ABORTED_MODEL_FAULT = "aborted_model_fault"
    ABORTED_INFRA_FAULT = "aborted_infra_fault"


class PlayerFault(Exception):
    """Raised by a player that cannot produce a valid action for its turn."""

    def __init__(self, kind: FaultKind, detail: str, trace: list[JSON] | None = None) -> None:
        super().__init__(f"{kind.value}: {detail}")
        self.kind = kind
        self.detail = detail
        self.trace: list[JSON] = trace or []

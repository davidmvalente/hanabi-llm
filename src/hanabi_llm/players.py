"""Players: anything that can turn a ``TurnContext`` into an action.

``LLMPlayer`` runs the model's tool-calling loop for ONE turn. It sees only
its own ``PlayerView`` (through the rendered text and the read tools), so a
model can never read another model's hidden information.

A player either returns a valid action or raises ``PlayerFault``; there is no
"best effort" substitute move (see ``faults.py`` for why).
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from .engine import Game
from .faults import FaultKind, PlayerFault
from .llm import JSON, ChatClient, Completion, LLMError
from .prompt import PromptSpec, load_prompt
from .model import Action, Seat
from .tools import ToolBox
from .states import state_text
from .views import PlayerView


@dataclass(frozen=True, slots=True)
class TurnContext:
    seat: Seat
    view: PlayerView
    toolbox: ToolBox
    legal: tuple[Action, ...]


@dataclass(slots=True)
class TurnResult:
    action: Action
    trace: list[JSON] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Player(Protocol):
    name: str

    async def take_turn(self, ctx: TurnContext) -> TurnResult:
        """Return a valid action, or raise ``PlayerFault``."""
        ...


class RandomPlayer:
    """Baseline: a uniformly random legal action. Useful offline and in tests."""

    def __init__(self, name: str, seed: int | None = None) -> None:
        self.name = name
        self._rng = random.Random(seed)

    async def take_turn(self, ctx: TurnContext) -> TurnResult:
        return TurnResult(self._rng.choice(ctx.legal))


#: Rejected action attempts allowed in one turn before the seat is at fault. Small models need room to recover.
DEFAULT_MAX_FAILED_ACTIONS = 5


def _brief(text: str, limit: int) -> str:
    """One line, trimmed: enough to see what happened without flooding the console."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class LLMPlayer:
    """Plays through any ``ChatClient`` using the tool-calling loop."""

    def __init__(
        self,
        name: str,
        client: ChatClient,
        game: Game,
        seat: Seat,
        *,
        prompt: PromptSpec | None = None,
        max_calls: int = 8,
        max_failed_actions: int = DEFAULT_MAX_FAILED_ACTIONS,
    ) -> None:
        self.name = name
        self._client = client
        self._max_calls = max_calls
        self._max_failed = max_failed_actions
        self._rules = game.rules
        self.prompt = prompt or load_prompt()  # latest version unless one is given
        self._system = self.prompt.render(
            name=game.state.names[seat],
            players=", ".join(game.state.names),
            max_calls=max_calls,
            max_failed_actions=max_failed_actions,
        )

    async def take_turn(self, ctx: TurnContext) -> TurnResult:
        trace: list[JSON] = []
        ptok = ctok = failed = 0
        attempts: list[str] = []          # what was rejected, in words, for the fault message
        tools_called: Counter[str] = Counter()
        messages: list[JSON] = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": state_text(ctx.view, self._rules) + "\n\nChoose your action now."},
        ]
        tools = ctx.toolbox.schemas()

        for _ in range(self._max_calls):
            try:
                completion: Completion = await self._client.complete(messages, tools)
            except LLMError as exc:
                # The client already retried with backoff. Not the model's fault.
                trace.append({"error": str(exc)})
                raise PlayerFault(FaultKind.PROVIDER_ERROR, str(exc), trace) from exc
            ptok += completion.prompt_tokens
            ctok += completion.completion_tokens
            messages.append(completion.to_message())
            trace.append(completion.to_message())

            if not completion.tool_calls:
                failed += 1
                said = _brief(completion.content or "", 100)
                attempts.append(f'no tool call, it said: "{said}"' if said else "no tool call and an empty reply")
                nudge: JSON = {"role": "user", "content": "Call a tool. To end your turn use exactly one of play_card, discard_card or give_hint."}
                messages.append(nudge)
                trace.append(nudge)
            for call in completion.tool_calls:
                outcome = ctx.toolbox.execute(call.name, call.arguments)
                tools_called[call.name] += 1
                tool_msg: JSON = {"role": "tool", "tool_call_id": call.id, "content": outcome.text}
                messages.append(tool_msg)
                trace.append(tool_msg)
                if outcome.action is not None:
                    return TurnResult(outcome.action, trace, ptok, ctok)  # first valid action ends the turn
                if outcome.failed_action:
                    failed += 1
                    attempts.append(f"{call.name}({_brief(call.arguments, 80)}) -> {_brief(outcome.text, 140)}")
            if failed >= self._max_failed:
                raise PlayerFault(FaultKind.INVALID_ACTIONS, f"{failed} rejected action attempts in one turn: " + " | ".join(attempts), trace)
        used = ", ".join(f"{n} x{c}" for n, c in tools_called.most_common()) or "no tools"
        extra = f"; rejected: {' | '.join(attempts)}" if attempts else ""
        raise PlayerFault(FaultKind.NO_ACTION, f"no action taken within {self._max_calls} model calls (it called: {used}){extra}", trace)

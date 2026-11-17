"""Players: anything that can turn a ``TurnContext`` into an action.

``LLMPlayer`` runs the model's tool-calling loop for ONE turn. It sees only
its own ``PlayerView`` (through the rendered text and the read tools), so a
model can never read another model's hidden information or private notes.

A player either returns a valid action or raises ``PlayerFault``; there is no
"best effort" substitute move (see ``faults.py`` for why).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from .engine import Game
from .faults import FaultKind, PlayerFault
from .llm import JSON, ChatClient, Completion, LLMError
from .model import Action, Seat
from .tools import ToolBox
from .views import PlayerView, render_view


@dataclass(frozen=True, slots=True)
class TurnContext:
    seat: Seat
    view: PlayerView
    toolbox: ToolBox
    legal: tuple[Action, ...]
    notes: str


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


def build_system_prompt(game: Game, seat: Seat, *, max_calls: int, max_failed_actions: int) -> str:
    """Rules text comes from the ontology, so prompt and engine cannot disagree."""
    rules = game.rules
    names = game.state.names
    rule_lines = "\n".join(f"- {t}" for t in rules.rule_texts)
    return f"""You are {names[seat]}, one of {len(names)} players ({', '.join(names)}) in a cooperative game of Hanabi.

GOAL: together, build five fireworks (colours: {', '.join(c.value for c in rules.colors)}), each from rank 1 up to {rules.top_rank} in order. Score = total cards placed (max {len(rules.colors) * rules.top_rank}).

The twist: you can see everyone's cards EXCEPT YOUR OWN. The only communication allowed is the `give_hint` action. Any text you write besides tool calls is NOT seen by anyone.

Deck: each colour has {', '.join(f'{n}x rank {r}' for r, n in sorted(rules.copies_by_rank.items()))}. You start with {rules.initial_hints} hint tokens (max {rules.max_hints}) and {rules.initial_fuses} fuse tokens.

Rules:
{rule_lines}

HOW A TURN WORKS
1. Information tools are free and unlimited in number (up to {max_calls} model calls per turn in total): get_state, get_rules, get_concepts (game-theory notes: mechanics, dynamics, kinetics), estimate_card (odds for one of your own cards), query_knowledge (SPARQL over what you may see) and take_notes (private memory for your later turns). Use them as often as you like before acting.
2. Your turn ends the moment you call exactly ONE action tool that is accepted: play_card, discard_card or give_hint. After that, the next player moves.
3. A rejected action returns an error and you may try again. If you make {max_failed_actions} rejected action attempts in one turn, or never take an action within the call budget, the game is aborted and counted as your fault."""


class LLMPlayer:
    """Plays through any ``ChatClient`` using the tool-calling loop."""

    def __init__(
        self,
        name: str,
        client: ChatClient,
        game: Game,
        seat: Seat,
        *,
        max_calls: int = 8,
        max_failed_actions: int = 3,
    ) -> None:
        self.name = name
        self._client = client
        self._max_calls = max_calls
        self._max_failed = max_failed_actions
        self._system = build_system_prompt(game, seat, max_calls=max_calls, max_failed_actions=max_failed_actions)

    async def take_turn(self, ctx: TurnContext) -> TurnResult:
        trace: list[JSON] = []
        ptok = ctok = failed = 0
        notes = f"\n\nYour private notes from earlier turns:\n{ctx.notes}" if ctx.notes else ""
        messages: list[JSON] = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": render_view(ctx.view) + notes + "\n\nChoose your action now."},
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
                nudge: JSON = {"role": "user", "content": "Call a tool. To end your turn use exactly one of play_card, discard_card or give_hint."}
                messages.append(nudge)
                trace.append(nudge)
            for call in completion.tool_calls:
                outcome = await ctx.toolbox.execute(call.name, call.arguments)
                tool_msg: JSON = {"role": "tool", "tool_call_id": call.id, "content": outcome.text}
                messages.append(tool_msg)
                trace.append(tool_msg)
                if outcome.action is not None:
                    return TurnResult(outcome.action, trace, ptok, ctok)  # first valid action ends the turn
                if outcome.failed_action:
                    failed += 1
            if failed >= self._max_failed:
                raise PlayerFault(FaultKind.INVALID_ACTIONS, f"{failed} rejected action attempts in one turn", trace)
        raise PlayerFault(FaultKind.NO_ACTION, f"no action taken within {self._max_calls} model calls", trace)

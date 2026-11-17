"""A player that ranks documents instead of calling tools.

Each turn is ONE rerank call:

  query      the goal, the rules and the current state (``prompts/rerank.v1.yaml``)
  documents  the candidate actions as short sentences (``actions.py``)

The best-scored document is played. A reranker cannot reason in steps, ask for the rules or fail
a tool call, so the only things that can go wrong are a provider failure (an infrastructure
fault) and, with the ``all`` candidate set, ranking impossible actions first: walking down the
ranking, each impossible choice is a rejected action, and the same budget as the chat players
applies (``max_failed_actions``) before that seat is at fault.

Fairness controls:
  * the order of the documents is shuffled to avoid position bias, with a seed fixed by
    (game seed, turn) (one seat acts per turn, so the turn identifies it), so a model's order
    is reproducible and every model facing the same turn of the same deal sees the same order;
  * ties break towards the lowest canonical index, independent of the shuffle.
"""

from __future__ import annotations

import random
from typing import Any

from .actions import action_text, candidate_actions
from .engine import Game
from .faults import FaultKind, PlayerFault
from .llm import LLMError
from .model import Action, Seat
from .ontology import Ontology
from .players import TurnContext, TurnResult
from .prompt import PromptSpec, load_prompt
from .rerank import RerankClient
from .states import compact_state_text


def document_order(seed: int | None, turn: int, n: int, shuffle: bool = True) -> list[int]:
    """The order in which ``n`` candidate documents are sent: ``order[i]`` is the canonical index of the i-th.

    A pure function of (game seed, turn, n), so it is reproducible and identical for every model that
    faces the same turn of the same deal. ``shuffle=False`` keeps the canonical order.
    """
    order = list(range(n))
    if shuffle:
        random.Random(f"{seed}:{turn}").shuffle(order)
    return order


def rules_summary(onto: Ontology) -> str:
    """The rules as short bullet lines, from the ontology (so the query cannot drift from the engine)."""
    order = {"precondition": 0, "effect": 1, "game rule": 2}  # what must hold, what happens, then the whole game
    docs = sorted(onto.rule_docs.values(), key=lambda d: (order[d.kind], d.name))
    return "\n".join(f"- {d.text}" for d in docs)


class RerankPlayer:
    """Plays by asking a rerank model which candidate action is most relevant to the situation."""

    def __init__(
        self,
        name: str,
        client: RerankClient,
        game: Game,
        seat: Seat,
        *,
        prompt: PromptSpec | None = None,
        candidates: str = "legal",
        shuffle: bool = True,
        max_failed_actions: int = 3,
    ) -> None:
        self.name = name
        self._client = client
        self._rules = game.rules
        self._seed = game.seed
        self._candidates = candidates
        self._shuffle = shuffle
        self._max_failed = max_failed_actions
        self.prompt = prompt or load_prompt(kind="rerank")
        self._rules_text = rules_summary(game.ontology)
        self._players = ", ".join(game.state.names)

    def query(self, ctx: TurnContext) -> str:
        return self.prompt.render(name=self.name, players=self._players, rules=self._rules_text,
                                  state=compact_state_text(ctx.view, self._rules))

    async def take_turn(self, ctx: TurnContext) -> TurnResult:
        names = ctx.view.names
        cands = candidate_actions(ctx.view, self._rules, ctx.legal, self._candidates)
        order = document_order(self._seed, ctx.view.turn, len(cands), self._shuffle)   # order[i] = canonical index of the i-th document sent
        documents = [action_text(cands[c], names) for c in order]
        query = self.query(ctx)
        trace: dict[str, Any] = {"query": query, "documents": documents}
        try:
            ranking = await self._client.rank(query, documents)
        except LLMError as exc:
            raise PlayerFault(FaultKind.PROVIDER_ERROR, str(exc), [{**trace, "error": str(exc)}]) from exc

        # best score first; ties go to the lowest canonical index, whatever order the documents were sent in
        ordered = sorted(((r.score, order[r.index]) for r in ranking.ranked), key=lambda t: (-t[0], t[1]))
        trace["scores"] = [{"document": documents[r.index], "score": r.score} for r in ranking.ranked]
        trace["usage"] = ranking.usage
        legal = set(ctx.legal)
        rejected: list[str] = []
        for _, canon in ordered:
            action: Action = cands[canon]
            if action in legal:
                trace["chosen"], trace["rejected_first"] = action_text(action, names), rejected
                return TurnResult(action, [trace], prompt_tokens=ranking.tokens)
            rejected.append(action_text(action, names))      # an impossible action ranked above a possible one
            if len(rejected) >= self._max_failed:
                raise PlayerFault(FaultKind.INVALID_ACTIONS, f"{len(rejected)} impossible actions ranked first: {rejected}", [{**trace, "rejected": rejected}])
        incomplete = len(ranking.ranked) < len(documents)
        raise PlayerFault(FaultKind.PROVIDER_ERROR if incomplete else FaultKind.INVALID_ACTIONS,
                          "no legal action among the ranked documents" + (" (the provider ranked only some)" if incomplete else ""),
                          [{**trace, "rejected": rejected}])

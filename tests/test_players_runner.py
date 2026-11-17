from __future__ import annotations

import asyncio
import json
import time
from collections import deque

import httpx
import pytest

import hanabi_llm.llm as llm
import hanabi_llm.tools as tools
from conftest import B, G, R, Y, rig
from hanabi_llm.__main__ import SEAT_NAMES
from hanabi_llm.engine import Game
from hanabi_llm.faults import FaultKind, GameStatus, PlayerFault
from hanabi_llm.llm import JSON, Completion, LLMError, OpenRouterClient, ToolCall
from hanabi_llm.model import DiscardCard, GiveHint, PlayCard
from hanabi_llm.players import LLMPlayer, RandomPlayer, TurnContext, TurnResult
from hanabi_llm.runner import play_game
from hanabi_llm.tools import ToolBox
from hanabi_llm.views import build_view


class ScriptedClient:
    """ChatClient that replays canned completions and records what it was shown."""

    def __init__(self, script: list[Completion | Exception]) -> None:
        self.model = "scripted"
        self._script = deque(script)
        self.seen: list[list[JSON]] = []

    async def complete(self, messages: list[JSON], tools: list[JSON]) -> Completion:
        self.seen.append([dict(m) for m in messages])
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return item


def call(name: str, args: dict[str, object], id: str = "c1") -> Completion:
    return Completion(None, (ToolCall(id, name, json.dumps(args)),))


def ctx_for(game: Game, seat: int = 0, notes: dict[int, str] | None = None) -> TurnContext:
    view = build_view(game, seat)
    return TurnContext(seat, view, ToolBox(game, seat, view, notes if notes is not None else {}),
                       tuple(game.legal_actions(seat)), "")


def rigged() -> Game:
    return rig(Game(["Ann", "Bob"], seed=0), [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 2)] * 10)


# --- toolbox ---------------------------------------------------------------


def test_tools_come_from_ontology(game2: Game) -> None:
    schemas = {s["function"]["name"]: s["function"] for s in ctx_for(game2).toolbox.schemas()}
    assert set(schemas) == {"play_card", "discard_card", "give_hint", "get_state", "get_rules",
                            "get_concepts", "estimate_card", "query_knowledge", "take_notes"}
    hint = schemas["give_hint"]["parameters"]
    assert hint["required"] == ["target"]
    assert hint["properties"]["target"]["enum"] == ["Bob"]  # never offers yourself


async def test_error_accounting(game2: Game) -> None:
    tb = ctx_for(game2).toolbox
    assert (await tb.execute("play_card", "not json")).failed_action
    assert (await tb.execute("give_hint", {"target": "Nobody", "color": "red"})).failed_action
    assert (await tb.execute("nonsense", {})).failed_action
    free = await tb.execute("query_knowledge", {"sparql": "SELEKT"})   # mistakes in READ tools are free
    assert free.is_error and not free.failed_action
    assert (await tb.execute("discard_card", {"position": "2"})).action == DiscardCard(2)


async def test_sparql_is_read_only_and_view_scoped() -> None:
    tb = ctx_for(rigged()).toolbox
    for bad in ("DELETE WHERE { ?s ?p ?o }", "SELECT * WHERE { SERVICE <http://x> { ?s ?p ?o } }", "SELECT * WHERE {" + " " * 2000 + "}"):
        assert (await tb.execute("query_knowledge", {"sparql": bad})).is_error
    mine = await tb.execute("query_knowledge", {"sparql": "SELECT ?c WHERE { ?c hn:hasColor hn:Red }"})
    assert mine.text == "(no results)"  # my own red cards are not in my graph


async def test_slow_query_times_out_without_blocking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "SPARQL_TIMEOUT_S", 0.05)
    monkeypatch.setattr(tools, "run_sparql_select", lambda g, q: (time.sleep(0.3), "late")[1])
    start = time.monotonic()
    out = await ctx_for(rigged()).toolbox.execute("query_knowledge", {"sparql": "SELECT * WHERE {?s ?p ?o}"})
    assert out.is_error and "longer than" in out.text and time.monotonic() - start < 0.25


async def test_knowledge_tools() -> None:
    tb = ctx_for(rigged()).toolbox
    assert "Deck: 50 cards" in (await tb.execute("get_rules", {})).text
    assert "red 1" not in (await tb.execute("get_state", {})).text        # still hides my hand
    index = (await tb.execute("get_concepts", {"category": "kinetic"})).text
    assert "card_mass" in index and "hand_shift" not in index
    assert "Guidance:" in (await tb.execute("get_concepts", {"name": "activation_energy"})).text
    assert (await tb.execute("get_concepts", {"name": "nope"})).is_error
    # the same knowledge is reachable through the graph
    q = "SELECT ?n WHERE { ?c a hn:Kinetic ; hn:conceptName ?n }"
    assert (await tb.execute("query_knowledge", {"sparql": q})).text.count("\n") == 6


async def test_estimate_card_counts_unseen_copies(game2: Game) -> None:
    rig(game2, [[(Y, 1)] * 5, [(R, 5), (B, 1), (B, 1), (B, 1), (B, 2)]])
    game2.state.knowledge[game2.state.hands[0][0].uid].rank = 5   # Ann was told card 0 is a 5
    out = (await ctx_for(game2).toolbox.execute("estimate_card", {"position": 0})).text
    assert "P(is a 5) = 100%" in out and "P(playable now) = 0%" in out
    assert "red 5" not in out and "yellow 5: 25%" in out          # Bob holds the red 5 -> 4 candidates
    assert (await ctx_for(game2).toolbox.execute("estimate_card", {"position": 9})).is_error


# --- LLM player ------------------------------------------------------------


async def test_reads_are_free_then_one_action_ends_turn() -> None:
    game, notes = rigged(), {}
    client = ScriptedClient([
        call("get_state", {}, "a"), call("get_rules", {}, "b"), call("estimate_card", {"position": 0}, "c"),
        call("take_notes", {"notes": "check later"}, "d"), call("give_hint", {"target": "Bob", "rank": 1}, "e"),
    ])
    result = await LLMPlayer("Ann", client, game, 0).take_turn(ctx_for(game, 0, notes))
    assert result.action == GiveHint(1, rank=1) and notes[0] == "check later"


async def test_noninterference_hidden_cards_never_reach_the_model() -> None:
    """Two games that differ ONLY in Ann's own hidden cards and the deck must
    produce byte-identical input to Ann's model, across every read tool."""

    async def play_script(hand: list[tuple[object, int]], deck: list[tuple[object, int]]) -> list[list[JSON]]:
        game = rig(Game(["Ann", "Bob"], seed=0), [hand, [(B, 1)] * 5], deck=deck)  # type: ignore[arg-type]
        client = ScriptedClient([
            call("get_state", {}, "a"), call("get_rules", {}, "b"), call("estimate_card", {"position": 0}, "c"),
            call("estimate_card", {"position": 4}, "d"),
            call("query_knowledge", {"sparql": "SELECT ?c ?p WHERE { ?c hn:heldBy ?p }"}, "e"),
            call("query_knowledge", {"sparql": "SELECT ?c ?col WHERE { ?c hn:hasColor ?col }"}, "f"),
            call("discard_card", {"position": 0}, "g"),
        ])
        await LLMPlayer("Ann", client, game, 0).take_turn(ctx_for(game))
        return client.seen

    seen_a = await play_script([(R, 1)] * 5, [(Y, 2)] * 10)
    seen_b = await play_script([(G, 4), (G, 5), (Y, 3), (B, 2), (R, 5)], [(Y, 5)] * 10)
    assert json.dumps(seen_a) == json.dumps(seen_b)


async def test_read_tool_mistakes_never_disqualify() -> None:
    game = rigged()
    client = ScriptedClient([call("query_knowledge", {"sparql": "garbage"}, str(i)) for i in range(5)]
                            + [call("discard_card", {"position": 0})])
    assert (await LLMPlayer("Ann", client, game, 0, max_calls=8).take_turn(ctx_for(game))).action == DiscardCard(0)


async def test_rejected_actions_are_a_model_fault() -> None:
    game = rigged()
    client = ScriptedClient([call("play_card", {"position": 99})] * 3)
    with pytest.raises(PlayerFault) as exc:
        await LLMPlayer("Ann", client, game, 0).take_turn(ctx_for(game))
    assert exc.value.kind is FaultKind.INVALID_ACTIONS and exc.value.kind.attributable


async def test_recovering_after_one_rejection() -> None:
    game = rigged()
    client = ScriptedClient([call("play_card", {"position": 99}), call("play_card", {"position": 0})])
    assert (await LLMPlayer("Ann", client, game, 0).take_turn(ctx_for(game))).action == PlayCard(0)


async def test_text_only_replies_then_budget_exhaustion() -> None:
    game = rigged()
    texty = ScriptedClient([Completion("thinking...", ())] * 3)
    with pytest.raises(PlayerFault) as e1:
        await LLMPlayer("Ann", texty, game, 0).take_turn(ctx_for(game))
    assert e1.value.kind is FaultKind.INVALID_ACTIONS
    readers = ScriptedClient([call("get_state", {}, str(i)) for i in range(4)])
    with pytest.raises(PlayerFault) as e2:
        await LLMPlayer("Ann", readers, game, 0, max_calls=4).take_turn(ctx_for(game))
    assert e2.value.kind is FaultKind.NO_ACTION


async def test_provider_error_is_not_the_models_fault() -> None:
    game = rigged()
    with pytest.raises(PlayerFault) as exc:
        await LLMPlayer("Ann", ScriptedClient([LLMError("503 after retries")]), game, 0).take_turn(ctx_for(game))
    assert exc.value.kind is FaultKind.PROVIDER_ERROR and not exc.value.kind.attributable


# --- referee ---------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
async def test_random_games_complete(seed: int) -> None:
    names = ["a", "b", "c"]
    game = Game(names, seed=seed)
    result = await play_game(game, [RandomPlayer(n, seed=seed + i) for i, n in enumerate(names)])
    assert result.status is GameStatus.COMPLETED and result.counts_for_ranking
    assert 0 <= result.score <= 25 and result.official_score is not None


async def test_action_log_and_seed_replay_exactly() -> None:
    names = ["a", "b"]
    first = await play_game(Game(names, seed=11), [RandomPlayer(n, seed=i) for i, n in enumerate(names)])
    replay = Game(names, seed=first.seed)
    for seat, action in first.action_log:
        replay.apply(seat, action)
    assert replay.is_over and replay.state.score == first.score


class Failing:
    def __init__(self, name: str, kind: FaultKind | None, sleep: float = 0.0) -> None:
        self.name, self._kind, self._sleep = name, kind, sleep

    async def take_turn(self, ctx: TurnContext) -> TurnResult:
        await asyncio.sleep(self._sleep)
        if self._kind is None:
            return TurnResult(ctx.legal[0])
        raise PlayerFault(self._kind, "scripted failure")


@pytest.mark.parametrize(("kind", "status", "ranked_blame"), [
    (FaultKind.INVALID_ACTIONS, GameStatus.ABORTED_MODEL_FAULT, True),
    (FaultKind.NO_ACTION, GameStatus.ABORTED_MODEL_FAULT, True),
    (FaultKind.PROVIDER_ERROR, GameStatus.ABORTED_INFRA_FAULT, False),
])
async def test_faults_abort_the_game_and_never_change_it(kind: FaultKind, status: GameStatus, ranked_blame: bool) -> None:
    game = Game(["Ann", "Bob"], seed=2)
    result = await play_game(game, [Failing("Ann", kind), RandomPlayer("Bob", 1)])
    assert result.status is status and result.fault is not None and result.fault.seat == 0
    assert not result.counts_for_ranking and result.official_score is None
    assert result.fault.kind.attributable is ranked_blame
    assert game.state.turn == 1 and not game.state.events      # no substitute move was ever made


async def test_timeout_is_infra_not_blame() -> None:
    result = await play_game(Game(["Ann", "Bob"], seed=2), [Failing("Ann", None, sleep=1.0), RandomPlayer("Bob", 1)],
                             turn_timeout=0.05)
    assert result.status is GameStatus.ABORTED_INFRA_FAULT and result.fault is not None
    assert result.fault.kind is FaultKind.TIMEOUT


async def test_a_fault_mid_game_keeps_the_partial_trace() -> None:
    class DiesLater(RandomPlayer):
        async def take_turn(self, ctx: TurnContext) -> TurnResult:
            if ctx.view.turn > 3:
                raise PlayerFault(FaultKind.NO_ACTION, "gave up")
            return await super().take_turn(ctx)

    result = await play_game(Game(["a", "b"], seed=4), [DiesLater("a", 1), RandomPlayer("b", 2)])
    assert result.status is GameStatus.ABORTED_MODEL_FAULT and result.turns == 4 and result.fault.turn == 5  # type: ignore[union-attr]


async def test_referee_rejects_names_mismatch() -> None:
    with pytest.raises(ValueError):
        await play_game(Game(["a", "b"]), [RandomPlayer("a"), RandomPlayer("zzz")])


# --- fairness invariants ---------------------------------------------------


def test_seat_names_are_neutral() -> None:
    joined = " ".join(SEAT_NAMES).lower()
    assert not any(w in joined for w in ("claude", "gpt", "gemini", "llama", "mistral", "random"))


def test_system_prompt_is_identical_across_seats_except_the_name() -> None:
    game = Game(["Ann", "Bob"], seed=1)
    p0 = LLMPlayer("Ann", ScriptedClient([]), game, 0)._system.replace("Ann", "X")
    p1 = LLMPlayer("Bob", ScriptedClient([]), game, 1)._system.replace("Bob", "X")
    # the two players' names appear in a different order in the roster line; compare the rest
    assert p0.split("\n", 1)[1] == p1.split("\n", 1)[1]


# --- OpenRouter client -----------------------------------------------------


async def test_openrouter_parses_tool_calls_retries_and_sends_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429)
        body = json.loads(request.content)
        assert body["model"] == "x/y" and body["tools"] and body["max_tokens"] == 4096
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "play_card", "arguments": "{\"position\": 1}"}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        })

    real_sleep = asyncio.sleep
    monkeypatch.setattr(llm.asyncio, "sleep", lambda *_: real_sleep(0))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        c = await OpenRouterClient("x/y", "k", http).complete([{"role": "user", "content": "hi"}], [{"type": "function"}])
    assert attempts["n"] == 2 and c.tool_calls[0].name == "play_card" and c.prompt_tokens == 10


async def test_openrouter_surfaces_provider_errors() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"error": {"message": "no credits"}}))
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(LLMError, match="no credits"):
            await OpenRouterClient("m", "k", http).complete([], [])

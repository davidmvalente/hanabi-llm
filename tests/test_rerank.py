"""Rerank mode: the client, the player, and the whole CLI over HTTP (3 players)."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable, Iterator, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

import hanabi_llm.rerank as rerank_module
from conftest import B, G, R, Y, rig
from hanabi_llm.__main__ import build_parser, run
from hanabi_llm.actions import action_text, candidate_actions
from hanabi_llm.engine import Game
from hanabi_llm.faults import FaultKind, GameStatus, PlayerFault
from hanabi_llm.feed import read_feed
from hanabi_llm.fold import fold
from hanabi_llm.llm import LLMError
from hanabi_llm.model import DiscardCard, GiveHint, PlayCard
from hanabi_llm.players import TurnContext
from hanabi_llm.prompt import available_versions, load_prompt
from hanabi_llm.rerank import OpenRouterRerankClient, Ranked, Ranking
from hanabi_llm.rerank_player import RerankPlayer, document_order
from hanabi_llm.runner import play_game
from hanabi_llm.tools import ToolBox
from hanabi_llm.views import build_view

NAMES = ["Alice", "Bob", "Carol"]


# --- the client ---------------------------------------------------------------------------------------------


def client_for(handler: Callable[[httpx.Request], httpx.Response], monkeypatch: pytest.MonkeyPatch | None = None) -> tuple[OpenRouterRerankClient, httpx.AsyncClient]:
    if monkeypatch is not None:                                            # skip the retry backoff
        real_sleep = asyncio.sleep
        monkeypatch.setattr(rerank_module.asyncio, "sleep", lambda _s: real_sleep(0))
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenRouterRerankClient("cohere/rerank-v3.5", "k", http), http


def ok(results: list[dict[str, Any]], **extra: Any) -> httpx.Response:
    return httpx.Response(200, json={"results": results, **extra})


async def test_the_request_follows_the_documented_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"], seen["auth"], seen["url"] = json.loads(req.content), req.headers["authorization"], str(req.url)
        return ok([{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}], usage={"total_tokens": 42})

    client, http = client_for(handler)
    ranking = await client.rank("the query", ["doc a", "doc b"])
    assert seen["body"] == {"model": "cohere/rerank-v3.5", "query": "the query", "documents": ["doc a", "doc b"], "top_n": 2}
    assert seen["auth"] == "Bearer k" and seen["url"] == "https://openrouter.ai/api/v1/rerank"
    assert ranking.ranked == (Ranked(1, 0.9), Ranked(0, 0.1)) and ranking.tokens == 42
    await http.aclose()


@pytest.mark.parametrize(("results", "expected"), [
    ([{"index": 2, "relevance_score": 0.2}, {"index": 0, "relevance_score": 0.8}], (Ranked(2, 0.2), Ranked(0, 0.8))),   # any order
    ([{"index": 0, "score": 0.5}], (Ranked(0, 0.5),)),                                                                 # `score` accepted
    ([{"document": {"text": "doc c"}, "relevance_score": 0.7}], (Ranked(2, 0.7),)),                                   # matched by text
    ([{"document": "doc b", "relevance_score": 1}], (Ranked(1, 1.0),)),
])
async def test_the_results_are_parsed_defensively(results: list[dict[str, Any]], expected: tuple[Ranked, ...]) -> None:
    client, http = client_for(lambda r: ok(results))
    assert (await client.rank("q", ["doc a", "doc b", "doc c"])).ranked == expected
    await http.aclose()


@pytest.mark.parametrize("payload", [
    {}, {"results": []}, {"results": "nope"}, {"error": {"message": "boom"}},
    {"results": [{"index": 9, "relevance_score": 1}]},                         # out of range
    {"results": [{"index": 0, "relevance_score": 1}, {"index": 0, "relevance_score": 2}]},   # duplicate
    {"results": [{"index": 0, "relevance_score": float("inf")}]},
    {"results": [{"index": 0}]},                                               # no score
    {"results": [{"document": "unknown text", "relevance_score": 1}]},        # unmatched, no index
    {"results": [{"document": "dup", "relevance_score": 1}]},                  # ambiguous text
])
async def test_unusable_responses_are_provider_errors(payload: dict[str, Any]) -> None:
    client, http = client_for(lambda r: httpx.Response(200, content=json.dumps(payload, allow_nan=True)))
    docs = ["doc a", "dup", "dup"] if "dup" in json.dumps(payload) else ["doc a", "doc b"]
    with pytest.raises(LLMError):
        await client.rank("q", docs)
    await http.aclose()


@pytest.mark.parametrize("status", [402, 404, 413])
async def test_errors_that_retrying_cannot_fix_fail_at_once(status: int) -> None:
    calls = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status, text="no")

    client, http = client_for(handler)
    with pytest.raises(LLMError, match=str(status)):
        await client.rank("q", ["a"])
    assert calls["n"] == 1
    await http.aclose()


@pytest.mark.parametrize("status", [429, 502, 524, 529])
async def test_transient_errors_are_retried(status: int, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(status) if calls["n"] < 3 else ok([{"index": 0, "relevance_score": 1}])

    client, http = client_for(handler, monkeypatch)
    assert (await client.rank("q", ["a"])).ranked == (Ranked(0, 1.0),) and calls["n"] == 3
    await http.aclose()


async def test_giving_up_after_the_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(529)

    client, http = client_for(handler, monkeypatch)
    with pytest.raises(LLMError, match="gave up after 4"):
        await client.rank("q", ["a"])
    assert calls["n"] == 4
    await http.aclose()


# --- the player -----------------------------------------------------------------------------------------------


class Scripted:
    """A rerank client that scores documents with a function of their text."""

    model = "scripted"

    def __init__(self, scorer: Callable[[str], float], *, keep: Callable[[str], bool] | None = None) -> None:
        self.scorer, self.keep, self.calls = scorer, keep, []  # type: ignore[var-annotated]

    async def rank(self, query: str, documents: Sequence[str]) -> Ranking:
        self.calls.append((query, list(documents)))
        scored = [Ranked(i, self.scorer(d)) for i, d in enumerate(documents) if self.keep is None or self.keep(d)]  # a partial ranking drops the rest
        return Ranking(tuple(scored), {"total_tokens": 7})


def ctx_for(game: Game, seat: int = 0) -> TurnContext:
    view = build_view(game, seat)
    return TurnContext(seat, view, ToolBox(game, seat, view), tuple(game.legal_actions(seat)))


def player(game: Game, client: Scripted, seat: int = 0, **kw: Any) -> RerankPlayer:
    return RerankPlayer(NAMES[seat], client, game, seat, **kw)  # type: ignore[arg-type]


async def test_the_best_scored_document_is_played() -> None:
    game = Game(NAMES, seed=3)
    pick = Scripted(lambda d: 1.0 if d == "discard the card at position 3" else 0.1)
    result = await player(game, pick).take_turn(ctx_for(game))
    assert result.action == DiscardCard(3) and result.prompt_tokens == 7
    (trace,) = result.trace
    assert trace["chosen"] == "discard the card at position 3" and trace["query"].startswith("You are Alice") and trace["documents"]
    assert {s["document"] for s in trace["scores"]} == set(trace["documents"])


@pytest.mark.parametrize("shuffle", [True, False])
async def test_every_document_maps_back_to_its_own_action(shuffle: bool) -> None:
    """With or without the shuffle, choosing a document gives exactly the action it describes."""
    game = Game(NAMES, seed=3)
    for action in candidate_actions(build_view(game, 0), game.rules, game.legal_actions(0)):
        text = action_text(action, NAMES)
        result = await player(game, Scripted(lambda d, t=text: 1.0 if d == t else 0.0), shuffle=shuffle).take_turn(ctx_for(game))
        assert result.action == action


async def test_ties_go_to_the_lowest_canonical_index_whatever_the_shuffle() -> None:
    game = Game(NAMES, seed=3)
    for shuffle in (True, False):
        result = await player(game, Scripted(lambda d: 0.5), shuffle=shuffle).take_turn(ctx_for(game))
        assert result.action == PlayCard(0)


def test_the_document_order_is_a_pure_function_of_seed_turn_and_size() -> None:
    base = document_order(3, 5, 22)
    assert base == document_order(3, 5, 22)                                  # reproducible
    assert sorted(base) == list(range(22)) and base != list(range(22))       # a real permutation
    assert document_order(3, 6, 22) != base                                  # a later turn: another order
    assert document_order(4, 5, 22) != base                                  # another deal: another order
    assert document_order(3, 5, 22, shuffle=False) == list(range(22))        # the canonical order on request
    assert document_order(None, 5, 22) == document_order(None, 5, 22)        # an unseeded game is still deterministic per turn


async def test_the_player_sends_exactly_that_order() -> None:
    game = Game(NAMES, seed=3)
    canonical = Scripted(lambda d: 0.0)
    await player(game, canonical, shuffle=False).take_turn(ctx_for(game))
    shuffled = Scripted(lambda d: 0.0)
    await player(game, shuffled).take_turn(ctx_for(game))
    n = len(canonical.calls[0][1])
    assert shuffled.calls[0][1] == [canonical.calls[0][1][i] for i in document_order(3, 1, n)]


async def test_a_provider_failure_is_an_infrastructure_fault_not_the_models() -> None:
    class Down:
        model = "down"

        async def rank(self, q: str, d: Sequence[str]) -> Ranking:
            raise LLMError("HTTP 529 after retries")

    game = Game(NAMES, seed=3)
    with pytest.raises(PlayerFault) as exc:
        await RerankPlayer("Alice", Down(), game, 0).take_turn(ctx_for(game))  # type: ignore[arg-type]
    assert exc.value.kind is FaultKind.PROVIDER_ERROR and not exc.value.kind.attributable


def impossible_hint(game: Game) -> str:
    """The text of a hint the current player cannot give (a colour the receiver does not hold)."""
    legal = {action_text(a, NAMES) for a in game.legal_actions(0)}
    return next(action_text(a, NAMES) for a in candidate_actions(build_view(game, 0), game.rules, (), "all")
                if isinstance(a, GiveHint) and action_text(a, NAMES) not in legal)


async def test_ranking_an_impossible_action_first_is_a_rejected_action_in_all_mode() -> None:
    game = Game(NAMES, seed=3)
    bad = impossible_hint(game)
    legal_pick = "discard the card at position 0"
    pick = Scripted(lambda d: {bad: 1.0, legal_pick: 0.9}.get(d, 0.0))
    result = await player(game, pick, candidates="all").take_turn(ctx_for(game))
    assert result.action == DiscardCard(0)                                   # it falls through to the next best, legal one
    assert result.trace[0]["rejected_first"] == [bad]                        # and the miss is recorded


async def test_three_impossible_choices_in_a_row_are_the_models_fault() -> None:
    game = Game(NAMES, seed=3)
    legal = {action_text(a, NAMES) for a in game.legal_actions(0)}
    impossible = [action_text(a, NAMES) for a in candidate_actions(build_view(game, 0), game.rules, (), "all") if action_text(a, NAMES) not in legal]
    assert len(impossible) >= 3
    pick = Scripted(lambda d: 1.0 if d in impossible else 0.0)
    with pytest.raises(PlayerFault) as exc:
        await player(game, pick, candidates="all").take_turn(ctx_for(game))
    assert exc.value.kind is FaultKind.INVALID_ACTIONS and exc.value.kind.attributable


async def test_in_legal_mode_no_choice_can_be_impossible() -> None:
    game = Game(NAMES, seed=3)
    legal = set(game.legal_actions(0))
    docs = Scripted(lambda d: 0.0)
    await player(game, docs).take_turn(ctx_for(game))
    assert len(docs.calls[0][1]) == len(legal)                               # there is nothing impossible to choose


async def test_a_partial_ranking_without_a_legal_action_is_the_providers_problem() -> None:
    game = Game(NAMES, seed=3)
    bad = impossible_hint(game)
    with pytest.raises(PlayerFault) as exc:
        await player(game, Scripted(lambda d: 1.0, keep=lambda d: d == bad), candidates="all", shuffle=False).take_turn(ctx_for(game))
    assert exc.value.kind is FaultKind.PROVIDER_ERROR


@pytest.mark.parametrize("seed", range(5))
async def test_whole_games_with_arbitrary_scores_never_fault_in_legal_mode(seed: int) -> None:
    import random

    rng = random.Random(seed)
    game = Game(NAMES, seed=seed)
    players = [player(game, Scripted(lambda d: rng.random()), seat=i) for i in range(3)]
    result = await play_game(game, players)  # type: ignore[arg-type]
    assert result.status is GameStatus.COMPLETED and result.fault is None


# --- no hidden information in what is sent --------------------------------------------------------------------


async def test_the_query_and_documents_never_depend_on_the_players_own_hidden_cards() -> None:
    """Two games that differ only in Alice's own cards and the deck send Alice's reranker identical input."""

    async def sent(hand: list, deck: list) -> list[tuple[str, list[str]]]:  # type: ignore[type-arg]
        game = rig(Game(NAMES, seed=0), [hand, [(B, 1), (G, 2), (R, 3), (Y, 4), (B, 5)], [(R, 1)] * 5], deck=deck)
        c = Scripted(lambda d: 0.0)
        await player(game, c).take_turn(ctx_for(game))
        return c.calls

    assert await sent([(R, 1)] * 5, [(Y, 2)] * 10) == await sent([(G, 4), (G, 5), (Y, 3), (B, 2), (R, 5)], [(Y, 5)] * 10)


# --- the query -----------------------------------------------------------------------------------------------------


def test_the_rerank_prompt_is_a_versioned_file_with_four_variables() -> None:
    spec = load_prompt(kind="rerank")
    assert (spec.kind, spec.version, available_versions("rerank")) == ("rerank", 1, [1])
    assert set(spec.variables) == {"name", "players", "rules", "state"}
    with pytest.raises(ValueError, match="unknown prompt kind"):
        load_prompt(kind="nope")


@pytest.mark.parametrize("players", [2, 3, 4, 5])
async def test_the_query_stays_short_and_holds_goal_rules_and_state(players: int) -> None:
    names = [f"P{i}" for i in range(players)]
    game = Game(names, seed=1)
    p = RerankPlayer("P0", Scripted(lambda d: 0.0), game, 0)  # type: ignore[arg-type]
    q = p.query(ctx_for(game))
    assert len(q) < 3500                                                     # about 800 tokens at 5 players
    for part in ("You are P0", "Goal:", "Rules:", "Current state:", "## hands", "## fireworks", "## tokens", "Which single action"):
        assert part in q
    assert "## history" not in q and "explain_rules" not in q               # compact, and no tool talk: a reranker has no tools
    assert q.index("- A played") < q.index("- Discarding") < q.index("- When the deck")   # rules grouped: must hold, happens, whole game


# --- the whole CLI over HTTP --------------------------------------------------------------------------------------


class FakeRerankServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def answer(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        model, docs = body["model"], body["documents"]
        if model.endswith("down"):
            return 503, {"error": "overloaded"}
        if model.endswith("garbage"):
            return 200, {"results": [{"index": len(docs) + 5, "relevance_score": 1.0}]}
        # prefers discarding position 0; everything else scores 0.1 (so the game is discard-only and runs the deck out)
        return 200, {"results": [{"index": i, "relevance_score": 1.0 if d == "discard the card at position 0" else 0.1} for i, d in enumerate(docs)],
                     "usage": {"total_tokens": 100}}


@pytest.fixture
def rerank_server() -> Iterator[tuple[FakeRerankServer, str]]:
    fake = FakeRerankServer()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.requests.append(body)
            fake.headers.append({k.lower(): v for k, v in self.headers.items()})
            status, payload = fake.answer(body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield fake, f"http://127.0.0.1:{server.server_address[1]}/api/v1/rerank"
    server.shutdown()
    server.server_close()


def play_cli(tmp_path: Path, url: str, models: str, monkeypatch: pytest.MonkeyPatch, *extra: str):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    args = build_parser().parse_args(["--mode", "rerank", "--models", models, "--seed", "5", "--api-url", url,
                                      "--log", str(tmp_path / "g.jsonl"), "--trace", str(tmp_path / "t.jsonl"), *extra])
    return asyncio.run(run(args))


def test_three_rerank_seats_play_a_whole_game_with_one_call_per_turn(rerank_server, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake, url = rerank_server
    result = play_cli(tmp_path, url, "fake/ranker-a,fake/ranker-b,fake/ranker-c", monkeypatch)
    assert result.status is GameStatus.COMPLETED and result.fault is None and "final round" in result.reason
    assert len(fake.requests) == result.turns                                # exactly one rerank call per turn
    assert {h["authorization"] for h in fake.headers} == {"Bearer test-key"}
    first = fake.requests[0]
    assert set(first) == {"model", "query", "documents", "top_n"} and first["top_n"] == len(first["documents"])
    assert first["query"].startswith("You are Alice") and "Current state:" in first["query"]
    assert sorted(first["documents"]).count("discard the card at position 0") == 1
    assert [r["model"] for r in fake.requests[:3]] == ["fake/ranker-a", "fake/ranker-b", "fake/ranker-c"]

    feed = read_feed(tmp_path / "g.jsonl")
    cfg = feed[0]["config"]
    assert (cfg["mode"], cfg["candidates"], cfg["shuffle"], cfg["knowledge"]) == ("rerank", "legal", True, [])
    assert cfg["prompt"]["kind"] == "rerank" and cfg["prompt"]["version"] == 1 and len(cfg["prompt"]["sha256"]) == 64
    assert fold(feed).score == result.score and feed[-1]["status"] == "completed"
    rows = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    t = rows[0]["messages"][0]
    assert rows[0]["action"] == {"tool": "discard_card", "position": 0} and t["chosen"] == "discard the card at position 0"
    assert t["scores"] and t["usage"] == {"total_tokens": 100} and rows[0]["tokens"] == [100, 0]


def test_a_provider_outage_aborts_without_blaming_the_model(rerank_server, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    real_sleep = asyncio.sleep
    monkeypatch.setattr(rerank_module.asyncio, "sleep", lambda _s: real_sleep(0))
    fake, url = rerank_server
    result = play_cli(tmp_path, url, "fake/down,fake/ranker-b,fake/ranker-c", monkeypatch)
    assert result.status is GameStatus.ABORTED_INFRA_FAULT and not result.counts_for_ranking
    assert result.fault is not None and result.fault.kind is FaultKind.PROVIDER_ERROR and len(fake.requests) == 4


def test_a_nonsense_ranking_is_a_provider_problem_too(rerank_server, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _, url = rerank_server
    result = play_cli(tmp_path, url, "fake/ranker-a,fake/garbage,fake/ranker-c", monkeypatch)
    assert result.status is GameStatus.ABORTED_INFRA_FAULT and result.fault is not None and result.fault.player == "Bob"


def test_the_options_are_recorded_and_knowledge_is_refused(rerank_server, tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _, url = rerank_server
    play_cli(tmp_path, url, "fake/ranker-a,fake/ranker-b,fake/ranker-c", monkeypatch, "--candidates", "all", "--no-shuffle")
    cfg = read_feed(tmp_path / "g.jsonl")[0]["config"]
    assert (cfg["candidates"], cfg["shuffle"]) == ("all", False)
    with pytest.raises(SystemExit):
        play_cli(tmp_path, url, "fake/ranker-a,fake/ranker-b,fake/ranker-c", monkeypatch, "--knowledge", "mdk")

from __future__ import annotations

import pytest

from conftest import B, G, R, W, Y, rig
from hanabi_llm.engine import Game
from hanabi_llm.model import CardKnowledge, DiscardCard, EndEvent, GiveHint, IllegalAction, PlayCard
from hanabi_llm.ontology import load_ontology


def test_ontology_drives_setup() -> None:
    onto = load_ontology()
    assert onto.rules.deck_size == 50
    assert onto.rules.hand_size_by_players == {2: 5, 3: 5, 4: 4, 5: 4}
    assert set(onto.affordances) == {"play_card", "discard_card", "give_hint", "get_state", "explain_rules"}


@pytest.mark.parametrize(("n", "hand"), [(2, 5), (3, 5), (4, 4), (5, 4)])
def test_deal(n: int, hand: int) -> None:
    g = Game([f"p{i}" for i in range(n)], seed=1)
    assert all(len(h) == hand for h in g.state.hands)
    assert len(g.state.deck) == 50 - n * hand
    assert (g.state.hints, g.state.fuses) == (8, 3)


def test_successful_play_and_draw(game2: Game) -> None:
    rig(game2, [[(R, 1), (B, 3)], [(G, 1), (W, 2)]], deck=[(Y, 5), (Y, 4)])
    game2.apply(0, PlayCard(0))
    assert game2.state.stacks[R] == 1
    assert [str(c) for c in game2.state.hands[0]] == ["blue 3", "yellow 4"]  # new card in LAST slot
    assert game2.state.current == 1


def test_misfire_costs_fuse_and_discards(game2: Game) -> None:
    rig(game2, [[(R, 3), (B, 3)], [(G, 1), (W, 2)]], deck=[(Y, 5)])
    game2.apply(0, PlayCard(0))
    assert game2.state.fuses == 2 and game2.state.stacks[R] == 0
    assert [str(c) for c in game2.state.discard] == ["red 3"]


def test_three_misfires_end_game(game2: Game) -> None:
    rig(game2, [[(R, 3)] * 5, [(B, 3)] * 5], deck=[(Y, 1)] * 20)
    for seat in (0, 1, 0):
        game2.apply(seat, PlayCard(0))
    assert game2.is_over and game2.state.lost
    assert isinstance(game2.state.events[-1], EndEvent)


def test_discard_refunds_hint_up_to_max(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 1)] * 20)
    game2.apply(0, GiveHint(1, B))
    assert game2.state.hints == 7
    game2.apply(1, DiscardCard(0))
    assert game2.state.hints == 8
    game2.apply(0, DiscardCard(0))
    assert game2.state.hints == 8  # capped


def test_completing_a_stack_refunds_hint(game2: Game) -> None:
    rig(game2, [[(R, 5)] * 5, [(B, 1)] * 5], deck=[(Y, 1)] * 20)
    game2.rig(stacks={R: 4}, hints=3)
    game2.apply(0, PlayCard(0))
    assert game2.state.hints == 4


@pytest.mark.parametrize(
    ("action", "needle"),
    [
        (GiveHint(0, R), "yourself"),
        (GiveHint(1, R), "reveal nothing"),
        (GiveHint(1, 9), "Unknown hint"),
        (GiveHint(1, 0), "Unknown hint"),
        (PlayCard(9), "not in your hand"),
        (DiscardCard(-1), "not in your hand"),
    ],
)
def test_illegal_actions(game2: Game, action: GiveHint | PlayCard | DiscardCard, needle: str) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5])
    with pytest.raises(IllegalAction, match=needle):
        game2.apply(0, action)


def test_no_hint_tokens_and_wrong_turn(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5])
    game2.rig(hints=0)
    with pytest.raises(IllegalAction, match="No hint tokens"):
        game2.apply(0, GiveHint(1, B))
    with pytest.raises(IllegalAction, match="turn"):
        game2.apply(1, DiscardCard(0))


def test_hint_updates_public_knowledge(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1), (R, 2), (B, 3), (G, 4), (W, 5)]])
    game2.apply(0, GiveHint(1, B))
    k = [game2.state.knowledge[c.uid] for c in game2.state.hands[1]]
    assert [x.color for x in k] == [B, None, B, None, None]
    assert k[1].not_colors == {B}  # a blue hint that skipped this card rules blue out


def test_final_round_every_player_acts_once(game2: Game) -> None:
    # One card left in the deck: the draw on turn 1 empties it.
    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 1)])
    game2.apply(0, DiscardCard(0))
    assert game2.state.final_turns_left == 2
    game2.apply(1, DiscardCard(0))
    assert not game2.is_over
    game2.apply(0, DiscardCard(0))
    assert game2.is_over and "final round" in (game2.state.over_reason or "")


def test_perfect_score_ends_immediately(game2: Game) -> None:
    rig(game2, [[(R, 5)] * 5, [(B, 1)] * 5], deck=[(Y, 1)] * 10)
    game2.rig(stacks={c: 5 for c in (R, Y, G, B, W)} | {R: 4})
    game2.apply(0, PlayCard(0))
    assert game2.is_over and game2.state.score == 25


def test_legal_actions_all_pass_check(game2: Game) -> None:
    for a in game2.legal_actions(0):
        game2.check(0, a)
    assert game2.legal_actions(1) == []  # not their turn


# --- the encoding of a hint -----------------------------------------------------------------------


def test_a_hint_is_one_value_so_both_or_neither_cannot_be_written() -> None:
    GiveHint(1, B), GiveHint(1, 3)  # a colour, or a rank
    with pytest.raises(TypeError):
        GiveHint(1, B, 1)  # type: ignore[call-arg]  # there is no second field to fill
    with pytest.raises(TypeError):
        GiveHint(1)  # type: ignore[call-arg]
    assert GiveHint(1, B).attribute == "color" and GiveHint(1, 3).attribute == "rank"


def test_a_hint_targets_exactly_the_matching_subset_of_the_hand(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1), (R, 2), (B, 3), (G, 4), (B, 5)]])
    (event,) = game2.apply(0, GiveHint(1, B))
    assert (event.receiver, event.value, event.targets, event.attribute) == (1, B, (0, 2, 4), "color")  # type: ignore[union-attr]
    # complete: the targets ARE blue, every card left out is NOT blue
    k = [game2.state.knowledge[c.uid] for c in game2.state.hands[1]]
    assert [x.color for x in k] == [B, None, B, None, B]
    assert [x.not_colors for x in k] == [set(), {B}, set(), {B}, set()]


def test_a_rank_hint_is_complete_in_the_same_way(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1), (R, 2), (B, 1), (G, 4), (B, 5)]])
    (event,) = game2.apply(0, GiveHint(1, 1))
    assert (event.value, event.targets, event.attribute) == (1, (0, 2), "rank")  # type: ignore[union-attr]
    k = [game2.state.knowledge[c.uid] for c in game2.state.hands[1]]
    assert [x.rank for x in k] == [1, None, 1, None, None]
    assert [x.not_ranks for x in k] == [set(), {1}, set(), {1}, {1}]


def test_legal_hints_are_exactly_the_values_the_receiver_holds(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1), (R, 2)]])
    hints = {a.value for a in game2.legal_actions(0) if isinstance(a, GiveHint)}
    assert hints == {B, R, 1, 2}


# --- the engine decides, the fold applies -----------------------------------------------------------------


def test_the_engine_gets_what_an_action_does_from_the_fold(game2: Game, monkeypatch: pytest.MonkeyPatch) -> None:
    """Break the fold's refund rule and the engine's behaviour changes: there is no second copy of it."""
    import hanabi_llm.fold as fold_module

    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 2)] * 10)
    game2.rig(hints=3)
    game2.apply(0, DiscardCard(0))
    assert game2.state.hints == 4                                           # the fold refunds one

    game3 = rig(Game(["Ann", "Bob"], seed=0), [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 2)] * 10)
    game3.rig(hints=3)
    monkeypatch.setattr(fold_module, "refund_hint", lambda s: None)
    game3.apply(0, DiscardCard(0))
    assert game3.state.hints == 3                                           # nothing in the engine refunds on its own


def test_the_engine_source_holds_no_transition_logic() -> None:
    """Static guard: outside ``rig`` (scenario setup, the one deliberate exception) the engine never
    writes hints, fuses, stacks, turn, discard or hands itself; its only write path is ``step``."""
    import re
    from pathlib import Path

    src = (Path(__file__).parent.parent / "src" / "hanabi_llm" / "engine.py").read_text()
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    code = "\n".join(line.split("#")[0] for line in code.splitlines())
    code = code[: code.index("    def rig(")] + code[code.index("    def legal_actions(") :]        # drop rig itself
    forbidden = [r"\.(hints|fuses|turn|current|final_turns_left|deck_size)\s*([-+]=|=[^=])", r"\.stacks\[[^\]]*\]\s*([-+]=|=[^=])",
                 r"\.discard\.(append|pop)", r"\.played\.append", r"\.hands\[[^\]]*\]\.(append|pop)", r"\.knowledge\[[^\]]*\]\.\w+\s*="]
    assert [m.group(0) for pattern in forbidden for m in re.finditer(pattern, code)] == []
    assert code.count("step(self._fold") == 2                                                          # the action's events, and the end


def test_the_snapshot_cannot_be_mutated(game2: Game) -> None:
    st = game2.state
    with pytest.raises(AttributeError):
        st.hints = 1  # type: ignore[misc]
    with pytest.raises(TypeError):
        st.stacks[R] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        st.hands[0].append(st.hands[0][0])  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        next(iter(st.knowledge.values())).color = B  # type: ignore[misc]


def test_the_snapshot_follows_the_game(game2: Game) -> None:
    before = game2.state
    assert game2.state is before                                           # cached while nothing changes
    game2.apply(0, DiscardCard(0))
    after = game2.state
    assert after is not before and before.turn == 1 and after.turn == 2     # the old snapshot is a point in time
    assert len(after.discard) == 1 and len(before.discard) == 0


def test_rig_sets_a_position_and_leaves_the_rest_alone(game2: Game) -> None:
    deck_before = game2.state.deck
    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5])
    assert game2.state.deck == deck_before                                  # the deck is untouched unless asked
    game2.rig(stacks={R: 3}, hints=2, fuses=1)
    st = game2.state
    assert (st.stacks[R], st.hints, st.fuses, st.stacks[B]) == (3, 2, 1, 0)
    uid = st.hands[1][0].uid
    game2.rig(knowledge={uid: CardKnowledge(color=B, not_ranks=frozenset({2}))})
    assert game2.state.knowledge[uid] == CardKnowledge(color=B, not_ranks=frozenset({2}))
    with pytest.raises(KeyError):
        game2.rig(knowledge={999999: CardKnowledge()})                      # knowledge is for cards in hands only

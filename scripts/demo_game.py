"""Play one full game with a simple heuristic bot and write its feed (for trying the viewer).

    uv run python scripts/demo_game.py game.jsonl 1 3      # output, seed, players

Random bots lose all their fuses within a few turns and never build a stack; this bot plays only what
it KNOWS is playable and hints playable cards, so games run through the final round with stacks,
discards and hints to look at. It is a demo, not a baseline: it sees only its own view.
"""
import asyncio, sys
from pathlib import Path
from hanabi_llm.engine import Game
from hanabi_llm.feed import FeedWriter, event_record, header_record, result_record
from hanabi_llm.model import DiscardCard, GiveHint, PlayCard
from hanabi_llm.players import TurnContext, TurnResult
from hanabi_llm.runner import play_game


class Heuristic:
    """Plays what it KNOWS is playable; otherwise hints a teammate's playable card; otherwise discards."""
    def __init__(self, name): self.name = name

    async def take_turn(self, ctx: TurnContext) -> TurnResult:
        v, me = ctx.view, ctx.view.viewer
        for cv in v.hands[me]:                                   # 1. a card I know is playable
            k = cv.knowledge
            if k.color and k.rank and v.stacks[k.color] == k.rank - 1:
                return TurnResult(PlayCard(cv.position))
        if v.hints > 0:                                          # 2. help a teammate play
            for seat, hand in enumerate(v.hands):
                if seat == me: continue
                for cv in hand:
                    c = cv.card
                    if c and v.stacks[c.color] == c.rank - 1:
                        k = cv.knowledge
                        if k.color is None: return TurnResult(GiveHint(seat, c.color))
                        if k.rank is None:  return TurnResult(GiveHint(seat, c.rank))
        if v.hints < 8:                                          # 3. recycle a hint token
            return TurnResult(DiscardCard(0))
        for seat, hand in enumerate(v.hands):                    # 4. at the cap: hint something harmless
            if seat != me and hand and hand[0].card:
                return TurnResult(GiveHint(seat, hand[0].card.color))
        return TurnResult(DiscardCard(0))


async def main(out: Path, seed: int, n: int):
    names = ["Alice", "Bob", "Carol", "Dave", "Eve"][:n]
    game = Game(names, seed=seed)
    w = FeedWriter(out)
    w.write(header_record(game, {nm: "heuristic-bot" for nm in names}, {"note": "demo"}))
    def on_turn(t):
        for ev in t.events: w.write(event_record(ev))
    result = await play_game(game, [Heuristic(nm) for nm in names], on_turn=on_turn)
    w.write(result_record(result)); w.close()
    print(f"seed {seed}: {result.status.value}, {result.reason}, score {result.score}, turns {result.turns}")

if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])))

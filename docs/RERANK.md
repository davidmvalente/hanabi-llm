# Rerank mode (branch `rerankers-version`)

This branch tests **rerank models** as Hanabi players. A rerank model does not generate text or call
tools: it takes a `query` and a list of `documents` and returns a relevance score for each. So a
turn here is **one call**: the query is the goal, the rules and the current state, the documents are
the candidate actions, and the best-scored action is played.

```bash
export OPENROUTER_API_KEY=...
uv run hanabi-llm --mode rerank --models <rerank-id>,<rerank-id>,<rerank-id> --seed 1 --log game.jsonl --trace trace.jsonl
uv run hanabi-view game.jsonl
```

Everything else is shared with the chat arms: the same engine, feed, trace, viewer, fault policy and
neutral seat names. `--mode chat` (the default) is unchanged.

## The endpoint

`POST https://openrouter.ai/api/v1/rerank`, documented in the [OpenRouter API reference](https://openrouter.ai/docs/api/api-reference/rerank/create-rerank).

| | Contract | Status |
|---|---|---|
| Request | `model`, `query`, `documents` required; `top_n` optional. We send `top_n = number of documents`, so every score comes back. | from the documentation |
| Response | `results`, each with an `index` and a `relevance_score`; also `usage`. | **as described for this endpoint; the exact field names were not visible in the documentation excerpt I could read**, so they are parsed defensively (see below) and the first real call will confirm them |
| 402, 404, 413 | out of credits, unknown model, payload too large | fail at once (retrying cannot help), as an infrastructure fault |
| 429, 502, 524, 529, 5xx | rate limit, upstream failure, timeout, overloaded | retried with backoff (4 attempts), then an infrastructure fault |

Defensive parsing (`rerank.py`): a result without `index` is matched to its document by text when
that is unambiguous; `score` is accepted for `relevance_score`; an index out of range, a duplicate,
a non-finite score, or an empty `results` is a provider error, never a model fault.

## One turn

The query comes from `prompts/rerank.v1.yaml` (versioned, hashed into the log header) with four
variables. The rules are the ontology's own rule texts, so the query cannot drift from the engine.
Real example (3 players, turn 3; the state is shortened here):

```
You are Carol, playing the cooperative card game Hanabi with Alice, Bob, Carol.
Goal: together build the five coloured stacks, each from rank 1 up to rank 5; the team score is the number of cards placed.
You cannot see your own cards. The only way to communicate is a hint.

Rules:
- A played or discarded card must be in the acting player's hand at the given position.
- A hint names exactly one attribute: a colour or a rank, never both.
  ... (10 rules)

Current state:
## turn / ## tokens / ## fireworks / ## discard / ## hands      (what you can see; your own cards hidden)
## recent events
  T1 Alice hinted Bob: positions 0, 2 and 3 are rank 1
  T2 Bob hinted Carol (you): position 0 is green

Which single action should Carol take now?
```

Documents (22 at this position, canonical order; sent in a seeded shuffle):

```
play the card at position 0 ... position 4        discard the card at position 0 ... position 4
hint Alice: blue    hint Alice: green    hint Alice: white    hint Alice: rank 2    hint Alice: rank 3
hint Bob: red       hint Bob: white      hint Bob: yellow     hint Bob: rank 1      hint Bob: rank 2 ...
```

The query is 2,400 to 3,200 characters (about 600 to 800 tokens) for 2 to 5 players, and a test
fails above 3,500. No document is longer than 30 characters.

## Candidate actions: how many, as a function of the number of players

Hand size is 5 for 2 or 3 players and 4 for 4 or 5. A player can always play or discard any of its
`h` cards (`2h` actions). A hint is a (receiver, value) pair: 5 colours and 5 ranks, for each of the
`N - 1` other players.

* **Every combination**: `2h + (N - 1) x 10`.
* **Most that can be legal**: a hand of `h` cards shows at most `min(h, 5)` colours and `min(h, 5)`
  ranks, so `2h + (N - 1) x 2 min(h, 5)`. (With no hint token, no hint is legal.)

Measured on complete games (heuristic demo bot, 6 games per row; `scripts/action_space.py`):

| Players | Hand | Candidates, every combination | Max legal | Legal per turn (mean, min-max) | Turns per game (= rerank calls) |
|---|---|---|---|---|---|
| 2 | 5 | 20 | 20 | 15.6 (10-19) | 72 (71-73) |
| 3 | 5 | 30 | 30 | 18.2 (10-28) | 66 (62-67) |
| 4 | 4 | 38 | 32 | 18.6 (8-30) | 65 (64-67) |
| 5 | 4 | 48 | 40 | 21.5 (8-37) | 60 (59-61) |

So the set grows roughly linearly with the player count (about 10 more hint documents per extra
player), stays under 50 documents, and a complete game is 60 to 72 calls. The formulas are verified by
brute force (`tests/test_actions.py`). A game between weak players ends much sooner: random 3-player
games last about 18 turns.

### Two candidate sets (`--candidates`)

* `legal` (default): only actions the rules allow now. Nothing impossible can be chosen, so the
  only thing measured is the quality of the choice.
* `all`: every combination above, including hints that would touch no card and, with no token left,
  every hint. Choosing an impossible action is a **rejected action**: the ranking is walked from the
  best down, each impossible choice counts, and after 3 in a row (`--max-failed-actions`) the seat is
  at fault, the same budget the chat players have. This also measures whether the model understands
  the rules.

The wording is identical in both sets and never says what a hint would target
(`hint Bob: green`), so the text cannot reveal which hints are legal.

## Fairness controls

* **Position bias.** Rerankers can favour early documents, so the documents are sent in a seeded
  shuffle, a pure function of (game seed, turn, number of documents): reproducible, and the same for
  every model facing the same turn of the same deal. `--no-shuffle` keeps the canonical order.
* **Ties** go to the lowest canonical index, whatever order was sent.
* **No hidden information.** The query and documents depend only on what the player may see; a test
  sends two games that differ only in the player's own cards and the deck and requires identical
  input.
* **Faults.** In `legal` mode a rerank seat cannot make a model fault at all: a provider problem is
  an infrastructure fault (the game is void, nobody is blamed); a nonsense response is also the
  provider's.

## What this does and does not measure

A reranker scores how *relevant* a document is to a query; it does not plan. Expect weak play, and
read results against two references, not in isolation:

* **Random play** (the floor): 3 players, 300 games, mean stack score **1.28** (best 9); every game
  ended by losing all fuses, after about 18 turns.
* **The chat baseline** with the same seeds.

Use the same seeds across models (paired deals) and many games per model; one game says little.

## Not done

* The response field names are unconfirmed against a live call (see above).
* No capability pre-flight: a model that is not a rerank model fails every call (404 or 400).
* The per-model query length limit is unknown; the query is kept short (about 800 tokens at most).
* Cost is one "search" per turn, 60 to 72 per complete game; check the model's price per search.
* Notes for strategy (`--knowledge`) are not offered here: they need a tool a reranker cannot call.
  They could be added to the query later as a separate arm.

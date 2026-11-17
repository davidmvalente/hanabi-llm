# First tested draft

Hanabi played by tool-calling language models, refereed by an engine whose rules and tools are
declared in an RDF ontology, with a full audit trail (a replayable feed, a live viewer) and a
fairness-first design. This file records what exists at this milestone (the commit that introduced it,
on the `first-draft` branch), how it was tested, and what it deliberately does not do yet.

**Status.** 188 automated tests, `mypy --strict` clean, a 23-check headless-browser run, and the
whole CLI exercised over real HTTP against a local OpenAI-compatible server. **It has never been
run against a real model**: that is the first thing to do with real keys (see "Next").

## What it does

A game of 2 to 5 players where each seat is a model (or a bot). A turn is: the model receives the
state, may call free read tools (`get_state`, `explain_rules`), then takes exactly one action
(`play_card`, `discard_card`, `give_hint`). The game is logged as a JSONL **feed** that a browser
**viewer** can follow live or afterwards.

```
hanabi.ttl ─► ontology ─► Game (decides) ─► fold.step (applies) ─► state ─► views ─► tools ─► players ─► runner
 rules, tools   typed       validates,        the ONE state        truth    one seat   schemas,   LLM turn   referee,
 epistemics     loader      resolves, draws   machine                       per view   dispatch   loop       faults
                                                  │
                                                  └─► feed (JSONL) ─► viewer (SSE) ─► browser
```

## How to run

```bash
uv sync
uv run pytest                                                          # 188 tests, no network
uv run hanabi-llm --models random,random,random --seed 3 --log g.jsonl # offline, 3 bots
uv run hanabi-view g.jsonl                                             # watch it: http://127.0.0.1:8765

export OPENROUTER_API_KEY=...                                          # a real baseline run
uv run hanabi-llm --models <id>,<id>,<id> --seed 1 --log game.jsonl --trace trace.jsonl
uv run hanabi-llm --models <id>,<id>,<id> --knowledge mdk              # an arm with strategy notes
uv run hanabi-llm --models random,random --audit                       # SHACL-validate every turn (slow)
scripts/check_viewer.sh                                                # headless-browser check (Node + playwright)
```

Exit codes: 0 completed, 2 infrastructure fault (re-run it), 3 model fault (that seat is to blame).

## Design decisions (and where each is explained)

| Decision | Why | Details |
|---|---|---|
| Faults abort the game; no substitute moves. Model faults and infrastructure faults are separate; only completed games rank. | A model is judged on what it controls. | `docs/AUDIT.md` |
| Seats are named Alice, Bob, ...; the model id is only in the log header. | Models must not identify each other. | `docs/AUDIT.md` |
| Tools are exactly three families (action, state, rules), mutually exclusive and exhaustive. | Minimal and measurable; no helper does the thinking for the model. | `docs/AUDIT.md` |
| The system prompt is a versioned YAML with objective and protocol only; rules are learned through tools. | One channel for the rules; the prompt cannot drift from the ontology. | `prompts/system.v1.yaml` |
| The baseline has no strategy knowledge; knowledge graphs are optional, logged arms. | Compare with and without. | `docs/KNOWLEDGE.md` |
| Who may know what is declared once in the ontology and enforced by SHACL and non-interference tests. | Incomplete information is the heart of the game. | `docs/EPISTEMICS.md` |
| A hint is one value (`give_hint(player, hint)`); it covers every matching card, so its targets are a subset of the hand. | Invalid states are unrepresentable. | `docs/AUDIT.md` |
| The engine decides; the fold applies. One implementation of what an action does to the state. | No drift between engine, feed, viewer and tests. | below |
| Event wording is defined once (`narrate.py`) as typed segments. | Prompt history, CLI and viewer cannot disagree. | `docs/AUDIT.md` |

### The engine decides, the fold applies

`Game.apply` validates an action with the precondition functions the ontology names, **decides**
what happens (a play succeeds if the card is the next one its colour needs; a hint covers every
matching card; the next card is drawn from the secret deck), turns that into events, and applies
them with `fold.step`. `fold.step` is the only code that changes the state; it is the same code
that rebuilds a state from the feed in the viewer and in the tests. `game.state` is a read-only
snapshot of that fold (tuples and read-only mappings), so a mistake raises instead of silently
diverging. `Game.rig(...)` sets up a position for tests and, later, for fixed scenarios.

## How it was tested

| Layer | What the tests guarantee | Tests |
|---|---|---|
| Engine | Rules, legal actions, hint completeness, the final-round clock, the engine holds no transition logic of its own | 30 |
| Fold and feed | After every turn of whole games (2 to 5 players, including games run through the final round) the fold of the feed equals the engine's truth, and the fold of each player's projected feed equals that player's view | 15 |
| Epistemics and views | The declared regime is loaded, refused if unsupported, and interpreted (an open-hand variant works); a view never contains the viewer's own cards, and a drawn card is hidden from the drawer only | 14 |
| SHACL audit | Truth and every view conform at sampled turns of whole games; nine deliberate corruptions are each rejected | 13 |
| Ontology lint | Types, referenced terms, domains and ranges (also over every graph the code emits), dead vocabulary, documentation, and ontology-to-code binding in both directions | 17 |
| Tools and states | Schemas come from the ontology; the six state sections partition the view; token pools stay consistent with the engine; read-tool mistakes are free, rejected actions count | 26 |
| Players and referee | Each way a model can fail; faults abort and are attributed correctly; the whole conversation is byte-identical when only hidden cards differ | 27 |
| Prompt and knowledge | The prompt is validated and contains nothing from the graph; no note text reaches a baseline model; note names never collide with rule names | 21 |
| Narration | The exact wording of every event kind; the viewer is sent the same wording | 14 |
| CLI, viewer, HTTP | The feed and trace; the stream equals the fold; a half-written line waits; the CLI against a local server: a full 3-seat game, a provider outage (no blame), a model that never calls a tool (its fault) | 11 |
| Browser (`scripts/`) | Blinding, scrubbing, wording on the real page, phone width, no external requests, no console errors | 23 checks |

Tests were mutation-checked where it mattered: breaking the redaction fails five tests; re-introducing
each of six ontology defects fails the lint; nine corrupted graphs fail the shapes; breaking the
final-round clock, the refund rule or the projection fails the fold tests; planting a state write in
the engine fails the behaviour tests and the static guard. Three tests were found to be vacuous or
wrong this way and fixed (a fold test that never reached the final round, a leak test that only
matched strings, a rigged deal that correctly failed conservation).

## Known limits

* Never run against a real model; the HTTP path is only tested against a local fake.
* A reply cut off by `max_tokens` looks like "no tool call" and counts against the model.
* No tournament layer: no seat rotation, paired seeds, or aggregation; seat 0 moves first.
* Provider routing is not pinned and there is no capability pre-flight (a model without tool
  support faults on every game).
* The SHACL audit costs about 1.3 s per turn for three seats, so it is opt-in.
* Token use is not optimised: each turn is stateless and resends the full state.
* Scrubbing in the viewer is per event, so a frame can fall between a play and its draw.

## Next

1. A first real baseline run (3 players, the smallest cheapest tool-calling models), then tune the
   prompt wording and look at the faults.
2. Fix truncated-reply handling, add a capability pre-flight and provider pinning.
3. A tournament runner (seat rotation, paired seeds) once a model-fault policy is chosen.
4. Token parsimony (append-only per-seat threads with deltas).
5. The `rerankers-version` branch: decision models that score a list of action strings instead of
   calling tools.

## Changes since this milestone

* The rejected-action budget is 5 (was 3), and a fault now says exactly what was rejected; the
  faulted turn is in the trace; `hanabi-trace` summarises a trace; argument errors say what the tool
  takes. See "What a fault tells you" in `docs/AUDIT.md`. The test count is now 199 (the table above
  is the count at the milestone).
* Cards are plain `(colour, rank)` values with no id; a hand is a row of **slots** and a vacated slot is
  refilled in place (`docs/AUDIT.md`: "Cards are values", "Slots"). The test count is now 212.

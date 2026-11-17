# hanabi-llm

A simulation of Hanabi, a cooperative limited information exchange game

Hanabi played by tool-calling language models (via OpenRouter), refereed by an
engine whose rules and affordances are declared in an RDF ontology.

```bash
uv sync
uv run pytest                       # 59 tests, no network needed
uv run hanabi-llm --models random,random,random --seed 3          # offline bots
export OPENROUTER_API_KEY=...       # then, for real models (one per seat, 2-5):
uv run hanabi-llm --models <openrouter-model-id>,<openrouter-model-id> --seed 7 --log game.jsonl
# exit code: 0 completed, 2 infrastructure fault (re-run), 3 model fault (that seat is to blame)
```

## Layout

| File | Role |
|---|---|
| `hanabi.ttl` | Ontology + constants + card distribution + affordances (the world model) |
| `ontology.py` | Loads the TTL into typed objects; `verify_bindings` checks every function it names exists |
| `registry.py` | `@bind("name")` registry that connects ontology names to Python functions |
| `engine.py` | Authoritative state, legal-move enumeration, rule enforcement, event log |
| `views.py` | Per-seat redacted projection -> RDF graph + prompt text (the only place secrets are removed) |
| `tools.py` | Affordances -> OpenAI-style tool schemas; validates calls; read tools (state, rules, concepts, odds, SPARQL, notes) |
| `estimates.py` | Card-counting odds for a player's own cards, computed from their view only |
| `faults.py` | Who is to blame when a game cannot finish (model vs infrastructure) |
| `llm.py` | Async OpenRouter client (retries, typed result) |
| `players.py` | `LLMPlayer` (tool loop for one turn), `RandomPlayer` |
| `runner.py` | Referee loop: only write path to the engine; aborts (never patches) on faults |

## Key ideas

* **Rules as data.** Colours, copies per rank, hand sizes, token limits, each
  action's parameters, preconditions (`hn:requires -> hn:checkFunction`) and
  effects (`hn:effectFunction`) live in the TTL. Tool schemas and the rules
  text in the system prompt are generated from it.
* **One truth, many views.** There is a single authoritative state. Each seat
  gets a projection of it; the common part is identical in every projection
  because it comes from the same source, so there is nothing to synchronise.
* **Pointers to callable functions.** In each view graph, the active player has
  `hn:canInvoke` edges to affordance nodes, which carry `hn:functionName`.
* **Hints are the only channel.** Free text from a model is never forwarded.
* **Referee, not trust.** Actions are validated by `Game.check` inside the tool
  (so the model gets an error and can retry) and again by `Game.apply`.
* **No substitute moves.** A player that exhausts its rejected-action budget or
  its call budget aborts the game as a *model fault*; a provider failure or
  timeout aborts it as an *infrastructure fault*. Aborted games are never
  ranked. See `docs/AUDIT.md`.
* **Game knowledge in the graph.** Mechanics, dynamics and kinetics (mass,
  inertia, torque, activation energy, collision rate...) are `hn:Concept`
  nodes in the ontology, readable with `get_concepts` or SPARQL.
* **A turn.** Read tools (`get_state`, `get_rules`, `get_concepts`,
  `estimate_card`, `query_knowledge`, `take_notes`) are free and repeatable
  within a per-turn call budget; the first accepted action tool ends the turn.

## Conventions worth knowing

* Hand positions are 0-based. A played or discarded card leaves a gap that
  closes (later cards shift down one); the replacement enters at the last slot.
* Hint knowledge includes negative information (a colour hint that skipped a
  card tells its holder the card is not that colour).
* Losing all fuses ends the game; `official_score` is 0 in that case,
  `score` is the sum of the stacks.
* Seats are named Alice, Bob, ... never after the model, so models cannot
  identify each other. The seat -> model map is only in the log header.
* Each model turn is stateless apart from the model's private `take_notes`
  memory: the full current view is re-sent every turn (v1.1 plan: `docs/AUDIT.md`).

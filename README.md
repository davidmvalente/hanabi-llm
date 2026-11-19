# hanabi-llm

Hanabi played by tool-calling language models (via OpenRouter), refereed by an
engine whose rules and affordances are declared in an RDF ontology.

```bash
uv sync
uv run pytest                       # 216 tests, no network needed
uv run hanabi-llm --models random,random,random --seed 3          # offline bots
export OPENROUTER_API_KEY=...       # then, for real models (one per seat, 2-5):
uv run hanabi-llm --models <openrouter-model-id>,<openrouter-model-id> --seed 7 --log game.jsonl
# exit code: 0 completed, 2 infrastructure fault (re-run), 3 model fault (that seat is to blame)
```

> Status and how it was tested: [`docs/FIRST_DRAFT.md`](docs/FIRST_DRAFT.md). Never yet run against a real model.

## Layout

| File | Role |
|---|---|
| `hanabi.ttl` | Ontology + constants + card distribution + affordances (the world model) |
| `ontology.py` | Loads the TTL into typed objects; `verify_bindings` checks every function it names exists |
| `registry.py` | `@bind("name")` registry that connects ontology names to Python functions |
| `engine.py` | Authoritative state, legal-move enumeration, rule enforcement, event log |
| `views.py` | Per-seat redacted projection -> RDF graph + prompt text (the only place secrets are removed) |
| `tools.py` | Affordances -> OpenAI-style tool schemas; validates actions; dispatches the read tools |
| `narrate.py` | The one wording of events (typed segments, flattened to text), shared by the prompt history, the CLI and the viewer |
| `knowledge/` | Optional knowledge graphs (notes models may consult); none in the baseline. See `docs/KNOWLEDGE.md` |
| `feed.py`, `fold.py` | The game feed (JSONL), the observer projection, and the fold: the one state machine, used by the engine, the viewer and the tests |
| `scripts/` | A heuristic demo bot and the headless-browser check of the viewer (`scripts/check_viewer.sh`) |
| `viewer.py`, `web/` | Live viewer: follows a feed and streams folded states to a one-page browser UI |
| `validate.py`, `shapes/` | Ground-truth graph and SHACL audit: structure, conservation and the epistemic regime, checked on every graph |
| `states.py` | The state a player may see, as six sections that partition it (turn, tokens, fireworks, discard, hands, history) |
| `explain.py` | Rule and action explanations, generated from the ontology |
| `faults.py` | Who is to blame when a game cannot finish (model vs infrastructure) |
| `llm.py` | Async OpenRouter client (retries, typed result) |
| `prompt.py`, `prompts/system.v1.yaml` | The system prompt: versioned YAML, a few run-setting variables, nothing read from the graph |
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
* **Five tools, three families** (mutually exclusive, together exhaustive):
  *action* (`play_card`, `discard_card`, `give_hint`; the first accepted one
  ends the turn), *state* (`get_state`, optionally one `section`) and *rules*
  (`explain_rules`, optionally one `topic`: an action or a rule, with its
  rationale and an example). Read tools are free within a per-turn call budget.
* **The system prompt is a versioned YAML file** (`prompts/system.v1.yaml`):
  objective and turn protocol only, with four placeholders (`name`, `players`,
  `max_calls`, `max_failed_actions`). It states no rules, constants or tool
  lists: models learn those through `explain_rules` and `get_state`. Released
  versions are never edited; add `system.v2.yaml` instead. The version and
  sha256 go in every log header (`--prompt v1`).
* **The baseline has no strategy knowledge.** Optional knowledge graphs (`--knowledge mdk`) add a
  `get_concepts` tool so that arms can be compared; the choice is logged in the feed header.
  `--api-url` points the same pipeline at any OpenAI-compatible endpoint.
* **Events read as sentences, with the card's colour in the text:** `T7 Alice hinted Carol:
  slot 3 is green`, `T8 Bob discarded blue 1 from slot 1 and drew red 2 into slot 1`, `T10 Alice played
  blue 2 from slot 2 -> misfire (fuse lost)`. One definition (`narrate.py`) serves the prompt history, the
  CLI and the viewer.
* **A hint is one value.** `give_hint(player, hint)`, where `hint` is a colour or a rank; the
  attribute follows from the value, so "both" or "neither" cannot be written. The hint always
  covers every card of the receiver's hand with that value, so its *targets* are a subset of the
  hand, and every other card is thereby known not to match. In the feed:
  `{"kind":"hint","actor":0,"receiver":2,"attribute":"color","value":"green","targets":[3]}`.
* **The log is a feed, and the viewer follows it.** `--log game.jsonl` writes a header (with the
  initial deal), one record per event and a result, flushed line by line; `--trace` writes the
  model conversations separately. Watch a game live or afterwards:

  ```bash
  uv run hanabi-llm --models random,random,random --seed 3 --log game.jsonl --trace trace.jsonl &
  uv run hanabi-view game.jsonl            # open http://127.0.0.1:8765
  uv run hanabi-trace trace.jsonl          # what each model did, and exactly what was rejected
  ```

  Everything is visible by default; "Blind a hand" turns one player's cards face-down so you see
  what that player sees. Scrub with the slider or the arrow keys; "follow live" tracks the end.
* **Epistemics is declared once** (ontology section 7: who may see the identity of
  the cards in each zone), interpreted in `views.build_view`, and enforced by SHACL
  shapes and non-interference tests. `docs/EPISTEMICS.md` has the four layers and
  where each lives. `--audit` SHACL-validates the truth and every view each turn.
* **One vocabulary for tokens.** Hint and fuse tokens are both a pool with
  `available`, `spent` and `capacity`, where `spent = capacity - available`;
  refunds just raise `available`.
* **Deliberately left out of experiment 1:** card-odds helpers, a card census,
  strategy notes (the mechanics/dynamics/kinetics concepts stay in the
  ontology as data), private notes and dry-run helpers. Each changes what the
  experiment measures; see `docs/AUDIT.md`.

## Conventions worth knowing

* A hand is a row of **slots**, numbered from 0. A played or discarded card is replaced from the deck
  **in the same slot**, so the other slots never move and what you were told about a slot stays valid.
  Only when the deck is empty is there nothing to refill with: the slot is removed and the later slots
  shift down by one.
* Hint knowledge includes negative information (a colour hint that skipped a
  card tells its holder the card is not that colour).
* Losing all fuses ends the game; `official_score` is 0 in that case,
  `score` is the sum of the stacks.
* Seats are named Alice, Bob, ... never after the model, so models cannot
  identify each other. The seat -> model map is only in the log header.
* Each model turn is stateless: the full current state is re-sent every turn
  and models have no private memory (v1.1 plan: `docs/AUDIT.md`).

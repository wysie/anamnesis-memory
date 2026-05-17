# Anamnesis

Local-first governed memory for agents.

Anamnesis is an experimental memory layer focused on privacy, scope enforcement, contradiction handling, decay, and explainable recall. It starts with a deterministic SQLite/FTS core; local LLM maintenance workers are optional and must never become the source of truth.

Status: early scaffold.

## Goals

- Keep memory content local by default.
- Separate raw evidence, episodic memory, semantic memory, entity graph, and governance state.
- Make recall explainable: every recalled item should say why it appeared.
- Enforce owner/platform/visibility/action scopes before any agent sees memory.
- Treat contradictions, invalidation, supersession, and decay as first-class lifecycle states.
- Memory Inbox for proposed changes instead of blind auto-write.
- Deterministic duplicate hints for candidate memories.
- Make local LLMs optional maintenance workers, not mandatory runtime dependencies.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Minimal usage

```python
from anamnesis import Anamnesis

store = Anamnesis("anamnesis.db")
store.add_memory(
    "Primary user prefers local-only private memory.",
    owner="primary",
    visibility="private",
    platform_scope="all",
    domain="privacy",
    source="whatsapp",
    metadata={"source_platform": "whatsapp"},
    importance=0.9,
)

results = store.recall(
    "local memory",
    owner="primary",
    platform="whatsapp",
    allowed_visibility={"private"},
)

for result in results:
    print(result.record.text, result.score, result.reasons)
```

`platform_scope` controls where a memory may be recalled; `source` / `metadata.source_platform` record where it came from. Durable user/profile/project facts should normally use `platform_scope="all"` for the same canonical owner so a memory captured on WhatsApp can help later on Telegram or CLI. Use platform-specific scopes such as `whatsapp` only for explicitly platform-local or sensitive facts. Provider autopilot defaults conversation-derived durable facts to `all`, but keeps “only on WhatsApp/Telegram/etc.” facts and sensitive secrets on the current platform/inbox path.

## Optional embeddings

FTS recall works without any model. For semantic/paraphrase recall, provide a local embedder, embed missing rows, then pass the same embedder into recall. Scope and invalidation filters still run before vector scoring.

Embeddings are a rebuildable search cache, not canonical memory data. The `memories` table stays model-independent, while `memory_embeddings` is keyed by `(rid, model_id)` and stores the model dimension with each vector. If you switch from `potion-base-2M` to `potion-base-8M`, `potion-base-32M`, or `potion-retrieval-32M`, Anamnesis keeps the existing memories and old embedding cache, reports coverage for the newly active model, and backfills only missing vectors for that model. The active embedder is authoritative: recall does not automatically cascade through smaller models unless a future advanced mode explicitly implements and benchmarks that behaviour.

```python
from anamnesis import KeywordEmbedder

embedder = KeywordEmbedder(
    dimensions=("car", "vehicle", "wash", "clean"),
    synonyms={"vehicle": "car", "clean": "wash"},
)
store.embed_missing(embedder)
results = store.recall(
    "Who cleans the vehicle?",
    owner="primary",
    platform="whatsapp",
    allowed_visibility={"private"},
    embedder=embedder,
)
```

Production embedders only need to expose `model_id`, `dimension`, and `embed(text)`. Keep them local/private; Anamnesis does not require cloud embeddings.

## Official embedding model downloads

Anamnesis follows a release-asset model distribution pattern rather than making Hugging Face the default path:

- FTS-only recall remains the no-network fallback (`ANAMNESIS_RECALL_MODE=fts`).
- Embedder recall is the default mode (`ANAMNESIS_RECALL_MODE=embedder`) and defaults to `potion-base-2M` unless `ANAMNESIS_EMBEDDER` overrides it.
- Official model names resolve through an in-code registry with dimensions, release tags, asset URLs, and SHA-256 checksums.
- Models download only on first use, only when `ANAMNESIS_ALLOW_MODEL_DOWNLOADS=1` is set.
- Archives are verified before extraction and cached in the platform cache dir:
  - macOS: `~/Library/Caches/anamnesis/models/`
  - Linux: `~/.cache/anamnesis/models/` or `$XDG_CACHE_HOME/anamnesis/models/`
  - Windows: `%LOCALAPPDATA%/anamnesis/Cache/models/`
- Known registry models infer dimensions automatically; callers do not need to set a separate dimension env var.
- Local paths are supported for BYO/offline experiments. Hugging Face-style loaders can be added as optional advanced backends, not the default.

```bash
ANAMNESIS_ALLOW_MODEL_DOWNLOADS=1 \
python -c 'from anamnesis import load_embedder_from_env; print(load_embedder_from_env().model_id)'
```

Use FTS-only mode explicitly:

```bash
ANAMNESIS_RECALL_MODE=fts python -c 'from anamnesis import load_embedder_from_env; print(load_embedder_from_env())'
```

Manage embedding cache coverage with the CLI:

```bash
anamnesis --db ~/.anamnesis/anamnesis.db embeddings switch potion-base-32M
anamnesis --db ~/.anamnesis/anamnesis.db embeddings status
anamnesis --db ~/.anamnesis/anamnesis.db embeddings status --json
ANAMNESIS_ALLOW_MODEL_DOWNLOADS=1 anamnesis --db ~/.anamnesis/anamnesis.db embeddings backfill --model potion-base-32M
anamnesis --db ~/.anamnesis/anamnesis.db embeddings index-status --model potion-base-32M --json
anamnesis --db ~/.anamnesis/anamnesis.db embeddings index-rebuild --model potion-base-32M
anamnesis --db ~/.anamnesis/anamnesis.db recall-config set --model potion-base-32M --ann-backend sqlite-vec --recall-policy latency_first --ann-candidate-limit 50 --vector-candidate-limit 1000
anamnesis --db ~/.anamnesis/anamnesis.db recall "What is the user privacy preference?" --owner primary --platform whatsapp --explain --json
ANAMNESIS_ALLOW_MODEL_DOWNLOADS=1 anamnesis embeddings benchmark --models potion-base-2M,potion-base-8M,potion-base-32M,potion-retrieval-32M --synthetic-count 1000 --include-adversarial --vector-candidate-limit 1000 --ann-candidate-limit 200 --ann-backend sqlite-vec --recall-policy latency_first --json
```

`embeddings status` compares active memories against the active model's exact `model_id` and dimension. A non-zero `missing` count means semantic vectors for that model need backfill; FTS fallback remains available while the index warms. `embeddings index-status` compares cached embeddings with the sqlite-vec derived index and reports both missing vectors and stale governance metadata; `index-rebuild` rebuilds that per-model vector index from canonical cached embeddings plus governance metadata (owner, visibility, platform scope, status, domain) without re-embedding memories. `recall-config set` persists runtime recall defaults for the model, sqlite-vec DB, policy, and candidate limits. `recall ... --explain --json` runs governed recall using those defaults and includes the applied config and basic execution metadata. Recall applies a small deterministic intent-expansion layer before FTS/vector scoring for low-overlap paraphrases such as child/asleep/quiet-audio, buyer/outcome wording, transient operational chatter, and okay/go-ahead action signals; expanded hits are marked with `semantic_intent_expansion`. `embeddings benchmark` runs each selected model alone in an isolated fixture DB and reports backfill time, backfill throughput, DB size, recall p50/p95 latency, and a small recall score. `--synthetic-count` adds distractor memories for scale checks, `--include-adversarial` adds privacy/invalidation cases, `--vector-candidate-limit` enables candidate-pruned vector scoring from top keyword candidates, and `--ann-candidate-limit` adds candidates from a `VectorIndex` backend. The default backend is `sqlite-vec`; use `--ann-backend exact` for the dependency-free exact test backend. Recall policy options are `latency_first` (default: use sqlite-vec only when keyword candidates are below `--ann-min-keyword-candidates`), `recall_first` (always merge keyword and sqlite-vec candidates), and `semantic_only` (skip FTS-only results and use vector-index candidates). The benchmark deliberately does not enable tiered/cascade recall.

Current official registry entries:

| Model | Dim | Release | Source |
|---|---:|---|---|
| `potion-base-2M` | 64 | `v0.1.0` | `anamnesis-memory/anamnesis-models` release asset |
| `potion-base-8M` | 256 | `v0.1.0` | `anamnesis-memory/anamnesis-models` release asset |
| `potion-base-32M` | 512 | `v0.1.0` | `anamnesis-memory/anamnesis-models` release asset |
| `potion-retrieval-32M` | 512 | `v0.2.0` | `anamnesis-memory/anamnesis-models` release asset |

## Local/private LLM synthesis

Anamnesis has its own local/private synthesis configuration. Do not reuse the host agent's model/provider config by default: memory synthesis is privacy-sensitive and should be pinned to a deliberately configured OpenAI-compatible endpoint.

Store the endpoint and model in the Anamnesis DB. Store secrets in an environment variable and point Anamnesis at that variable; omit `--api-key-env` for no-auth local servers.

```bash
anamnesis --db ~/.anamnesis/anamnesis.db synthesis-config set \
  --base-url http://127.0.0.1:8060/v1 \
  --model local-private-model \
  --api-key-env ANAMNESIS_LLM_API_KEY \
  --temperature 0 \
  --max-tokens 512 \
  --max-context-chars 8000 \
  --max-memory-chars 1200

ANAMNESIS_ALLOW_MODEL_DOWNLOADS=1 anamnesis --db ~/.anamnesis/anamnesis.db synthesize \
  "What is the user privacy preference?" \
  --owner primary \
  --platform whatsapp \
  --json
```

Synthesis is read-only: it takes already-governed recall results, sends only a score-ordered, character-budgeted source pack to the configured local/private LLM, and asks the model to cite memory IDs inline. If an answer has factual text but no memory citations, Anamnesis retries once and then marks `citation_missing=true` in JSON output if citations are still absent. Clearly unsupported sensitive queries are refused before calling the LLM with `insufficient_evidence=true`. It does not write, update, delete, or propose memory changes. Memory mutation remains a separate governed Inbox/admin workflow.

## Memory Inbox

Use the inbox when a fact might be worth saving but should not be blindly promoted:

```python
item = store.propose_memory(
    "User prefers local-first memory.",
    source_snippet="Please keep my memory local.",
    owner="primary",
    visibility="private",
    platform_scope="whatsapp",
    domain="privacy",
    why_save="Durable user preference",
)

# Review item.duplicate_rids and item.hints before accepting.
record = store.accept_inbox_item(item.cid)
```

## Preview checks

Use `preview-turn` to preview what Anamnesis would save, inbox, reject, and inject for a turn without mutating the DB. The same Preview terminology is used by the CLI, dashboard, API, and backend.

```bash
anamnesis --db ~/.anamnesis/anamnesis.db preview-turn \
  "Primary user prefers cross-platform memory by default." \
  --owner primary \
  --platform whatsapp \
  --json
```

Use `preview-batch` to preview a JSONL/text transcript. Add `--apply` only when you intentionally want accepted items written and inbox candidates proposed:

```bash
anamnesis --db ~/.anamnesis/anamnesis.db preview-batch transcript.jsonl \
  --owner primary \
  --platform whatsapp \
  --json
```

Run safe maintenance and inspect recent maintenance history:

```bash
anamnesis --db ~/.anamnesis/anamnesis.db maintenance autopilot --json
anamnesis --db ~/.anamnesis/anamnesis.db maintenance report --json
```

## Contradictions

Deterministic contradiction checks flag simple polarity conflicts without auto-resolving them:

```python
store.add_memory("The project deploys on Fridays.", owner="primary", domain="project")
store.add_memory("The project must not deploy on Fridays.", owner="primary", domain="project")

conflicts = store.detect_contradictions(owner="primary", domain="project")
resolved = store.resolve_contradiction(
    conflicts[0].conflict_id,
    winner_rid=conflicts[0].right_rid,
    reason="newer correction",
)
```

## Recall benchmarks

Use recall benchmarks to prove scope safety and retrieval quality before claiming Anamnesis is better than another memory layer:

```python
from anamnesis import RecallBenchmarkCase, run_recall_benchmark

report = run_recall_benchmark(
    store,
    [
        RecallBenchmarkCase(
            name="privacy preference",
            query="local private memory",
            owner="primary",
            platform="whatsapp",
            allowed_visibility={"private"},
            expected_rids={record.rid},
        )
    ],
)

assert report.failed == 0
```

## Core behavior benchmark suite

The built-in suite seeds a small privacy/safety fixture and verifies the behaviours Anamnesis must preserve:

- owner scope: private memories stay invisible to other-owner contexts
- platform scope: recall only returns memories whose scope is `all` or includes the current platform; source provenance is stored separately
- invalidation: invalidated memories do not recall
- rejected inbox items: temporary task state stays out of durable recall
- contradiction resolution: winner recalls, loser is invalidated
- duplicate prevention: duplicate candidates are flagged but not auto-accepted

```python
from anamnesis.behavior_benchmarks import build_core_behavior_suite, seed_core_behavior_fixture

fixture = seed_core_behavior_fixture(store)
report = build_core_behavior_suite(store, fixture).run()
assert report.failed == 0
```

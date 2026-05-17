# Anamnesis Design

> Local-first governed memory for agents.

## Honest positioning

Anamnesis is not guaranteed to beat every existing memory system on day one. It should earn replacement status by outperforming current baselines on privacy, scope enforcement, explainability, contradiction handling, lifecycle hygiene, dashboard governance, and recall benchmarks.

The design goal is not “more recall”. The design goal is “safer, cleaner, explainable recall with memory governance”.

## Non-goals for v0.1

- No cloud calls.
- No mandatory LLM.
- No hidden prompt injection into agents.
- No automatic permanent memory writes without policy checks.
- No replacement of existing systems until benchmarks show improvement.

## Core principles

1. **Local-first:** memory content stays on the machine.
2. **Deterministic core:** core storage, ranking, scoping, invalidations, and recall work without an LLM.
3. **LLM optional:** a local LLM may propose summaries, entities, merges, conflicts, and lifecycle classes, but it is never the source of truth.
4. **Governance first:** contradictions, decay, provenance, scopes, and invalidations are first-class.
5. **Explain every recall:** every recall result includes reasons and penalties.
6. **Fail closed on scope:** if a memory’s visibility/owner/platform/action scope is uncertain, do not inject it.
7. **Raw evidence is not memory:** transcripts are audit/evidence; semantic memory is compact, durable, and curated.

## Memory layers

### 1. Raw evidence

Append-only transcript/events. Used for audit and reconstruction. Not normally injected.

### 2. Episodic memory

Session/project summaries. Useful for “what happened last time?” style recall.

### 3. Semantic memory

Durable facts, preferences, conventions, constraints, and project decisions.

### 4. Entity graph

People, projects, tools, places, and relationships. Used for scoped recall and neighborhood expansion.

### 5. Governance layer

Contradictions, invalidations, decay, pending memory inbox items, audit log, source provenance, and review state.

## Data model v0.1

### memories

Canonical curated memory rows.

- `rid`: stable UUID-like id
- `text`: declarative memory text
- `kind`: `semantic | episodic | evidence | entity_note`
- `owner`: actor/person/profile that owns the memory
- `visibility`: `private | household | team | public | agent_internal`
- `platform_scope`: comma-separated recall scope string, e.g. `all`, `whatsapp`, `telegram`, `cli`; this is authorization for where the memory may be injected, not provenance
- `action_scope`: `read_only | can_act | cannot_act`
- `domain`: optional domain/project tag
- `source`: source system/importer
- `importance`: 0.0–1.0
- `confidence`: 0.0–1.0
- `status`: `active | invalidated | superseded | pending` (displayed as Active, Invalidated, Superseded, Pending)
- `created_at`, `updated_at`, `last_access`: unix timestamps
- `ttl_days`: optional suggested decay TTL
- `metadata_json`: optional JSON payload, including provenance fields such as `source_platform` when the memory was captured from a platform-specific turn

Platform provenance and platform recall scope are intentionally separate: a WhatsApp-derived durable memory can have `metadata_json.source_platform="whatsapp"` and `platform_scope="all"`, allowing the same canonical owner to recall it later from Telegram/CLI while still preserving where it came from. Use platform-specific scope only for facts that are explicitly platform-local or sensitive. Provider autopilot applies this split by default for ordinary durable facts, but sends sensitive content to inbox/current-platform scope and keeps “only on <platform>” facts current-platform scoped.

### memory_fts

FTS5 virtual table over `text`, `domain`, `source`, `owner`.

### entities

- `eid`
- `name`
- `type`
- `metadata_json`

### relationships

- `source_eid`
- `target_eid`
- `relationship`
- `confidence`
- `source_rid`

### contradictions

- `cid`
- `left_rid`
- `right_rid`
- `reason`
- `status`: `open | resolved | dismissed`
- `resolution`

### memory_inbox

Candidate memories proposed by rules, imports, or optional local LLM.

- `cid`
- `proposed_text`
- `proposed_kind`
- `source_snippet`
- `source`
- `confidence`
- `decision`: `pending | accepted | rejected`
- `review_reason`

### audit_log

Append-only records of writes, invalidations, merges, imports, recalls, and admin changes.

## Recall pipeline v0.1

1. Build `RecallContext` from caller:
   - `query`
   - `owner`
   - `platform`
   - `allowed_visibility`
   - `action_intent`
   - `domain`
   - `limit`
2. Apply hard scope filters before ranking.
3. Retrieve candidates via FTS5, and optionally via local embeddings/vector similarity when an embedder is supplied.
4. Score candidates deterministically:
   - FTS relevance
   - vector cosine similarity
   - importance
   - confidence
   - recency/decay
   - domain match
   - owner/platform/visibility match
   - status penalties
5. Return `RecallResult` objects with reasons:
   - `keyword_match`
   - `important`
   - `recent`
   - `domain_match`
   - `scope_match`
   - `decay_penalty`
6. Update `last_access` and audit the recall.

Vector search is optional in v0.2: callers may pass a local embedder to `recall(...)`, and `embed_missing(...)` stores vectors keyed by `(rid, model_id)` with the vector dimension recorded per row. Canonical memory rows are model-independent; switching from 2M to 8M/32M is therefore an embedding-cache backfill/reindex, not a memory database migration. The configured embedder is authoritative for recall: Anamnesis does not automatically tier/cascade through smaller models, because different model sizes may have different dimensions/vector spaces. Scope, visibility, platform, domain, invalidate, and suppression filters still run before vector scoring, so semantic similarity cannot bypass governance. FTS-only recall remains the dependency-free fallback while a new active model is being backfilled.

## Embedding model distribution

Official Anamnesis embedding models should use a release-asset flow, not Hugging Face as the default distribution path:

- FTS-only recall remains available with no model and no network via `ANAMNESIS_RECALL_MODE=fts`.
- Embedder recall is the default mode (`ANAMNESIS_RECALL_MODE=embedder`) and defaults to `potion-base-2M` unless `ANAMNESIS_EMBEDDER` overrides it.
- Official model names are declared in an in-code registry with model name, dimension, release tag, asset URL, architecture, and SHA-256.
- First-use download requires explicit opt-in (`ANAMNESIS_ALLOW_MODEL_DOWNLOADS=1`).
- Downloads are verified before extraction and cached under the platform user cache directory.
- Known registry entries infer dimension automatically; no separate dimension env var is required.
- Local paths are supported for offline/BYO experiments. Hugging Face loaders are future optional advanced backends, not the default.

## Optional local LLM worker

The local LLM is a background worker, not a dependency of recall.

Suggested first model class: 4B–8B instruct, Q5_K_M if using GGUF.

Allowed jobs:

- candidate memory extraction
- episodic summaries
- entity/relationship proposals
- duplicate/merge suggestions
- contradiction proposals
- lifecycle classification

Forbidden jobs:

- bypassing hard scope filters
- directly writing permanent memory without policy
- deciding final recall injection alone
- sending memory content to cloud APIs

## Benchmarks before claiming “better”

Anamnesis must be compared with current baselines on:

- recall precision for known user/project facts
- duplicate rate
- stale-memory injection rate
- contradiction detection rate
- permission/scope leakage rate
- latency
- dashboard/debuggability
- import fidelity

## v0.1 acceptance criteria

- Initialize local SQLite database.
- Add semantic memory with owner/platform/visibility metadata.
- Recall by keyword through FTS5.
- Enforce scope before ranking.
- Invalidate memory and exclude it from recall.
- Return explainable recall reasons.
- Record recall/write/invalidate audit events.
- Full test coverage for the above.

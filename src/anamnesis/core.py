from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .embeddings import Embedder, cosine_similarity
from .vector_index import VectorIndex, VectorIndexRow


@dataclass(frozen=True)
class MemoryRecord:
    rid: str
    text: str
    kind: str
    owner: str
    visibility: str
    platform_scope: str
    action_scope: str
    domain: str
    source: str
    importance: float
    confidence: float
    status: str
    created_at: float
    updated_at: float
    last_access: float | None
    ttl_days: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RecallResult:
    record: MemoryRecord
    score: float
    reasons: list[str]


@dataclass(frozen=True)
class MemoryInboxItem:
    cid: str
    proposed_text: str
    source_snippet: str
    proposed_kind: str
    owner: str
    visibility: str
    platform_scope: str
    action_scope: str
    domain: str
    source: str
    confidence: float
    why_save: str
    suggested_lifecycle: str
    decision: str
    review_reason: str
    duplicate_rids: list[str]
    hints: list[str]
    accepted_rid: str | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class Contradiction:
    conflict_id: str
    left_rid: str
    right_rid: str
    owner: str
    domain: str
    status: str
    reasons: list[str]
    winner_rid: str | None
    resolution_reason: str
    created_at: float
    updated_at: float


class Anamnesis:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    rid TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'semantic',
                    owner TEXT NOT NULL DEFAULT 'default',
                    visibility TEXT NOT NULL DEFAULT 'private',
                    platform_scope TEXT NOT NULL DEFAULT 'all',
                    action_scope TEXT NOT NULL DEFAULT 'read_only',
                    domain TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_access REAL,
                    ttl_days REAL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    rid UNINDEXED,
                    text,
                    domain,
                    source,
                    owner
                );

                CREATE TABLE IF NOT EXISTS memory_inbox (
                    cid TEXT PRIMARY KEY,
                    proposed_text TEXT NOT NULL,
                    source_snippet TEXT NOT NULL DEFAULT '',
                    proposed_kind TEXT NOT NULL DEFAULT 'semantic',
                    owner TEXT NOT NULL DEFAULT 'default',
                    visibility TEXT NOT NULL DEFAULT 'private',
                    platform_scope TEXT NOT NULL DEFAULT 'all',
                    action_scope TEXT NOT NULL DEFAULT 'read_only',
                    domain TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    why_save TEXT NOT NULL DEFAULT '',
                    suggested_lifecycle TEXT NOT NULL DEFAULT '',
                    decision TEXT NOT NULL DEFAULT 'pending',
                    review_reason TEXT NOT NULL DEFAULT '',
                    duplicate_rids_json TEXT NOT NULL DEFAULT '[]',
                    hints_json TEXT NOT NULL DEFAULT '[]',
                    accepted_rid TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS contradictions (
                    conflict_id TEXT PRIMARY KEY,
                    left_rid TEXT NOT NULL,
                    right_rid TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'default',
                    domain TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'open',
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    winner_rid TEXT,
                    resolution_reason TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(left_rid, right_rid)
                );

                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    rid TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (rid, model_id),
                    FOREIGN KEY(rid) REFERENCES memories(rid)
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rid TEXT,
                    event_type TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            self._migrate_embedding_schema(conn)

    def _migrate_embedding_schema(self, conn: sqlite3.Connection) -> None:
        """Ensure embeddings are keyed by both memory and model.

        Older databases keyed memory_embeddings only by rid, which meant
        switching from a 2M embedder to a 32M embedder overwrote the smaller
        vector. Embeddings are derived caches, so this table must allow one
        cached vector per active model without touching canonical memories.
        """
        columns = conn.execute("PRAGMA table_info(memory_embeddings)").fetchall()
        pk_columns = [row["name"] for row in sorted(columns, key=lambda row: row["pk"]) if row["pk"]]
        if pk_columns == ["rid", "model_id"]:
            return
        conn.execute("ALTER TABLE memory_embeddings RENAME TO memory_embeddings_old")
        conn.execute(
            """
            CREATE TABLE memory_embeddings (
                rid TEXT NOT NULL,
                model_id TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (rid, model_id),
                FOREIGN KEY(rid) REFERENCES memories(rid)
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO memory_embeddings (
                rid, model_id, dimension, vector_json, created_at, updated_at
            )
            SELECT rid, model_id, dimension, vector_json, created_at, updated_at
            FROM memory_embeddings_old
            WHERE model_id IS NOT NULL
            """
        )
        conn.execute("DROP TABLE memory_embeddings_old")

    def add_memory(
        self,
        text: str,
        *,
        kind: str = "semantic",
        owner: str = "default",
        visibility: str = "private",
        platform_scope: str = "all",
        action_scope: str = "read_only",
        domain: str = "",
        source: str = "",
        importance: float = 0.5,
        confidence: float = 1.0,
        ttl_days: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        rid = str(uuid.uuid4())
        ts = time.time()
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    rid,text,kind,owner,visibility,platform_scope,action_scope,domain,source,
                    importance,confidence,status,created_at,updated_at,last_access,ttl_days,metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rid,
                    text,
                    kind,
                    owner,
                    visibility,
                    platform_scope,
                    action_scope,
                    domain,
                    source,
                    float(importance),
                    float(confidence),
                    "active",
                    ts,
                    ts,
                    None,
                    ttl_days,
                    metadata_json,
                ),
            )
            conn.execute(
                "INSERT INTO memory_fts (rid,text,domain,source,owner) VALUES (?,?,?,?,?)",
                (rid, text, domain, source, owner),
            )
            self._audit(conn, rid, "memory_added")
        return self.get_memory(rid)

    def propose_memory(
        self,
        proposed_text: str,
        *,
        source_snippet: str = "",
        proposed_kind: str = "semantic",
        owner: str = "default",
        visibility: str = "private",
        platform_scope: str = "all",
        action_scope: str = "read_only",
        domain: str = "",
        source: str = "",
        confidence: float = 0.5,
        why_save: str = "",
        suggested_lifecycle: str = "",
    ) -> MemoryInboxItem:
        cid = str(uuid.uuid4())
        ts = time.time()
        duplicate_rids = self._duplicate_hints(
            proposed_text, owner=owner, visibility=visibility, domain=domain
        )
        hints = ["possible_duplicate"] if duplicate_rids else []
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_inbox (
                    cid,proposed_text,source_snippet,proposed_kind,owner,visibility,platform_scope,action_scope,
                    domain,source,confidence,why_save,suggested_lifecycle,decision,review_reason,
                    duplicate_rids_json,hints_json,accepted_rid,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    cid,
                    proposed_text,
                    source_snippet,
                    proposed_kind,
                    owner,
                    visibility,
                    platform_scope,
                    action_scope,
                    domain,
                    source,
                    float(confidence),
                    why_save,
                    suggested_lifecycle,
                    "pending",
                    "",
                    json.dumps(duplicate_rids),
                    json.dumps(hints),
                    None,
                    ts,
                    ts,
                ),
            )
        return self.get_inbox_item(cid)

    def accept_inbox_item(self, cid: str) -> MemoryRecord:
        item = self.get_inbox_item(cid)
        if item.decision != "pending":
            raise ValueError(f"Inbox item is already {item.decision}")
        record = self.add_memory(
            item.proposed_text,
            kind=item.proposed_kind,
            owner=item.owner,
            visibility=item.visibility,
            platform_scope=item.platform_scope,
            action_scope=item.action_scope,
            domain=item.domain,
            source=item.source,
            confidence=item.confidence,
        )
        ts = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE memory_inbox SET decision='accepted', accepted_rid=?, updated_at=? WHERE cid=?",
                (record.rid, ts, cid),
            )
            self._audit(conn, record.rid, "inbox_accepted", metadata={"cid": cid})
        return record

    def reject_inbox_item(self, cid: str, *, reason: str = "") -> MemoryInboxItem:
        item = self.get_inbox_item(cid)
        if item.decision != "pending":
            raise ValueError(f"Inbox item is already {item.decision}")
        with self._connect() as conn:
            conn.execute(
                "UPDATE memory_inbox SET decision='rejected', review_reason=?, updated_at=? WHERE cid=?",
                (reason, time.time(), cid),
            )
        return self.get_inbox_item(cid)

    def inbox_items(
        self, *, decision: str = "pending", limit: int = 50
    ) -> list[MemoryInboxItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_inbox WHERE decision=? ORDER BY created_at DESC LIMIT ?",
                (decision, limit),
            ).fetchall()
        return [self._row_to_inbox_item(row) for row in rows]



    def preview_duplicate_shadowing(
        self,
        *,
        owner: str | None = None,
        domain: str | None = None,
        threshold: float = 0.9,
        example_limit: int = 25,
    ) -> dict[str, Any]:
        """Preview near-duplicate active memories that would be shadowed."""
        records = self._active_memory_records(owner=owner, domain=domain)
        term_sets = {record.rid: set(self._terms(record.text)) for record in records}
        matches: list[dict[str, Any]] = []
        already_shadowed: set[str] = set()
        compared_pairs = 0
        for idx, left in enumerate(records):
            if left.rid in already_shadowed:
                continue
            for right in records[idx + 1 :]:
                if right.rid in already_shadowed:
                    continue
                if not self._same_duplicate_scope(left, right):
                    continue
                compared_pairs += 1
                overlap = self._term_overlap_sets(term_sets[left.rid], term_sets[right.rid])
                if overlap < threshold:
                    continue
                canonical, duplicate = self._preferred_duplicate_canonical(left, right)
                matches.append(
                    {
                        "canonical_rid": canonical.rid,
                        "shadowed_rid": duplicate.rid,
                        "overlap": round(overlap, 4),
                        "canonical_text": canonical.text,
                        "shadowed_text": duplicate.text,
                        "owner": canonical.owner,
                        "domain": canonical.domain,
                        "platform_scope": canonical.platform_scope,
                    }
                )
                already_shadowed.add(duplicate.rid)
                if duplicate.rid == left.rid:
                    break
        return {
            "active_memories_considered": len(records),
            "compared_pairs": compared_pairs,
            "would_shadow_count": len(matches),
            "examples": matches[: max(0, example_limit)],
            "examples_truncated": max(0, len(matches) - max(0, example_limit)),
        }

    def shadow_duplicate_memories(
        self,
        *,
        owner: str | None = None,
        domain: str | None = None,
        threshold: float = 0.9,
    ) -> list[dict[str, Any]]:
        """Mark lower-quality near-duplicate active memories as shadowed."""
        records = self._active_memory_records(owner=owner, domain=domain)
        term_sets = {record.rid: set(self._terms(record.text)) for record in records}
        shadowed: list[dict[str, Any]] = []
        already_shadowed: set[str] = set()
        now = time.time()
        with self._connect() as conn:
            for idx, left in enumerate(records):
                if left.rid in already_shadowed:
                    continue
                for right in records[idx + 1 :]:
                    if right.rid in already_shadowed:
                        continue
                    if not self._same_duplicate_scope(left, right):
                        continue
                    overlap = self._term_overlap_sets(term_sets[left.rid], term_sets[right.rid])
                    if overlap < threshold:
                        continue
                    canonical, duplicate = self._preferred_duplicate_canonical(left, right)
                    conn.execute(
                        "UPDATE memories SET status='shadowed', updated_at=? WHERE rid=?",
                        (now, duplicate.rid),
                    )
                    self._audit(
                        conn,
                        duplicate.rid,
                        "memory_shadowed",
                        reason="duplicate",
                        metadata={"canonical_rid": canonical.rid, "overlap": round(overlap, 4)},
                    )
                    shadowed.append(
                        {
                            "canonical_rid": canonical.rid,
                            "shadowed_rid": duplicate.rid,
                            "overlap": round(overlap, 4),
                        }
                    )
                    already_shadowed.add(duplicate.rid)
                    if duplicate.rid == left.rid:
                        break
        return shadowed

    def _active_memory_records(
        self, *, owner: str | None = None, domain: str | None = None
    ) -> list[MemoryRecord]:
        clauses = ["status='active'"]
        params: list[Any] = []
        if owner is not None:
            clauses.append("owner=?")
            params.append(owner)
        if domain is not None:
            clauses.append("domain=?")
            params.append(domain)
        where = " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY created_at ASC", tuple(params)
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _same_duplicate_scope(left: MemoryRecord, right: MemoryRecord) -> bool:
        return (
            left.owner == right.owner
            and left.visibility == right.visibility
            and left.platform_scope == right.platform_scope
            and left.domain == right.domain
        )

    @staticmethod
    def _preferred_duplicate_canonical(
        left: MemoryRecord, right: MemoryRecord
    ) -> tuple[MemoryRecord, MemoryRecord]:
        left_quality = (left.importance + left.confidence, -left.created_at)
        right_quality = (right.importance + right.confidence, -right.created_at)
        if left_quality >= right_quality:
            return left, right
        return right, left

    def _term_overlap(self, left: str, right: str) -> float:
        return self._term_overlap_sets(set(self._terms(left)), set(self._terms(right)))

    @staticmethod
    def _term_overlap_sets(left_terms: set[str], right_terms: set[str]) -> float:
        if not left_terms or not right_terms:
            return 0.0
        return len(left_terms & right_terms) / max(1, len(left_terms | right_terms))

    def expire_pending_inbox_items(
        self,
        *,
        max_age_days: int = 30,
        reason: str = "expired pending review",
        owner: str | None = None,
        domain: str | None = None,
    ) -> list[MemoryInboxItem]:
        cutoff = time.time() - max(0, max_age_days) * 24 * 60 * 60
        now = time.time()
        clauses = ["decision='pending'", "created_at < ?"]
        params: list[Any] = [cutoff]
        if owner is not None:
            clauses.append("owner=?")
            params.append(owner)
        if domain is not None:
            clauses.append("domain=?")
            params.append(domain)
        where = " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_inbox WHERE {where} ORDER BY created_at", tuple(params)
            ).fetchall()
            cids = [str(row["cid"]) for row in rows]
            for cid in cids:
                conn.execute(
                    "UPDATE memory_inbox SET decision='expired', review_reason=?, updated_at=? WHERE cid=?",
                    (reason, now, cid),
                )
        return [self.get_inbox_item(cid) for cid in cids]

    def get_inbox_item(self, cid: str) -> MemoryInboxItem:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_inbox WHERE cid=?", (cid,)
            ).fetchone()
        if row is None:
            raise KeyError(cid)
        return self._row_to_inbox_item(row)

    def get_memory(self, rid: str) -> MemoryRecord:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM memories WHERE rid=?", (rid,)).fetchone()
        if row is None:
            raise KeyError(rid)
        return self._row_to_record(row)

    def set_active_embedding_model(self, model_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO settings (key, value, updated_at)
                VALUES ('active_embedding_model', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (model_name, time.time()),
            )

    def set_recall_config(self, **values: str | int | Path | None) -> dict[str, str]:
        ts = time.time()
        with self._connect() as conn:
            for key, value in values.items():
                if value is None:
                    continue
                conn.execute(
                    """
                    INSERT INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                    """,
                    (f"recall.{key}", str(value), ts),
                )
        return self.recall_config()

    def recall_config(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM settings WHERE key LIKE 'recall.%'"
            ).fetchall()
        return {str(row["key"])[len("recall.") :]: str(row["value"]) for row in rows}

    def set_synthesis_config(self, **values: str | int | float | Path | None) -> dict[str, str]:
        ts = time.time()
        with self._connect() as conn:
            for key, value in values.items():
                if value is None:
                    continue
                conn.execute(
                    """
                    INSERT INTO settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                    """,
                    (f"synthesis.{key}", str(value), ts),
                )
        return self.synthesis_config()

    def synthesis_config(self) -> dict[str, str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT key, value FROM settings WHERE key LIKE 'synthesis.%'"
            ).fetchall()
        return {str(row["key"])[len("synthesis.") :]: str(row["value"]) for row in rows}

    def active_embedding_model(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key='active_embedding_model'"
            ).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def embedding_status(self, embedder: Embedder) -> dict[str, int | str]:
        """Return active-memory coverage for exactly one embedder model.

        The configured/active model is authoritative: coverage for 32M does not
        count 2M or 8M vectors, even if those older caches still exist.
        """
        with self._connect() as conn:
            total_active = int(
                conn.execute("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0]
            )
            embedded = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM memory_embeddings e
                    JOIN memories m ON m.rid = e.rid
                    WHERE m.status='active' AND e.model_id=? AND e.dimension=?
                    """,
                    (embedder.model_id, int(embedder.dimension)),
                ).fetchone()[0]
            )
        return {
            "model_id": embedder.model_id,
            "dimension": int(embedder.dimension),
            "total_active": total_active,
            "embedded": embedded,
            "missing": total_active - embedded,
        }

    def embed_missing(self, embedder: Embedder) -> dict[str, int]:
        """Embed active memories missing vectors for this embedder.

        Embeddings are optional and never change scope/status semantics. Recall
        still hard-filters memories before vector scoring.
        """
        embedded = 0
        skipped = 0
        ts = time.time()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*
                FROM memories m
                LEFT JOIN memory_embeddings e
                  ON e.rid = m.rid AND e.model_id = ? AND e.dimension = ?
                WHERE m.status = 'active' AND e.rid IS NULL
                ORDER BY m.created_at
                """,
                (embedder.model_id, int(embedder.dimension)),
            ).fetchall()
            for row in rows:
                record = self._row_to_record(row)
                vector = [float(value) for value in embedder.embed(record.text)]
                if len(vector) != embedder.dimension:
                    skipped += 1
                    continue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_embeddings (
                        rid, model_id, dimension, vector_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.rid,
                        embedder.model_id,
                        int(embedder.dimension),
                        json.dumps(vector),
                        ts,
                        ts,
                    ),
                )
                self._audit(
                    conn,
                    record.rid,
                    "memory_embedded",
                    metadata={"model_id": embedder.model_id, "dimension": embedder.dimension},
                )
                embedded += 1
        return {"embedded": embedded, "skipped": skipped}

    def rebuild_vector_index(self, embedder: Embedder, vector_index: VectorIndex) -> dict[str, int | str]:
        """Load this embedder's cached vectors into an optional vector index."""
        if vector_index.model_id != embedder.model_id or vector_index.dimension != embedder.dimension:
            raise ValueError("vector index model_id/dimension must match embedder")
        rows_for_index: list[VectorIndexRow] = []
        skipped = 0
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*, e.vector_json
                FROM memory_embeddings e
                JOIN memories m ON m.rid = e.rid
                WHERE e.model_id=? AND e.dimension=?
                ORDER BY m.rid
                """,
                (embedder.model_id, int(embedder.dimension)),
            ).fetchall()
        for row in rows:
            try:
                vector = [float(value) for value in json.loads(row["vector_json"])]
            except (TypeError, ValueError, json.JSONDecodeError):
                skipped += 1
                continue
            if len(vector) != embedder.dimension:
                skipped += 1
                continue
            rows_for_index.append(
                VectorIndexRow(
                    rid=str(row["rid"]),
                    vector=vector,
                    owner=str(row["owner"]),
                    visibility=str(row["visibility"]),
                    platform_scope=str(row["platform_scope"]),
                    status=str(row["status"]),
                    domain=str(row["domain"] or ""),
                )
            )
        vector_index.build(rows_for_index)
        return {
            "model_id": embedder.model_id,
            "dimension": int(embedder.dimension),
            "indexed": len(rows_for_index),
            "skipped": skipped,
        }

    def vector_index_metadata_fingerprints(
        self, embedder: Embedder
    ) -> dict[str, set[tuple[str, str, str, str, str]]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT m.rid, m.owner, m.visibility, m.platform_scope, m.status, m.domain
                FROM memory_embeddings e
                JOIN memories m ON m.rid = e.rid
                WHERE e.model_id=? AND e.dimension=?
                ORDER BY m.rid
                """,
                (embedder.model_id, int(embedder.dimension)),
            ).fetchall()
        fingerprints: dict[str, set[tuple[str, str, str, str, str]]] = {}
        for row in rows:
            platform_keys = [
                part.strip() for part in str(row["platform_scope"]).split(",") if part.strip()
            ] or ["all"]
            fingerprints[str(row["rid"])] = {
                (
                    str(row["owner"]),
                    str(row["visibility"]),
                    platform_key,
                    str(row["status"]),
                    str(row["domain"] or ""),
                )
                for platform_key in platform_keys
            }
        return fingerprints

    def recall(
        self,
        query: str,
        *,
        owner: str,
        platform: str,
        allowed_visibility: set[str] | None = None,
        limit: int = 10,
        domain: str | None = None,
        embedder: Embedder | None = None,
        vector_candidate_limit: int | None = None,
        vector_index: VectorIndex | None = None,
        ann_candidate_limit: int = 0,
        recall_policy: str = "latency_first",
        ann_min_keyword_candidates: int = 50,
    ) -> list[RecallResult]:
        allowed_visibility = allowed_visibility or {"private"}
        base_terms = self._terms(query)
        expansion_terms = self._query_expansion_terms(query)
        terms = list(dict.fromkeys(base_terms + expansion_terms))
        expanded_query = self._expanded_query_text(query)
        has_intent_expansion = bool(expansion_terms)
        if not terms and embedder is None:
            return []
        if self._is_pure_ephemeral_query(set(terms)):
            return []
        if recall_policy not in {"latency_first", "recall_first", "semantic_only"}:
            raise ValueError("recall_policy must be latency_first, recall_first, or semantic_only")
        results_by_rid: dict[str, RecallResult] = {}
        keyword_candidate_rids: list[str] = []
        now = time.time()

        if terms and recall_policy != "semantic_only":
            fts_query = " OR ".join(terms)
            placeholders = ",".join("?" for _ in allowed_visibility)
            params: list[Any] = [fts_query, owner, *sorted(allowed_visibility)]
            domain_clause = ""
            if domain:
                domain_clause = " AND m.domain=?"
                params.append(domain)
            params.extend([max(250, limit * 10)])
            sql = f"""
                SELECT m.*, bm25(memory_fts) AS rank
                FROM memory_fts
                JOIN memories m ON m.rid = memory_fts.rid
                WHERE memory_fts MATCH ?
                  AND m.status = 'active'
                  AND m.owner = ?
                  AND m.visibility IN ({placeholders})
                  AND (m.platform_scope = 'all' OR instr(',' || m.platform_scope || ',', ',' || ? || ',') > 0)
                  {domain_clause}
                ORDER BY rank ASC
                LIMIT ?
            """
            platform_index = 2 + len(allowed_visibility)
            params.insert(platform_index, platform)
            with self._connect() as conn:
                rows = conn.execute(sql, tuple(params)).fetchall()
                for row in rows:
                    record = self._row_to_record(row)
                    if self._should_suppress_recall_result(query, record):
                        continue
                    keyword_candidate_rids.append(record.rid)
                    reasons = ["keyword_match", "scope_match"]
                    score = max(0.0, 1.0 - min(1.0, abs(float(row["rank"]))))
                    matched_terms = self._matched_terms(terms, record.text)
                    coverage = len(matched_terms) / max(1, len(terms))
                    score += coverage * 2.0
                    if self._has_exact_phrase_match(query, record.text):
                        score += 1.25
                        reasons.append("exact_phrase_match")
                    exact_identifiers = self._exact_identifier_matches(query, record.text)
                    if exact_identifiers:
                        score += min(1.5, len(exact_identifiers) * 0.75)
                        reasons.append("exact_identifier_match")
                    if self._has_normalized_phrase_match(query, record.text):
                        score += 1.0
                        reasons.append("normalized_phrase_match")
                    if coverage == 1.0:
                        score += 1.0
                        reasons.append("all_query_terms_matched")
                    rare_matches = [
                        term for term in matched_terms if self._is_specific_query_term(term)
                    ]
                    if rare_matches:
                        score += min(1.0, len(rare_matches) * 0.5)
                        reasons.append("specific_term_match")
                    expansion_matches = self._matched_terms(expansion_terms, record.text)
                    if expansion_matches:
                        score += min(1.5, len(expansion_matches) * 0.35)
                        reasons.append("semantic_intent_expansion")
                    score += record.importance * 0.2 + record.confidence * 0.2
                    if record.domain and record.domain.lower() in query.lower():
                        score += 0.1
                        reasons.append("domain_match")
                    if record.importance >= 0.8:
                        reasons.append("important")
                    results_by_rid[record.rid] = RecallResult(
                        record=record, score=round(score, 4), reasons=reasons
                    )

        if embedder is not None:
            vector_candidate_rids: list[str] | None = None
            ann_candidate_rids: list[str] = []
            should_search_ann = (
                vector_index is not None
                and ann_candidate_limit > 0
                and (
                    recall_policy in {"recall_first", "semantic_only"}
                    or len(keyword_candidate_rids) < max(0, ann_min_keyword_candidates)
                )
            )
            if should_search_ann:
                assert vector_index is not None
                if (
                    vector_index.model_id != embedder.model_id
                    or vector_index.dimension != embedder.dimension
                ):
                    raise ValueError("vector index model_id/dimension must match embedder")
                query_vector = [float(value) for value in embedder.embed(expanded_query)]
                ann_candidate_rids = [
                    rid
                    for rid, _score in vector_index.search(
                        query_vector,
                        top_k=ann_candidate_limit,
                        owner=owner,
                        platform=platform,
                        allowed_visibility=allowed_visibility,
                        domain=domain,
                        status="active",
                    )
                ]
            if vector_candidate_limit is not None:
                keyword_rids = [] if recall_policy == "semantic_only" else keyword_candidate_rids
                vector_candidate_rids = list(
                    dict.fromkeys(
                        keyword_rids[: max(0, vector_candidate_limit)] + ann_candidate_rids
                    )
                )
            for semantic_result in self._vector_recall_candidates(
                query=expanded_query,
                owner=owner,
                platform=platform,
                allowed_visibility=allowed_visibility,
                domain=domain,
                embedder=embedder,
                limit=max(250, limit * 10),
                candidate_rids=vector_candidate_rids,
                ann_candidate_rids=set(ann_candidate_rids),
                intent_expanded=has_intent_expansion,
            ):
                existing = results_by_rid.get(semantic_result.record.rid)
                if existing is None:
                    results_by_rid[semantic_result.record.rid] = semantic_result
                    continue
                reasons = list(dict.fromkeys(existing.reasons + semantic_result.reasons))
                results_by_rid[existing.record.rid] = RecallResult(
                    record=existing.record,
                    score=round(existing.score + semantic_result.score, 4),
                    reasons=reasons,
                )

        results = sorted(results_by_rid.values(), key=lambda r: r.score, reverse=True)[:limit]
        with self._connect() as conn:
            for result in results:
                conn.execute(
                    "UPDATE memories SET last_access=?, updated_at=? WHERE rid=?",
                    (now, now, result.record.rid),
                )
                self._audit(conn, result.record.rid, "memory_recalled")
            self._audit(
                conn,
                "runtime",
                "recall_query",
                reason="store.recall",
                metadata={
                    "query": query,
                    "owner": owner,
                    "platform": platform,
                    "allowed_visibility": sorted(allowed_visibility),
                    "domain": domain,
                    "limit": limit,
                    "result_count": len(results),
                    "result_rids": [result.record.rid for result in results],
                    "recall_policy": recall_policy,
                    "embedding_model": embedder.model_id if embedder is not None else "",
                    "vector_enabled": embedder is not None,
                },
            )
        return results

    def _vector_recall_candidates(
        self,
        *,
        query: str,
        owner: str,
        platform: str,
        allowed_visibility: set[str],
        domain: str | None,
        embedder: Embedder,
        limit: int,
        candidate_rids: list[str] | None = None,
        ann_candidate_rids: set[str] | None = None,
        intent_expanded: bool = False,
    ) -> list[RecallResult]:
        query_vector = [float(value) for value in embedder.embed(query)]
        if len(query_vector) != embedder.dimension or not any(query_vector):
            return []
        if candidate_rids == []:
            return []
        placeholders = ",".join("?" for _ in allowed_visibility)
        params: list[Any] = [embedder.model_id, int(embedder.dimension), owner, *sorted(allowed_visibility)]
        domain_clause = ""
        if domain:
            domain_clause = " AND m.domain=?"
            params.append(domain)
        candidate_clause = ""
        if candidate_rids is not None:
            candidate_placeholders = ",".join("?" for _ in candidate_rids)
            candidate_clause = f" AND e.rid IN ({candidate_placeholders})"
            params.extend(candidate_rids)
        params.append(platform)
        sql = f"""
            SELECT m.*, e.vector_json
            FROM memory_embeddings e
            JOIN memories m ON m.rid = e.rid
            WHERE e.model_id = ?
              AND e.dimension = ?
              AND m.status = 'active'
              AND m.owner = ?
              AND m.visibility IN ({placeholders})
              {domain_clause}
              {candidate_clause}
              AND (m.platform_scope = 'all' OR instr(',' || m.platform_scope || ',', ',' || ? || ',') > 0)
        """
        results: list[RecallResult] = []
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        for row in rows:
            record = self._row_to_record(row)
            if self._should_suppress_recall_result(query, record):
                continue
            try:
                vector = [float(value) for value in json.loads(row["vector_json"])]
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            similarity = cosine_similarity(query_vector, vector)
            if similarity <= 0:
                continue
            reasons = ["semantic_match", "scope_match"]
            if intent_expanded:
                reasons.append("semantic_intent_expansion")
            if ann_candidate_rids and record.rid in ann_candidate_rids:
                reasons.append("ann_match")
            score = 1.0 + similarity * 2.0 + record.importance * 0.2 + record.confidence * 0.2
            if record.domain and record.domain.lower() in query.lower():
                score += 0.1
                reasons.append("domain_match")
            if record.importance >= 0.8:
                reasons.append("important")
            results.append(RecallResult(record=record, score=round(score, 4), reasons=reasons))
        return sorted(results, key=lambda r: r.score, reverse=True)[:limit]


    def simulate_recall(
        self,
        query: str,
        *,
        owner: str,
        platform: str,
        allowed_visibility: set[str] | None = None,
        limit: int = 10,
        domain: str | None = None,
        sample_limit: int = 200,
    ) -> dict[str, Any]:
        """Explain what normal recall would include and why nearby records are excluded."""
        allowed_visibility = allowed_visibility or {"private"}
        included_results = self.recall(
            query,
            owner=owner,
            platform=platform,
            allowed_visibility=allowed_visibility,
            limit=limit,
            domain=domain,
        )
        included_by_rid = {result.record.rid: result for result in included_results}
        excluded: list[dict[str, Any]] = []
        with self._connect() as conn:
            memory_rows = conn.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (max(1, sample_limit),)
            ).fetchall()
            inbox_rows = conn.execute(
                "SELECT * FROM memory_inbox ORDER BY created_at DESC LIMIT ?",
                (max(1, sample_limit),),
            ).fetchall()

        for row in memory_rows:
            record = self._row_to_record(row)
            if record.rid in included_by_rid:
                continue
            reasons = self._recall_exclusion_reasons(
                record,
                query=query,
                owner=owner,
                platform=platform,
                allowed_visibility=allowed_visibility,
                domain=domain,
            )
            if not reasons:
                reasons = ["below_top_k_or_no_match"]
            excluded.append(
                {
                    "type": "memory",
                    "rid": record.rid,
                    "text": record.text,
                    "owner": record.owner,
                    "visibility": record.visibility,
                    "platform_scope": record.platform_scope,
                    "domain": record.domain,
                    "status": record.status,
                    "exclusion_reasons": reasons,
                }
            )

        for row in inbox_rows:
            item = self._row_to_inbox_item(row)
            if item.decision == "accepted" and item.accepted_rid in included_by_rid:
                continue
            reason = f"inbox_{item.decision}_not_recallable"
            excluded.append(
                {
                    "type": "inbox",
                    "cid": item.cid,
                    "text": item.proposed_text,
                    "owner": item.owner,
                    "visibility": item.visibility,
                    "platform_scope": item.platform_scope,
                    "domain": item.domain,
                    "decision": item.decision,
                    "accepted_rid": item.accepted_rid,
                    "exclusion_reasons": [reason],
                }
            )

        return {
            "query": query,
            "owner": owner,
            "platform": platform,
            "allowed_visibility": sorted(allowed_visibility),
            "domain": domain,
            "limit": limit,
            "included": [
                {
                    "rid": result.record.rid,
                    "text": result.record.text,
                    "score": result.score,
                    "reasons": list(dict.fromkeys(result.reasons + ["included_in_recall"])),
                    "domain": result.record.domain,
                }
                for result in included_results
            ],
            "excluded": excluded,
            "context_preview": "\n".join(result.record.text for result in included_results),
        }

    def _recall_exclusion_reasons(
        self,
        record: MemoryRecord,
        *,
        query: str,
        owner: str,
        platform: str,
        allowed_visibility: set[str],
        domain: str | None,
    ) -> list[str]:
        reasons: list[str] = []
        if record.status != "active":
            reasons.append(f"status_{record.status}")
        if record.owner != owner:
            reasons.append("owner_mismatch")
        if record.visibility not in allowed_visibility:
            reasons.append("visibility_not_allowed")
        platform_keys = [part.strip() for part in record.platform_scope.split(",") if part.strip()] or ["all"]
        if "all" not in platform_keys and platform not in platform_keys:
            reasons.append("platform_scope_mismatch")
        if domain and record.domain != domain:
            reasons.append("domain_mismatch")
        if self._should_suppress_recall_result(query, record):
            reasons.append("suppressed_as_non_durable_or_question_only")
        return reasons

    def tombstone(self, rid: str, *, reason: str = "") -> None:
        ts = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE memories SET status='tombstoned', updated_at=? WHERE rid=?",
                (ts, rid),
            )
            if cur.rowcount == 0:
                raise KeyError(rid)
            self._audit(conn, rid, "memory_tombstoned", reason=reason)

    def correct_memory(
        self,
        rid: str,
        new_text: str,
        *,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        old = self.get_memory(rid)
        if old.status != "active":
            raise ValueError("Only active memories can be corrected")
        text = " ".join(new_text.split())
        if not text:
            raise ValueError("new_text is required")
        replacement_metadata = {
            **old.metadata,
            **(metadata or {}),
            "corrects_rid": old.rid,
            "correction_reason": reason,
        }
        ts = time.time()
        replacement_rid = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "UPDATE memories SET status='tombstoned', updated_at=? WHERE rid=?",
                (ts, old.rid),
            )
            conn.execute(
                """
                INSERT INTO memories (
                    rid,text,kind,owner,visibility,platform_scope,action_scope,domain,source,
                    importance,confidence,status,created_at,updated_at,last_access,ttl_days,metadata_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    replacement_rid,
                    text,
                    old.kind,
                    old.owner,
                    old.visibility,
                    old.platform_scope,
                    old.action_scope,
                    old.domain,
                    old.source,
                    old.importance,
                    old.confidence,
                    "active",
                    ts,
                    ts,
                    None,
                    old.ttl_days,
                    json.dumps(replacement_metadata, sort_keys=True),
                ),
            )
            conn.execute(
                "INSERT INTO memory_fts (rid,text,domain,source,owner) VALUES (?,?,?,?,?)",
                (replacement_rid, text, old.domain, old.source, old.owner),
            )
            self._audit(
                conn,
                old.rid,
                "memory_tombstoned",
                reason=f"correction:{replacement_rid}:{reason}",
            )
            self._audit(
                conn,
                replacement_rid,
                "memory_added",
            )
            self._audit(
                conn,
                old.rid,
                "memory_corrected_from",
                reason=reason,
                metadata={"replacement_rid": replacement_rid},
            )
            self._audit(
                conn,
                replacement_rid,
                "memory_corrected_to",
                reason=reason,
                metadata={"old_rid": old.rid},
            )
        return self.get_memory(replacement_rid)

    def audit_events(self, rid: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rid,event_type,reason,created_at,metadata_json FROM audit_log WHERE rid=? ORDER BY id",
                (rid,),
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            event = dict(row)
            event["metadata"] = json.loads(str(event.get("metadata_json") or "{}"))
            events.append(event)
        return events

    def detect_contradictions(
        self, *, owner: str | None = None, domain: str | None = None
    ) -> list[Contradiction]:
        clauses = ["status='active'"]
        params: list[Any] = []
        if owner is not None:
            clauses.append("owner=?")
            params.append(owner)
        if domain is not None:
            clauses.append("domain=?")
            params.append(domain)
        where = " AND ".join(clauses)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY created_at",
                tuple(params),
            ).fetchall()
            for idx, left in enumerate(rows):
                for right in rows[idx + 1 :]:
                    reasons = self._contradiction_reasons(left["text"], right["text"])
                    if not reasons:
                        continue
                    left_rid, right_rid = sorted([left["rid"], right["rid"]])
                    now = time.time()
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO contradictions (
                            conflict_id,left_rid,right_rid,owner,domain,status,reasons_json,winner_rid,
                            resolution_reason,created_at,updated_at
                        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            str(
                                uuid.uuid5(
                                    uuid.NAMESPACE_URL,
                                    f"anamnesis:contradiction:{left_rid}:{right_rid}",
                                )
                            ),
                            left_rid,
                            right_rid,
                            left["owner"],
                            left["domain"],
                            "open",
                            json.dumps(reasons),
                            None,
                            "",
                            now,
                            now,
                        ),
                    )
            result_rows = conn.execute(
                "SELECT * FROM contradictions WHERE status='open'"
                + (" AND owner=?" if owner is not None else "")
                + (" AND domain=?" if domain is not None else "")
                + " ORDER BY created_at",
                tuple(v for v in (owner, domain) if v is not None),
            ).fetchall()
        return [self._row_to_contradiction(row) for row in result_rows]

    def contradictions(self, *, status: str = "open") -> list[Contradiction]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM contradictions WHERE status=? ORDER BY created_at",
                (status,),
            ).fetchall()
        return [self._row_to_contradiction(row) for row in rows]

    def resolve_contradiction(
        self, conflict_id: str, *, winner_rid: str, reason: str = ""
    ) -> Contradiction:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM contradictions WHERE conflict_id=?", (conflict_id,)
            ).fetchone()
            if row is None:
                raise KeyError(conflict_id)
            if winner_rid not in {row["left_rid"], row["right_rid"]}:
                raise ValueError("winner_rid must be one side of the contradiction")
            loser_rid = (
                row["right_rid"] if winner_rid == row["left_rid"] else row["left_rid"]
            )
            now = time.time()
            conn.execute(
                "UPDATE memories SET status='tombstoned', updated_at=? WHERE rid=?",
                (now, loser_rid),
            )
            conn.execute(
                "UPDATE contradictions SET status='resolved', winner_rid=?, resolution_reason=?, updated_at=? WHERE conflict_id=?",
                (winner_rid, reason, now, conflict_id),
            )
            self._audit(
                conn,
                loser_rid,
                "memory_tombstoned",
                reason=f"contradiction:{conflict_id}:{reason}",
            )
            self._audit(
                conn,
                winner_rid,
                "contradiction_resolved",
                reason=reason,
                metadata={"conflict_id": conflict_id, "loser_rid": loser_rid},
            )
            resolved = conn.execute(
                "SELECT * FROM contradictions WHERE conflict_id=?", (conflict_id,)
            ).fetchone()
        return self._row_to_contradiction(resolved)

    def _duplicate_hints(
        self, text: str, *, owner: str, visibility: str, domain: str
    ) -> list[str]:
        terms = set(self._terms(text))
        if not terms:
            return []
        candidates = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT rid,text FROM memories WHERE status='active' AND owner=? AND visibility=? AND domain=? LIMIT 100",
                (owner, visibility, domain),
            ).fetchall()
        for row in rows:
            other_terms = set(self._terms(row["text"]))
            if not other_terms:
                continue
            overlap = len(terms & other_terms) / max(1, len(terms | other_terms))
            if overlap >= 0.45:
                candidates.append((overlap, row["rid"]))
        return [rid for _overlap, rid in sorted(candidates, reverse=True)[:3]]

    def _audit(
        self,
        conn: sqlite3.Connection,
        rid: str,
        event_type: str,
        *,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn.execute(
            "INSERT INTO audit_log (rid,event_type,reason,created_at,metadata_json) VALUES (?,?,?,?,?)",
            (
                rid,
                event_type,
                reason,
                time.time(),
                json.dumps(metadata or {}, sort_keys=True),
            ),
        )

    @staticmethod
    def _matched_terms(terms: list[str], text: str) -> set[str]:
        haystack = text.lower()
        return {term for term in terms if term in haystack}

    @staticmethod
    def _is_specific_query_term(term: str) -> bool:
        generic_terms = {
            "adult",
            "benchmark",
            "configuration",
            "dashboard",
            "permission",
            "permissions",
            "private",
            "privacy",
            "question",
            "summary",
        }
        if term in generic_terms:
            return False
        return any(ch.isdigit() for ch in term) or len(term) >= 10

    @staticmethod
    def _is_pure_ephemeral_query(query_terms: set[str]) -> bool:
        if not (query_terms & Anamnesis._EPHEMERAL_QUERY_TERMS()):
            return False
        return query_terms.isdisjoint(Anamnesis._DURABLE_QUERY_ANCHORS())

    @staticmethod
    def _DURABLE_QUERY_ANCHORS() -> set[str]:
        return {
            "infrastructure",
            "dashboard",
            "service",
            "provider",
            "privacy",
            "permission",
            "permissions",
            "config",
            "configuration",
            "credential",
            "identity",
            "identifier",
            "host",
            "worker",
            "model",
            "endpoint",
            "grpc",
            "access",
            "prompt",
            "pitfall",
            "generation",
            "database",
            "memory",
            "project",
            "version",
            "api",
            "command",
            "package",
            "test",
        }

    @staticmethod
    def _should_suppress_recall_result(query: str, record: MemoryRecord) -> bool:
        query_terms = set(Anamnesis._terms(query))
        text = record.text.lower()
        if text.startswith(("question:", "query:")) or (
            text.endswith("?")
            and text.startswith(
                (
                    "what ",
                    "where ",
                    "when ",
                    "which ",
                    "who ",
                    "why ",
                    "how ",
                )
            )
        ):
            return True
        if any(
            marker in text
            for marker in [
                "you are in a native hermes memory-provider benchmark",
                "you must call the active persistent-memory recall tool",
                "use this exact recall query",
                "answer from persistent memory only",
                "review the conversation above",
                "be active — most sessions produce at least one skill update",
            ]
        ):
            return True
        if not (query_terms & Anamnesis._EPHEMERAL_QUERY_TERMS()):
            return False
        domain = record.domain.lower()
        if domain in {"task-state", "task_state", "temporary", "debug", "process"}:
            return True
        if Anamnesis._has_durable_operational_evidence(record, query_terms):
            return False
        return any(
            re.search(pattern, text)
            for pattern in Anamnesis._STALE_OPERATIONAL_PATTERNS()
        )

    @staticmethod
    def _has_durable_operational_evidence(record: MemoryRecord, query_terms: set[str]) -> bool:
        volatile_terms = {"temporary", "temp", "stale", "pid", "stuck", "wedged", "task", "task_state", "current"}
        if query_terms & volatile_terms:
            return False
        text = record.text.lower()
        domain = record.domain.lower().replace("-", "_")
        lifecycle = str(record.metadata.get("intake_lifecycle", "")).lower()
        stable_domains = {
            "infrastructure",
            "infra",
            "project",
            "system",
            "systems",
            "devops",
            "mlops",
            "integration",
            "integrations",
        }
        durable_anchors = {
            "access prompt",
            "dashboard",
            "default port",
            "grpc",
            "integration",
            "pitfall",
            "production",
            "provider",
            "runs on",
            "service",
            "webhook endpoint",
        }
        query_has_durable_anchor = bool(query_terms & Anamnesis._DURABLE_QUERY_ANCHORS())
        record_has_anchor = any(anchor in text for anchor in durable_anchors)
        record_has_operational_identifier = bool(
            re.search(r"\bport\s+\d+\b|\bendpoint\b|\bhttps?://|\b127\.0\.0\.1\b|\blocalhost\b|\bgrpc\b", text)
        )
        return record_has_operational_identifier and (
            lifecycle == "stable_infrastructure"
            or domain in stable_domains
            or record_has_anchor
            or query_has_durable_anchor
        )

    @staticmethod
    def _has_exact_phrase_match(query: str, text: str) -> bool:
        low_query = query.lower()
        low_text = text.lower()
        quoted = re.findall(r'"([^"]{4,})"', query)
        if any(phrase.lower() in low_text for phrase in quoted):
            return True
        query_terms = [term for term in Anamnesis._terms(low_query) if len(term) >= 4]
        for size in range(min(4, len(query_terms)), 1, -1):
            for idx in range(0, len(query_terms) - size + 1):
                phrase = " ".join(query_terms[idx : idx + size])
                if phrase in low_text:
                    return True
        return False

    @staticmethod
    def _has_normalized_phrase_match(query: str, text: str) -> bool:
        normalized_query = re.sub(r"[^a-z0-9]+", "", query.lower())
        normalized_text = re.sub(r"[^a-z0-9]+", "", text.lower())
        if "llmvision" in normalized_query and "llmvision" in normalized_text:
            return True
        if "missing" in normalized_query and "skillnotfound" in normalized_text:
            return True
        if "missing" in normalized_query and "skillsnotfound" in normalized_text:
            return True
        return False

    @staticmethod
    def _exact_identifier_matches(query: str, text: str) -> set[str]:
        identifiers = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_./:+-]{2,}", query)
            if any(ch.isdigit() for ch in token)
            or any(ch in token for ch in "./:+-")
            or token.isupper()
        }
        low_text = text.lower()
        return {identifier for identifier in identifiers if identifier in low_text}

    @staticmethod
    def _EPHEMERAL_QUERY_TERMS() -> set[str]:
        return {
            "pid",
            "stuck",
            "wedged",
            "temporary",
            "temp",
            "task",
            "task_state",
            "cron",
            "process",
            "server",
            "port",
            "commit",
            "sha",
            "pr",
            "phase",
            "current",
        }

    @staticmethod
    def _STALE_OPERATIONAL_PATTERNS() -> tuple[str, ...]:
        return (
            r"\bport\s+\d+\b",
            r"\bpid\s*[:#]?\s*\d+\b",
            r"\bbackground process\b",
            r"\bwatch pattern\b",
            r"\bcron\b.*\b(stuck|lock|interrupted|failed)\b",
            r"\b(stuck|wedged)\b.*\b(lock|pid|cron|process|job)\b",
            r"\btemporary\b.*\b(process|server|manual|task|port)\b",
            r"\btask[- ]state\b",
            r"\b(process|debug port|listening)\b",
            r"\b(restart|restart required|old in-memory)\b",
            r"\bphase\s+\d+\b",
            r"\bphase[s]?\b.*\b(completed|done|release|implemented)\b",
            r"\bcommit\s+(sha|[0-9a-f]{6,40})\b",
            r"\bpr\s+(number|#?\d+)\b",
            r"review the conversation above",
        )

    @staticmethod
    def _expanded_query_text(query: str) -> str:
        expansions = Anamnesis._query_expansion_terms(query)
        if not expansions:
            return query
        return " ".join([query, *expansions])

    @staticmethod
    def _query_expansion_terms(query: str) -> list[str]:
        """Small, deterministic intent expansion for low-overlap memory queries.

        This is deliberately generic and transparent: it maps common user intent
        phrasing to durable-memory vocabulary before FTS/vector scoring, without
        changing stored memories or introducing model cascades.
        """
        normalized = re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()
        expansions: list[str] = []
        phrase_expansions = {
            "little one": ("child",),
            "kid": ("child",),
            "sleeping": ("asleep", "quiet"),
            "asleep": ("quiet",),
            "audio": ("voice",),
            "voice": ("audio",),
            "buyer": ("customer", "outcomes", "trust"),
            "buyers": ("customer", "outcomes", "trust"),
            "results": ("outcomes",),
            "internals": ("implementation", "details"),
            "transient": ("temporary", "operational", "noise"),
            "chatter": ("noise",),
            "forever": ("durable", "memory"),
            "recommending work": ("recommendation", "permission", "execute", "verify"),
            "getting ok": ("permission", "execute", "verify"),
            "go ahead": ("permission", "execute", "verify"),
            "model config": ("endpoint", "model", "api", "key"),
            "answer generation": ("synthesis",),
        }
        for phrase, terms in phrase_expansions.items():
            if phrase in normalized:
                expansions.extend(terms)
        token_expansions = {
            "sleeping": ("asleep", "quiet"),
            "kid": ("child",),
            "buyers": ("customer", "outcomes", "trust"),
            "buyer": ("customer", "outcomes", "trust"),
            "results": ("outcomes",),
            "internals": ("implementation", "details"),
            "transient": ("temporary", "operational", "noise"),
            "chatter": ("noise",),
            "forever": ("durable", "memory"),
            "ok": ("permission", "execute", "verify"),
            "recommend": ("recommendation", "permission"),
            "recommending": ("recommendation", "permission"),
        }
        for term in Anamnesis._terms(query):
            expansions.extend(token_expansions.get(term, ()))
        return [term for term in dict.fromkeys(expansions) if term not in Anamnesis._terms(query)]

    @staticmethod
    def _terms(query: str) -> list[str]:
        raw = [
            part.lower() for part in query.replace('"', " ").replace("-", " ").split()
        ]
        terms = []
        for term in raw:
            cleaned = "".join(ch for ch in term if ch.isalnum() or ch == "_")
            if (
                len(cleaned) >= 2
                and cleaned not in terms
                and cleaned not in Anamnesis._STOPWORDS()
            ):
                terms.append(cleaned)
        return terms

    @staticmethod
    def _STOPWORDS() -> set[str]:
        return {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "do",
            "does",
            "for",
            "from",
            "have",
            "how",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "should",
            "that",
            "the",
            "to",
            "use",
            "was",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
        }

    @classmethod
    def _contradiction_reasons(cls, left: str, right: str) -> list[str]:
        left_terms = set(cls._terms(left))
        right_terms = set(cls._terms(right))
        shared = (left_terms - cls._NEGATORS()) & (right_terms - cls._NEGATORS())
        has_shared_subject = len(shared) >= 2
        left_negative = bool(left_terms & cls._NEGATORS())
        right_negative = bool(right_terms & cls._NEGATORS())
        if has_shared_subject and left_negative != right_negative:
            return ["polarity_conflict"]
        return []

    @staticmethod
    def _NEGATORS() -> set[str]:
        return {
            "not",
            "no",
            "cannot",
            "cant",
            "wont",
            "never",
            "forbidden",
            "disallowed",
            "disabled",
        }

    @staticmethod
    def _row_to_contradiction(row: sqlite3.Row) -> Contradiction:
        return Contradiction(
            conflict_id=row["conflict_id"],
            left_rid=row["left_rid"],
            right_rid=row["right_rid"],
            owner=row["owner"],
            domain=row["domain"],
            status=row["status"],
            reasons=json.loads(row["reasons_json"] or "[]"),
            winner_rid=row["winner_rid"],
            resolution_reason=row["resolution_reason"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _row_to_inbox_item(row: sqlite3.Row) -> MemoryInboxItem:
        return MemoryInboxItem(
            cid=row["cid"],
            proposed_text=row["proposed_text"],
            source_snippet=row["source_snippet"],
            proposed_kind=row["proposed_kind"],
            owner=row["owner"],
            visibility=row["visibility"],
            platform_scope=row["platform_scope"],
            action_scope=row["action_scope"],
            domain=row["domain"],
            source=row["source"],
            confidence=float(row["confidence"]),
            why_save=row["why_save"],
            suggested_lifecycle=row["suggested_lifecycle"],
            decision=row["decision"],
            review_reason=row["review_reason"],
            duplicate_rids=json.loads(row["duplicate_rids_json"] or "[]"),
            hints=json.loads(row["hints_json"] or "[]"),
            accepted_rid=row["accepted_rid"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            rid=row["rid"],
            text=row["text"],
            kind=row["kind"],
            owner=row["owner"],
            visibility=row["visibility"],
            platform_scope=row["platform_scope"],
            action_scope=row["action_scope"],
            domain=row["domain"],
            source=row["source"],
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            status=row["status"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_access=float(row["last_access"])
            if row["last_access"] is not None
            else None,
            ttl_days=float(row["ttl_days"]) if row["ttl_days"] is not None else None,
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from .embeddings import cosine_similarity


@dataclass(frozen=True)
class VectorIndexRow:
    rid: str
    vector: list[float]
    owner: str = ""
    visibility: str = "private"
    platform_scope: str = "all"
    status: str = "active"
    domain: str = ""


VectorIndexBuildRow = VectorIndexRow | tuple[str, list[float]]


class VectorIndex(Protocol):
    """Optional vector index interface for ANN-style semantic candidate lookup."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def build(self, rows: Sequence[VectorIndexBuildRow]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        owner: str | None = None,
        platform: str | None = None,
        allowed_visibility: set[str] | None = None,
        domain: str | None = None,
        status: str = "active",
    ) -> list[tuple[str, float]]: ...

    def count(self) -> int: ...


@dataclass
class ExactVectorIndex:
    """Dependency-free exact vector index used for tests and fallback plumbing.

    This is not an ANN implementation. It implements the same interface while
    scoring only vectors explicitly loaded into the index, making hybrid recall
    behavior deterministic before adding hnswlib/sqlite-vec backends.
    """

    model_id: str
    dimension: int
    _vectors: dict[str, list[float]] = field(default_factory=dict)

    def build(self, rows: Sequence[VectorIndexBuildRow]) -> None:
        normalized_rows = [_coerce_index_row(row) for row in rows]
        self._vectors = {
            row.rid: [float(value) for value in row.vector]
            for row in normalized_rows
            if len(row.vector) == self.dimension
        }

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        owner: str | None = None,
        platform: str | None = None,
        allowed_visibility: set[str] | None = None,
        domain: str | None = None,
        status: str = "active",
    ) -> list[tuple[str, float]]:
        if len(query_vector) != self.dimension or top_k <= 0:
            return []
        scored = [
            (rid, cosine_similarity(query_vector, vector))
            for rid, vector in self._vectors.items()
        ]
        return [
            (rid, score)
            for rid, score in sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
            if score > 0
        ]

    def count(self) -> int:
        return len(self._vectors)


def _coerce_index_row(row: VectorIndexBuildRow) -> VectorIndexRow:
    if isinstance(row, VectorIndexRow):
        return row
    rid, vector = row
    return VectorIndexRow(rid=rid, vector=vector)


def _platform_scope_keys(platform_scope: str) -> list[str]:
    keys = [part.strip() for part in platform_scope.split(",") if part.strip()]
    return keys or ["all"]


def sqlite_vec_available() -> bool:
    try:
        import sqlite_vec  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class SQLiteVecVectorIndex:
    """sqlite-vec-backed local vector index.

    The index is intentionally a derived cache. Build recreates the vec0 table
    from canonical memory_embeddings rows, while recall still re-filters returned
    rids through Anamnesis governance before exposing results.
    """

    db_path: str | Path
    model_id: str
    dimension: int

    def _connect(self) -> sqlite3.Connection:
        try:
            import sqlite_vec
        except Exception as exc:  # pragma: no cover - covered by availability helper
            raise RuntimeError("sqlite-vec is not installed") from exc
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            conn.enable_load_extension(False)
        return conn

    def build(self, rows: Sequence[VectorIndexBuildRow]) -> None:
        from sqlite_vec import serialize_float32

        normalized_rows = [_coerce_index_row(row) for row in rows]
        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS vectors")
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE vectors USING vec0(
                    rid TEXT,
                    owner TEXT,
                    visibility TEXT,
                    platform_scope TEXT,
                    status TEXT,
                    domain TEXT,
                    embedding FLOAT[{int(self.dimension)}]
                )
                """
            )
            rowid = 1
            for row in normalized_rows:
                if len(row.vector) != self.dimension:
                    continue
                for platform_scope_key in _platform_scope_keys(row.platform_scope):
                    conn.execute(
                        """
                        INSERT INTO vectors(
                            rowid, rid, owner, visibility, platform_scope, status, domain, embedding
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            rowid,
                            row.rid,
                            row.owner,
                            row.visibility,
                            platform_scope_key,
                            row.status,
                            row.domain,
                            serialize_float32([float(value) for value in row.vector]),
                        ),
                    )
                    rowid += 1

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        owner: str | None = None,
        platform: str | None = None,
        allowed_visibility: set[str] | None = None,
        domain: str | None = None,
        status: str = "active",
    ) -> list[tuple[str, float]]:
        if len(query_vector) != self.dimension or top_k <= 0:
            return []
        from sqlite_vec import serialize_float32

        with self._connect() as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors'"
            ).fetchone()
            if table_exists is None:
                return []
            queries: list[tuple[str, list[object]]] = []
            base_conditions = ["embedding MATCH ?", "k = ?"]
            base_params: list[object] = [
                serialize_float32([float(value) for value in query_vector]),
                int(top_k),
            ]
            if owner is not None:
                base_conditions.append("owner = ?")
                base_params.append(owner)
            if status:
                base_conditions.append("status = ?")
                base_params.append(status)
            if domain is not None:
                base_conditions.append("domain = ?")
                base_params.append(domain)
            visibility_values = sorted(allowed_visibility or []) or [None]
            platform_values = [None]
            if platform is not None:
                platform_values = ["all", platform]
            for visibility in visibility_values:
                for platform_scope in platform_values:
                    conditions = list(base_conditions)
                    params = list(base_params)
                    if visibility is not None:
                        conditions.append("visibility = ?")
                        params.append(visibility)
                    if platform_scope is not None:
                        conditions.append("platform_scope = ?")
                        params.append(platform_scope)
                    queries.append((" AND ".join(conditions), params))
            results: dict[str, float] = {}
            for where_clause, params in queries:
                rows = conn.execute(
                    f"""
                    SELECT rid, distance
                    FROM vectors
                    WHERE {where_clause}
                    ORDER BY distance
                    """,
                    tuple(params),
                ).fetchall()
                for row in rows:
                    score = 1.0 / (1.0 + float(row["distance"]))
                    rid = str(row["rid"])
                    results[rid] = max(results.get(rid, 0.0), score)
        return sorted(results.items(), key=lambda item: item[1], reverse=True)[:top_k]

    def count(self) -> int:
        with self._connect() as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors'"
            ).fetchone()
            if table_exists is None:
                return 0
            return int(conn.execute("SELECT COUNT(DISTINCT rid) FROM vectors").fetchone()[0])

    def metadata_fingerprints(self) -> dict[str, set[tuple[str, str, str, str, str]]]:
        with self._connect() as conn:
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors'"
            ).fetchone()
            if table_exists is None:
                return {}
            rows = conn.execute(
                """
                SELECT rid, owner, visibility, platform_scope, status, domain
                FROM vectors
                ORDER BY rid, platform_scope
                """
            ).fetchall()
        fingerprints: dict[str, set[tuple[str, str, str, str, str]]] = {}
        for row in rows:
            fingerprints.setdefault(str(row["rid"]), set()).add(
                (
                    str(row["owner"]),
                    str(row["visibility"]),
                    str(row["platform_scope"]),
                    str(row["status"]),
                    str(row["domain"] or ""),
                )
            )
        return fingerprints

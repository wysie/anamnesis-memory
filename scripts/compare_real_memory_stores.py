#!/usr/bin/env python3
"""Compare real local memory stores side-by-side.

This is a read-only inspection tool for YC's local stores. It does not call an
LLM and does not mutate YantrikDB, Mnemosyne, or Anamnesis. It answers: for the
same query, what does each store actually contain/retrieve?
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
DEFAULT_QUERIES = [
    "memory dashboard light theme",
    "Anamnesis sandbox gateway approval",
    "WhatsApp memories local only",
    "Hope cannot control devices",
    "Draw Things gRPC port access prompt",
]

TOKEN_RE = re.compile(r"[a-zA-Z0-9_@.:-]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "be", "but", "by", "for", "from", "how", "i", "in",
    "is", "it", "of", "on", "or", "our", "the", "this", "to", "u", "we", "what", "with",
}


@dataclass(frozen=True)
class MemoryHit:
    provider: str
    rid: str
    text: str
    source: str
    status: str
    score: float
    metadata: dict[str, object]


@dataclass(frozen=True)
class StoreSummary:
    provider: str
    path: str
    exists: bool
    total_rows: int
    active_rows: int
    note: str = ""


class StoreAdapter:
    name: str

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()

    def exists(self) -> bool:
        return self.path.exists()

    def summary(self) -> StoreSummary:
        rows = self._rows()
        return StoreSummary(
            provider=self.name,
            path=str(self.path),
            exists=self.exists(),
            total_rows=len(rows),
            active_rows=sum(1 for row in rows if row.status == "active"),
        )

    def search(self, query: str, *, limit: int) -> list[MemoryHit]:
        terms = tokens(query)
        if not terms:
            return []
        hits: list[MemoryHit] = []
        for row in self._rows():
            if row.status != "active":
                continue
            score = lexical_score(terms, row.text)
            if score > 0:
                hits.append(
                    MemoryHit(
                        provider=row.provider,
                        rid=row.rid,
                        text=compact(row.text),
                        source=row.source,
                        status=row.status,
                        score=round(score, 4),
                        metadata=row.metadata,
                    )
                )
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:limit]

    def _rows(self) -> list[MemoryHit]:
        raise NotImplementedError


class YantrikAdapter(StoreAdapter):
    name = "YantrikDB"

    def _rows(self) -> list[MemoryHit]:
        if not self.exists():
            return []
        rows: list[MemoryHit] = []
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(
                """
                SELECT rid,text,domain,source,importance,tombstone_reason,metadata
                FROM memories
                WHERE COALESCE(text, '') <> ''
                """
            ):
                status = "active" if row["tombstone_reason"] is None else "invalidated"
                rows.append(
                    MemoryHit(
                        provider=self.name,
                        rid=str(row["rid"]),
                        text=str(row["text"]),
                        source=str(row["source"] or ""),
                        status=status,
                        score=0,
                        metadata={
                            "domain": row["domain"],
                            "importance": row["importance"],
                        },
                    )
                )
        return rows


class MnemosyneAdapter(StoreAdapter):
    name = "Mnemosyne"

    def _rows(self) -> list[MemoryHit]:
        if not self.exists():
            return []
        rows: list[MemoryHit] = []
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            rows.extend(self._table_rows(conn, table="working_memory", id_col="id", text_col="content", source="working_memory"))
            rows.extend(self._table_rows(conn, table="episodic_memory", id_col="id", text_col="content", source="episodic_memory"))
            rows.extend(self._facts(conn))
            rows.extend(self._consolidated_facts(conn))
        return rows

    def _table_rows(
        self, conn: sqlite3.Connection, *, table: str, id_col: str, text_col: str, source: str
    ) -> list[MemoryHit]:
        if not table_exists(conn, table):
            return []
        out: list[MemoryHit] = []
        cols = table_columns(conn, table)
        superseded_col = "superseded_by" if "superseded_by" in cols else None
        valid_until_col = "valid_until" if "valid_until" in cols else None
        sql = f"SELECT {id_col} AS rid, {text_col} AS text"  # noqa: S608 - fixed table/cols above.
        if superseded_col:
            sql += f", {superseded_col} AS superseded_by"
        if valid_until_col:
            sql += f", {valid_until_col} AS valid_until"
        sql += f" FROM {table} WHERE COALESCE({text_col}, '') <> ''"
        now = time.time()
        for row in conn.execute(sql):
            superseded = bool(row["superseded_by"]) if superseded_col else False
            expired = False
            if valid_until_col and row["valid_until"]:
                try:
                    expired = float(row["valid_until"]) < now
                except (TypeError, ValueError):
                    expired = False
            out.append(
                MemoryHit(
                    provider=self.name,
                    rid=f"{source}:{row['rid']}",
                    text=str(row["text"]),
                    source=source,
                    status="superseded" if superseded else "expired" if expired else "active",
                    score=0,
                    metadata={},
                )
            )
        return out

    def _facts(self, conn: sqlite3.Connection) -> list[MemoryHit]:
        if not table_exists(conn, "facts"):
            return []
        out = []
        for row in conn.execute("SELECT fact_id,subject,predicate,object,confidence FROM facts"):
            text = " ".join(str(row[key] or "") for key in ["subject", "predicate", "object"]).strip()
            if text:
                out.append(
                    MemoryHit(self.name, f"facts:{row['fact_id']}", text, "facts", "active", 0, {"confidence": row["confidence"]})
                )
        return out

    def _consolidated_facts(self, conn: sqlite3.Connection) -> list[MemoryHit]:
        if not table_exists(conn, "consolidated_facts"):
            return []
        out = []
        for row in conn.execute(
            "SELECT id,subject,predicate,object,confidence,superseded_by FROM consolidated_facts"
        ):
            text = " ".join(str(row[key] or "") for key in ["subject", "predicate", "object"]).strip()
            if text:
                out.append(
                    MemoryHit(
                        self.name,
                        f"consolidated_facts:{row['id']}",
                        text,
                        "consolidated_facts",
                        "superseded" if row["superseded_by"] else "active",
                        0,
                        {"confidence": row["confidence"]},
                    )
                )
        return out


class AnamnesisAdapter(StoreAdapter):
    name = "Anamnesis"

    def _rows(self) -> list[MemoryHit]:
        if not self.exists():
            return []
        rows: list[MemoryHit] = []
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            if not table_exists(conn, "memories"):
                return []
            for row in conn.execute(
                """
                SELECT rid,text,status,owner,visibility,platform_scope,domain,source,importance,confidence
                FROM memories
                WHERE COALESCE(text, '') <> ''
                """
            ):
                rows.append(
                    MemoryHit(
                        provider=self.name,
                        rid=str(row["rid"]),
                        text=str(row["text"]),
                        source=str(row["source"] or ""),
                        status=str(row["status"]),
                        score=0,
                        metadata={
                            "owner": row["owner"],
                            "visibility": row["visibility"],
                            "platform_scope": row["platform_scope"],
                            "domain": row["domain"],
                            "importance": row["importance"],
                            "confidence": row["confidence"],
                        },
                    )
                )
        return rows


def main() -> int:
    args = parse_args()
    adapters: list[StoreAdapter] = [
        YantrikAdapter(args.yantrikdb),
        MnemosyneAdapter(args.mnemosyne),
        AnamnesisAdapter(args.anamnesis),
    ]
    queries = args.query or DEFAULT_QUERIES
    report = {
        "method": "read-only lexical side-by-side comparison of local real memory stores",
        "queries": queries,
        "summaries": [asdict(adapter.summary()) for adapter in adapters],
        "results": [],
    }
    for query in queries:
        report["results"].append(
            {
                "query": query,
                "providers": {
                    adapter.name: [asdict(hit) for hit in adapter.search(query, limit=args.limit)]
                    for adapter in adapters
                },
            }
        )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_text(report))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yantrikdb", type=Path, default=Path.home() / ".hermes/yantrikdb-memory.db")
    parser.add_argument("--mnemosyne", type=Path, default=Path.home() / ".hermes/mnemosyne/data/mnemosyne.db")
    parser.add_argument(
        "--anamnesis",
        type=Path,
        default=Path.home() / ".hermes/profiles/anamnesis-sandbox/anamnesis/anamnesis.db",
    )
    parser.add_argument("--query", action="append", help="Query to compare. Repeat for multiple queries.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def render_text(report: dict[str, object]) -> str:
    lines = ["Real memory store comparison", ""]
    lines.append("Stores:")
    for summary in report["summaries"]:  # type: ignore[index]
        s = dict(summary)  # type: ignore[arg-type]
        lines.append(
            f"- {s['provider']}: active={s['active_rows']} total={s['total_rows']} exists={s['exists']} path={s['path']}"
        )
    for result in report["results"]:  # type: ignore[index]
        r = dict(result)  # type: ignore[arg-type]
        lines.extend(["", f"Query: {r['query']}"])
        providers = dict(r["providers"])
        for provider, hits in providers.items():
            lines.append(f"  {provider}:")
            if not hits:
                lines.append("    - <no hits>")
                continue
            for idx, hit in enumerate(hits, start=1):
                h = dict(hit)
                lines.append(
                    f"    {idx}. score={h['score']} status={h['status']} source={h['source']} rid={h['rid']}"
                )
                lines.append(f"       {h['text']}")
    return "\n".join(lines)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return bool(conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (table,)).fetchone())


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def tokens(text: str) -> set[str]:
    return {tok.lower() for tok in TOKEN_RE.findall(text) if tok.lower() not in STOPWORDS and len(tok) > 1}


def lexical_score(query_terms: set[str], text: str) -> float:
    text_terms = tokens(text)
    if not text_terms:
        return 0.0
    overlap = query_terms & text_terms
    if not overlap:
        return 0.0
    coverage = len(overlap) / len(query_terms)
    density = len(overlap) / math.sqrt(len(text_terms))
    exact_bonus = 0.3 if " ".join(sorted(query_terms)) in text.lower() else 0.0
    return coverage * 2.0 + density + exact_bonus


def compact(text: str, *, max_chars: int = 500) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= max_chars else one_line[: max_chars - 1] + "…"


if __name__ == "__main__":
    raise SystemExit(main())

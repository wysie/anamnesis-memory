from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

from .core import Anamnesis, MemoryInboxItem, MemoryRecord
from .embedding_models import DEFAULT_EMBEDDER_MODEL, get_model_spec, load_embedder_by_name
from .synthesis import LocalLLMConfig, synthesize_from_recall
from .vector_index import ExactVectorIndex, SQLiteVecVectorIndex, VectorIndex, sqlite_vec_available
from .embeddings import normalize
from .intake import classify_intake, is_platform_local_text


@dataclass(frozen=True)
class _SpecKeywordEmbedder:
    """Dependency-free embedder for CLI tests and local plumbing checks."""

    name: str
    model_id_value: str
    dimension_value: int

    @property
    def model_id(self) -> str:
        return self.model_id_value

    @property
    def dimension(self) -> int:
        return self.dimension_value

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension_value
        for token in text.lower().replace("-", " ").split():
            idx = sum(ord(ch) for ch in token) % self.dimension_value
            vector[idx] += 1.0
        return normalize(vector)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    store = Anamnesis(args.db)
    if args.command == "embeddings":
        return _handle_embeddings(args, store)
    if args.command == "recall-config":
        return _handle_recall_config(args, store)
    if args.command == "synthesis-config":
        return _handle_synthesis_config(args, store)
    if args.command == "recall":
        return _handle_recall(args, store)
    if args.command == "inbox":
        return _handle_inbox(args, store)
    if args.command == "synthesize":
        return _handle_synthesize(args, store)
    if args.command == "simulate":
        return _handle_simulate(args, store)
    if args.command == "preview-turn":
        return _handle_preview_turn(args, store)
    if args.command == "preview-batch":
        return _handle_preview_batch(args, store)
    if args.command == "preview-memory-write":
        return _handle_preview_memory_write(args, store)
    if args.command == "correct":
        return _handle_correct(args, store)
    if args.command == "audit":
        return _handle_audit(args, store)
    if args.command == "maintenance":
        return _handle_maintenance(args, store)
    parser.error("missing command")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anamnesis")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path.home() / ".anamnesis" / "anamnesis.db",
        help="Path to Anamnesis SQLite DB.",
    )
    subparsers = parser.add_subparsers(dest="command")

    embeddings = subparsers.add_parser("embeddings", help="Manage embedding model cache.")
    embedding_sub = embeddings.add_subparsers(dest="embedding_command", required=True)

    status = embedding_sub.add_parser("status", help="Show embedding coverage for active model.")
    status.add_argument("--model", help="Model name to inspect instead of active model.")
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    switch = embedding_sub.add_parser("switch", help="Persist the active embedding model name.")
    switch.add_argument("model", help="Official embedding model name, e.g. potion-base-32M.")

    backfill = embedding_sub.add_parser("backfill", help="Embed rows missing for a model.")
    backfill.add_argument("--model", help="Model to backfill. Defaults to active model.")
    backfill.add_argument(
        "--test-keyword-embedder",
        action="store_true",
        help="Use deterministic dependency-free vectors; intended for tests/smoke checks.",
    )

    index_status = embedding_sub.add_parser("index-status", help="Show sqlite-vec index coverage.")
    index_status.add_argument("--model", help="Model to inspect. Defaults to active model.")
    index_status.add_argument("--index-db", type=Path, help="Path to sqlite-vec index DB.")
    index_status.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    index_rebuild = embedding_sub.add_parser("index-rebuild", help="Rebuild sqlite-vec index for a model.")
    index_rebuild.add_argument("--model", help="Model to rebuild. Defaults to active model.")
    index_rebuild.add_argument("--index-db", type=Path, help="Path to sqlite-vec index DB.")

    benchmark = embedding_sub.add_parser("benchmark", help="Compare embedding models on a small recall suite.")
    benchmark.add_argument(
        "--models",
        default="potion-base-2M,potion-base-8M,potion-base-32M,potion-retrieval-32M",
        help="Comma-separated official model names to compare.",
    )
    benchmark.add_argument(
        "--test-keyword-embedder",
        action="store_true",
        help="Use deterministic dependency-free vectors; intended for tests/smoke checks.",
    )
    benchmark.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    benchmark.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Recall repetitions per case for latency measurement.",
    )
    benchmark.add_argument(
        "--synthetic-count",
        type=int,
        default=0,
        help="Add N synthetic distractor memories per model to measure scale effects.",
    )
    benchmark.add_argument(
        "--include-adversarial",
        action="store_true",
        help="Include privacy/invalidation/distractor cases in the benchmark score.",
    )
    benchmark.add_argument(
        "--vector-candidate-limit",
        type=int,
        default=1000,
        help="Limit vector scoring to top keyword candidates when keyword recall has hits; use 0 to disable pruning.",
    )
    benchmark.add_argument(
        "--ann-candidate-limit",
        type=int,
        default=0,
        help="Use a VectorIndex backend to add this many ANN-style semantic candidates; 0 disables ANN.",
    )
    benchmark.add_argument(
        "--ann-backend",
        choices=("exact", "sqlite-vec"),
        default="sqlite-vec",
        help="VectorIndex backend to use when --ann-candidate-limit is enabled.",
    )
    benchmark.add_argument(
        "--recall-policy",
        choices=("latency_first", "recall_first", "semantic_only"),
        default="latency_first",
        help="Hybrid recall policy for benchmark queries.",
    )
    benchmark.add_argument(
        "--ann-min-keyword-candidates",
        type=int,
        default=50,
        help="In latency_first mode, use ANN only when keyword candidates are below this count.",
    )
    recall_config = subparsers.add_parser("recall-config", help="Configure runtime recall defaults.")
    recall_config_sub = recall_config.add_subparsers(dest="recall_config_command", required=True)
    recall_config_set = recall_config_sub.add_parser("set", help="Persist recall defaults.")
    recall_config_set.add_argument("--model", help="Embedding model used for recall.")
    recall_config_set.add_argument("--ann-backend", choices=("exact", "sqlite-vec"), default="sqlite-vec")
    recall_config_set.add_argument("--index-db", type=Path, help="Path to sqlite-vec index DB.")
    recall_config_set.add_argument(
        "--recall-policy",
        choices=("latency_first", "recall_first", "semantic_only"),
        default="latency_first",
    )
    recall_config_set.add_argument("--ann-candidate-limit", type=int, default=50)
    recall_config_set.add_argument("--ann-min-keyword-candidates", type=int, default=50)
    recall_config_set.add_argument("--vector-candidate-limit", type=int, default=1000)
    recall_config_set.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    synthesis_config = subparsers.add_parser("synthesis-config", help="Configure local/private LLM synthesis defaults.")
    synthesis_config_sub = synthesis_config.add_subparsers(dest="synthesis_config_command", required=True)
    synthesis_config_set = synthesis_config_sub.add_parser("set", help="Persist synthesis defaults.")
    synthesis_config_set.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8060/v1")
    synthesis_config_set.add_argument("--model", required=True, help="Local/private model name sent to the endpoint.")
    synthesis_config_set.add_argument("--api-key-env", help="Environment variable holding the endpoint API key. Omit for no auth.")
    synthesis_config_set.add_argument("--temperature", type=float, default=0.0)
    synthesis_config_set.add_argument("--max-tokens", type=int, default=512)
    synthesis_config_set.add_argument("--timeout", type=int, default=60)
    synthesis_config_set.add_argument("--max-context-chars", type=int, default=8000)
    synthesis_config_set.add_argument("--max-memory-chars", type=int, default=1200)
    synthesis_config_set.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    recall = subparsers.add_parser("recall", help="Run governed recall for a query.")
    recall.add_argument("query")
    recall.add_argument("--owner", required=True)
    recall.add_argument("--platform", required=True)
    recall.add_argument("--visibility", action="append", default=["private"])
    recall.add_argument("--domain")
    recall.add_argument("--limit", type=int, default=10)
    recall.add_argument("--model", help="Embedding model to use; defaults to recall config/active model.")
    recall.add_argument("--index-db", type=Path, help="Path to sqlite-vec index DB.")
    recall.add_argument("--test-keyword-embedder", action="store_true")
    recall.add_argument("--explain", action="store_true")
    recall.add_argument("--json", action="store_true")

    inbox = subparsers.add_parser("inbox", help="Review proposed memories before they become recallable.")
    inbox_sub = inbox.add_subparsers(dest="inbox_command", required=True)

    inbox_list = inbox_sub.add_parser("list", help="List inbox items by decision state.")
    inbox_list.add_argument("--decision", choices=("pending", "accepted", "rejected", "expired"), default="pending")
    inbox_list.add_argument("--limit", type=int, default=50)
    inbox_list.add_argument("--json", action="store_true")

    inbox_show = inbox_sub.add_parser("show", help="Show one inbox item.")
    inbox_show.add_argument("cid")
    inbox_show.add_argument("--json", action="store_true")

    inbox_propose = inbox_sub.add_parser("propose", help="Create a pending inbox item.")
    inbox_propose.add_argument("text")
    inbox_propose.add_argument("--source-snippet", default="")
    inbox_propose.add_argument("--kind", default="semantic")
    inbox_propose.add_argument("--owner", default="default")
    inbox_propose.add_argument("--visibility", default="private")
    inbox_propose.add_argument("--platform", dest="platform_scope", default="all")
    inbox_propose.add_argument("--action-scope", default="all")
    inbox_propose.add_argument("--domain", default="")
    inbox_propose.add_argument("--source", default="cli")
    inbox_propose.add_argument("--confidence", type=float, default=0.5)
    inbox_propose.add_argument("--why-save", default="")
    inbox_propose.add_argument("--lifecycle", default="")
    inbox_propose.add_argument("--json", action="store_true")

    inbox_accept = inbox_sub.add_parser("accept", help="Accept a pending inbox item into canonical memory.")
    inbox_accept.add_argument("cid")
    inbox_accept.add_argument("--json", action="store_true")

    inbox_reject = inbox_sub.add_parser("reject", help="Reject a pending inbox item.")
    inbox_reject.add_argument("cid")
    inbox_reject.add_argument("--reason", default="")
    inbox_reject.add_argument("--json", action="store_true")

    inbox_expire = inbox_sub.add_parser("expire", help="Expire stale pending inbox items.")
    inbox_expire.add_argument("--max-age-days", type=int, default=30)
    inbox_expire.add_argument("--reason", default="expired pending review")
    inbox_expire.add_argument("--json", action="store_true")

    synthesize = subparsers.add_parser("synthesize", help="Recall memories, then synthesize a cited read-only answer with a local/private LLM.")
    synthesize.add_argument("query")
    synthesize.add_argument("--owner", required=True)
    synthesize.add_argument("--platform", required=True)
    synthesize.add_argument("--visibility", action="append", default=["private"])
    synthesize.add_argument("--domain")
    synthesize.add_argument("--limit", type=int, default=10)
    synthesize.add_argument("--model", help="Embedding model to use for recall; defaults to recall config/active model.")
    synthesize.add_argument("--index-db", type=Path, help="Path to sqlite-vec index DB.")
    synthesize.add_argument("--test-keyword-embedder", action="store_true")
    synthesize.add_argument("--llm-base-url", help="Override configured local/private LLM base URL.")
    synthesize.add_argument("--llm-model", help="Override configured local/private LLM model.")
    synthesize.add_argument("--api-key-env", help="Override configured local/private LLM API key env var.")
    synthesize.add_argument("--max-tokens", type=int)
    synthesize.add_argument("--temperature", type=float)
    synthesize.add_argument("--max-context-chars", type=int)
    synthesize.add_argument("--max-memory-chars", type=int)
    synthesize.add_argument("--json", action="store_true")

    simulate = subparsers.add_parser("simulate", help="Explain which memories would be recalled or excluded.")
    simulate.add_argument("query")
    simulate.add_argument("--owner", required=True)
    simulate.add_argument("--platform", required=True)
    simulate.add_argument("--visibility", action="append", default=["private"])
    simulate.add_argument("--domain")
    simulate.add_argument("--limit", type=int, default=10)
    simulate.add_argument("--sample-limit", type=int, default=200)
    simulate.add_argument("--json", action="store_true")

    preview_turn = subparsers.add_parser("preview-turn", help="Preview intake and recall for one turn without writing memory.")
    preview_turn.add_argument("text")
    preview_turn.add_argument("--owner", required=True)
    preview_turn.add_argument("--platform", required=True)
    preview_turn.add_argument("--visibility", action="append", default=["private"])
    preview_turn.add_argument("--domain", default="")
    preview_turn.add_argument("--limit", type=int, default=5)
    preview_turn.add_argument("--json", action="store_true")

    preview_batch = subparsers.add_parser("preview-batch", help="Preview/apply intake and recall over a JSONL or text transcript.")
    preview_batch.add_argument("transcript", type=Path)
    preview_batch.add_argument("--owner", required=True)
    preview_batch.add_argument("--platform", required=True)
    preview_batch.add_argument("--visibility", action="append", default=["private"])
    preview_batch.add_argument("--domain", default="")
    preview_batch.add_argument("--limit", type=int, default=5)
    preview_batch.add_argument("--apply", action="store_true")
    preview_batch.add_argument("--json", action="store_true")

    preview_memory = subparsers.add_parser(
        "preview-memory-write",
        help="Preview/apply the governed policy for a Hermes memory-tool write.",
    )
    preview_memory.add_argument("text")
    preview_memory.add_argument("--target", default="memory")
    preview_memory.add_argument("--origin", default="")
    preview_memory.add_argument("--owner", required=True)
    preview_memory.add_argument("--platform", required=True)
    preview_memory.add_argument("--visibility", default="private")
    preview_memory.add_argument("--apply", action="store_true")
    preview_memory.add_argument("--json", action="store_true")

    correct = subparsers.add_parser("correct", help="Invalidate a memory and replace it with corrected text.")
    correct.add_argument("rid", help="Active memory rid to correct.")
    correct.add_argument("text", help="Replacement memory text.")
    correct.add_argument("--reason", default="", help="Audit reason for the correction.")
    correct.add_argument("--json", action="store_true")

    audit = subparsers.add_parser("audit", help="Show audit events for a memory rid.")
    audit.add_argument("rid", help="Memory rid to inspect.")
    audit.add_argument("--json", action="store_true")

    maintenance = subparsers.add_parser("maintenance", help="Run local memory hygiene tasks.")
    maintenance_sub = maintenance.add_subparsers(dest="maintenance_command", required=True)
    supersede = maintenance_sub.add_parser("supersede-duplicates", help="Mark near-duplicate active memories as superseded.")
    supersede.add_argument("--owner")
    supersede.add_argument("--domain")
    supersede.add_argument("--threshold", type=float, default=0.9)
    supersede.add_argument("--json", action="store_true")

    autopilot = maintenance_sub.add_parser("autopilot", help="Run safe local memory hygiene tasks.")
    autopilot.add_argument("--owner")
    autopilot.add_argument("--domain")
    autopilot.add_argument("--max-inbox-age-days", type=int, default=30)
    autopilot.add_argument("--duplicate-threshold", type=float, default=0.9)
    autopilot.add_argument("--json", action="store_true")
    report = maintenance_sub.add_parser("report", help="Show recent maintenance runs.")
    report.add_argument("--limit", type=int, default=5)
    report.add_argument("--json", action="store_true")

    return parser




def _preview_platform_scope(text: str, *, platform: str, lifecycle: str) -> str:
    if lifecycle == "sensitive" or is_platform_local_text(text):
        return platform
    return "all"


def _preview_turn_payload(
    store: Anamnesis,
    *,
    text: str,
    owner: str,
    platform: str,
    visibility: list[str],
    domain: str,
    limit: int,
) -> dict[str, object]:
    decision = classify_intake(text, domain=domain or "")
    platform_scope = _preview_platform_scope(
        text, platform=platform, lifecycle=decision.lifecycle
    )
    simulation = store.simulate_recall(
        text,
        owner=owner,
        platform=platform,
        allowed_visibility=set(visibility or ["private"]),
        limit=max(1, limit),
        domain=domain or None,
    )
    return {
        "input": {
            "text": text,
            "owner": owner,
            "platform": platform,
            "domain": domain or "",
        },
        "would_write": {
            "action": decision.action,
            "lifecycle": decision.lifecycle,
            "reasons": decision.reasons,
            "confidence": decision.confidence,
            "platform_scope": platform_scope,
            "source_platform": platform,
        },
        "would_inject": {
            "included": simulation["included"],
            "context_preview": simulation["context_preview"],
        },
    }


def _handle_preview_turn(args: argparse.Namespace, store: Anamnesis) -> int:
    payload = {
        "mode": "preview",
        **_preview_turn_payload(
            store,
            text=args.text,
            owner=args.owner,
            platform=args.platform,
            visibility=args.visibility,
            domain=args.domain or "",
            limit=args.limit,
        ),
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        would_write = cast(dict[str, object], payload["would_write"])
        print(
            f"would_write={would_write['action']} lifecycle={would_write['lifecycle']} "
            f"platform_scope={would_write['platform_scope']} source_platform={args.platform}"
        )
        would_inject = cast(dict[str, object], payload["would_inject"])
        print(str(would_inject.get("context_preview") or ""))
    return 0


def _extract_turn_text(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        for key in ("text", "user", "user_content", "content", "message"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _apply_preview_write(
    store: Anamnesis,
    *,
    text: str,
    owner: str,
    visibility: str,
    domain: str,
    payload: dict[str, object],
    source: str = "preview_batch",
    metadata: dict[str, object] | None = None,
) -> dict[str, object] | None:
    would_write = cast(dict[str, object], payload["would_write"])
    action = str(would_write["action"])
    write_metadata = {
        "source_platform": would_write["source_platform"],
        "intake_reasons": would_write["reasons"],
        "intake_lifecycle": would_write["lifecycle"],
        **(metadata or {}),
    }
    if action == "accept":
        record = store.add_memory(
            text,
            owner=owner,
            visibility=visibility,
            platform_scope=str(would_write["platform_scope"]),
            domain=domain or str(would_write["lifecycle"]),
            source=source,
            confidence=float(would_write["confidence"]),
            metadata=write_metadata,
        )
        return {"rid": record.rid, "kind": "memory"}
    if action == "inbox":
        item = store.propose_memory(
            text,
            source_snippet=text[:500],
            owner=owner,
            visibility=visibility,
            platform_scope=str(would_write["platform_scope"]),
            domain=domain or str(would_write["lifecycle"]),
            source="preview_batch",
            confidence=float(would_write["confidence"]),
            why_save=", ".join(cast(list[str], would_write["reasons"])),
            suggested_lifecycle=str(would_write["lifecycle"]),
        )
        return {"cid": item.cid, "kind": "inbox_item"}
    return None


def _handle_preview_batch(args: argparse.Namespace, store: Anamnesis) -> int:
    turns: list[dict[str, object]] = []
    summary = {"total": 0, "accept": 0, "inbox": 0, "reject": 0}
    reason_counts: dict[str, int] = {}
    for index, line in enumerate(args.transcript.read_text(encoding="utf-8").splitlines(), start=1):
        text = _extract_turn_text(line)
        if not text:
            continue
        payload = _preview_turn_payload(
            store,
            text=text,
            owner=args.owner,
            platform=args.platform,
            visibility=args.visibility,
            domain=args.domain or "",
            limit=args.limit,
        )
        would_write = cast(dict[str, object], payload["would_write"])
        action = str(would_write["action"])
        summary["total"] += 1
        summary[action] += 1
        for reason in cast(list[str], would_write["reasons"]):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        applied = None
        if args.apply:
            applied = _apply_preview_write(
                store,
                text=text,
                owner=args.owner,
                visibility=(args.visibility or ["private"])[0],
                domain=args.domain or "",
                payload=payload,
                metadata={"preview_batch_applied": True},
            )
        turns.append({"line": index, **payload, "applied": applied})
    result = {
        "mode": "preview_batch",
        "apply": bool(args.apply),
        "summary": summary,
        "reason_counts": reason_counts,
        "turns": turns,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"total={summary['total']} accept={summary['accept']} inbox={summary['inbox']} reject={summary['reject']}")
    return 0


def _handle_preview_memory_write(args: argparse.Namespace, store: Anamnesis) -> int:
    payload = _preview_turn_payload(
        store,
        text=args.text,
        owner=args.owner,
        platform=args.platform,
        visibility=[args.visibility],
        domain=args.target,
        limit=1,
    )
    payload["mode"] = "preview_memory_write"
    input_payload = cast(dict[str, object], payload["input"])
    input_payload["target"] = args.target
    input_payload["origin"] = args.origin
    input_payload["source"] = "hermes_memory_tool"
    applied = None
    if args.apply:
        applied = _apply_preview_write(
            store,
            text=args.text,
            owner=args.owner,
            visibility=args.visibility,
            domain=args.target,
            payload=payload,
            source="hermes_memory_tool",
            metadata={"origin": args.origin, "preview_memory_write_applied": True},
        )
    result = {**payload, "apply": bool(args.apply), "applied": applied}
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        would_write = cast(dict[str, object], payload["would_write"])
        print(
            f"would_write={would_write['action']} target={args.target} "
            f"origin={args.origin or '(none)'} platform_scope={would_write['platform_scope']}"
        )
        if applied:
            print(f"applied={applied}")
    return 0


def _handle_correct(args: argparse.Namespace, store: Anamnesis) -> int:
    replacement = store.correct_memory(args.rid, args.text, reason=args.reason)
    old = store.get_memory(args.rid)
    payload = {
        "old": _memory_record_dict(old),
        "replacement": _memory_record_dict(replacement),
        "reason": args.reason,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"corrected={old.rid} replacement={replacement.rid}")
        print(replacement.text)
    return 0


def _audit_payload(store: Anamnesis, rid: str) -> dict[str, object]:
    record = store.get_memory(rid)
    events = store.audit_events(rid)
    chain: dict[str, str] = {}
    for event in events:
        metadata = cast(dict[str, object], event.get("metadata", {}))
        if event.get("event_type") == "memory_corrected_from" and metadata.get("replacement_rid"):
            chain["replacement_rid"] = str(metadata["replacement_rid"])
        if event.get("event_type") == "memory_corrected_to" and metadata.get("old_rid"):
            chain["old_rid"] = str(metadata["old_rid"])
    if "corrects_rid" in record.metadata:
        chain.setdefault("old_rid", str(record.metadata["corrects_rid"]))
    return {
        "rid": rid,
        "memory": _memory_record_dict(record),
        "events": events,
        "correction_chain": chain,
    }


def _format_audit(payload: dict[str, object]) -> str:
    memory = cast(dict[str, object], payload["memory"])
    lines = [
        f"rid={payload['rid']}",
        f"status={memory['status']} domain={memory['domain']} source={memory['source']}",
        f"text={memory['text']}",
    ]
    chain = cast(dict[str, str], payload.get("correction_chain", {}))
    if chain:
        lines.append("correction_chain=" + " ".join(f"{key}={value}" for key, value in chain.items()))
    lines.append("events:")
    for event in cast(list[dict[str, object]], payload["events"]):
        reason = f" reason={event['reason']}" if event.get("reason") else ""
        lines.append(f"  {event['event_type']}{reason}")
    return "\n".join(lines)


def _handle_audit(args: argparse.Namespace, store: Anamnesis) -> int:
    payload = _audit_payload(store, args.rid)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(_format_audit(payload))
    return 0


def _record_maintenance_run(store: Anamnesis, payload: dict[str, object]) -> None:
    with store._connect() as conn:  # noqa: SLF001 - CLI writes maintenance audit event.
        conn.execute(
            "INSERT INTO audit_log (rid,event_type,reason,created_at,metadata_json) VALUES (?,?,?,?,?)",
            (
                "maintenance",
                "maintenance_autopilot",
                "autopilot run",
                time.time(),
                json.dumps(payload, sort_keys=True),
            ),
        )


def _maintenance_report(store: Anamnesis, *, limit: int) -> list[dict[str, object]]:
    with store._connect() as conn:  # noqa: SLF001 - CLI reads audit report.
        rows = conn.execute(
            """
            SELECT event_type, reason, created_at, metadata_json
            FROM audit_log
            WHERE event_type='maintenance_autopilot'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    runs: list[dict[str, object]] = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        runs.append(
            {
                "event_type": row["event_type"],
                "reason": row["reason"],
                "created_at": row["created_at"],
                "summary": metadata.get("summary", {}),
                "metadata": metadata,
            }
        )
    return runs


def _handle_maintenance(args: argparse.Namespace, store: Anamnesis) -> int:
    if args.maintenance_command == "supersede-duplicates":
        superseded = store.supersede_duplicate_memories(
            owner=args.owner, domain=args.domain, threshold=args.threshold
        )
        payload = {"superseded": superseded}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            if not superseded:
                print("no duplicate memories superseded")
            else:
                for item in superseded:
                    print(
                        f"superseded={item['superseded_rid']} canonical={item['canonical_rid']} overlap={item['overlap']}"
                    )
        return 0
    if args.maintenance_command == "autopilot":
        expired = store.expire_pending_inbox_items(
            max_age_days=max(0, args.max_inbox_age_days),
            reason="autopilot stale pending expiry",
            owner=args.owner,
            domain=args.domain,
        )
        superseded = store.supersede_duplicate_memories(
            owner=args.owner, domain=args.domain, threshold=args.duplicate_threshold
        )
        payload = {
            "expired_inbox": [_inbox_item_dict(item) for item in expired],
            "superseded_duplicates": superseded,
            "summary": {
                "expired_inbox": len(expired),
                "superseded_duplicates": len(superseded),
            },
        }
        _record_maintenance_run(store, payload)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(
                f"expired_inbox={len(expired)} superseded_duplicates={len(superseded)}"
            )
        return 0
    if args.maintenance_command == "report":
        payload = {"runs": _maintenance_report(store, limit=max(1, args.limit))}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            for run in payload["runs"]:
                print(f"{run['created_at']} {run['event_type']} {run['summary']}")
        return 0
    raise AssertionError(f"unknown maintenance command {args.maintenance_command!r}")

def _handle_simulate(args: argparse.Namespace, store: Anamnesis) -> int:
    payload = store.simulate_recall(
        args.query,
        owner=args.owner,
        platform=args.platform,
        allowed_visibility=set(args.visibility or ["private"]),
        limit=max(1, args.limit),
        domain=args.domain,
        sample_limit=max(1, args.sample_limit),
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(_format_simulation(payload))
    return 0


def _format_simulation(payload: dict[str, object]) -> str:
    lines = [f"query={payload['query']}", "included:"]
    included = cast(list[dict[str, object]], payload.get("included", []))
    if not included:
        lines.append("  none")
    for item in included:
        reasons = ",".join(cast(list[str], item.get("reasons", [])))
        lines.append(f"  {item['rid']} score={item['score']} reasons={reasons} text={item['text']}")
    lines.append("excluded:")
    excluded = cast(list[dict[str, object]], payload.get("excluded", []))
    if not excluded:
        lines.append("  none")
    for item in excluded[:50]:
        identifier = item.get("rid") or item.get("cid")
        reasons = ",".join(cast(list[str], item.get("exclusion_reasons", [])))
        lines.append(f"  {identifier} reasons={reasons} text={item['text']}")
    if len(excluded) > 50:
        lines.append(f"  ... {len(excluded) - 50} more excluded")
    return "\n".join(lines)

def _handle_inbox(args: argparse.Namespace, store: Anamnesis) -> int:
    if args.inbox_command == "list":
        items = store.inbox_items(decision=args.decision, limit=max(1, args.limit))
        payload = {"decision": args.decision, "items": [_inbox_item_dict(item) for item in items]}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(_format_inbox_list(payload))
        return 0

    if args.inbox_command == "show":
        item = store.get_inbox_item(args.cid)
        payload = _inbox_item_dict(item)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(_format_inbox_item(payload))
        return 0

    if args.inbox_command == "propose":
        item = store.propose_memory(
            args.text,
            source_snippet=args.source_snippet,
            proposed_kind=args.kind,
            owner=args.owner,
            visibility=args.visibility,
            platform_scope=args.platform_scope,
            action_scope=args.action_scope,
            domain=args.domain,
            source=args.source,
            confidence=max(0.0, min(1.0, args.confidence)),
            why_save=args.why_save,
            suggested_lifecycle=args.lifecycle,
        )
        payload = _inbox_item_dict(item)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(_format_inbox_item(payload))
        return 0

    if args.inbox_command == "accept":
        record = store.accept_inbox_item(args.cid)
        item = store.get_inbox_item(args.cid)
        payload = _inbox_item_dict(item)
        payload["memory"] = _memory_record_dict(record)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(_format_inbox_item(payload))
        return 0

    if args.inbox_command == "reject":
        item = store.reject_inbox_item(args.cid, reason=args.reason)
        payload = _inbox_item_dict(item)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(_format_inbox_item(payload))
        return 0

    if args.inbox_command == "expire":
        items = store.expire_pending_inbox_items(
            max_age_days=max(0, args.max_age_days), reason=args.reason
        )
        payload = {"expired": [_inbox_item_dict(item) for item in items]}
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(_format_inbox_list({"decision": "expired", "items": payload["expired"]}))
        return 0

    raise AssertionError(f"unknown inbox command {args.inbox_command!r}")


def _inbox_item_dict(item: MemoryInboxItem) -> dict[str, object]:
    return {
        "cid": item.cid,
        "proposed_text": item.proposed_text,
        "source_snippet": item.source_snippet,
        "proposed_kind": item.proposed_kind,
        "owner": item.owner,
        "visibility": item.visibility,
        "platform_scope": item.platform_scope,
        "action_scope": item.action_scope,
        "domain": item.domain,
        "source": item.source,
        "confidence": item.confidence,
        "why_save": item.why_save,
        "suggested_lifecycle": item.suggested_lifecycle,
        "decision": item.decision,
        "review_reason": item.review_reason,
        "duplicate_rids": item.duplicate_rids,
        "hints": item.hints,
        "accepted_rid": item.accepted_rid,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
    }


def _memory_record_dict(record: MemoryRecord) -> dict[str, object]:
    return {
        "rid": record.rid,
        "text": record.text,
        "kind": record.kind,
        "owner": record.owner,
        "visibility": record.visibility,
        "platform_scope": record.platform_scope,
        "action_scope": record.action_scope,
        "domain": record.domain,
        "source": record.source,
        "importance": record.importance,
        "confidence": record.confidence,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _format_inbox_list(payload: dict[str, object]) -> str:
    items = list(payload.get("items", []))
    if not items:
        return f"no {payload.get('decision')} inbox items"
    lines = []
    for raw in items:
        item = cast(dict[str, object], raw)
        hints = ",".join(cast(list[str], item.get("hints", [])))
        suffix = f" hints={hints}" if hints else ""
        lines.append(f"{item['cid']} [{item['decision']}] {item['proposed_text']}{suffix}")
    return "\n".join(lines)


def _format_inbox_item(item: dict[str, object]) -> str:
    lines = [
        f"cid={item['cid']}",
        f"decision={item['decision']}",
        f"text={item['proposed_text']}",
        f"owner={item['owner']} visibility={item['visibility']} platform={item['platform_scope']} domain={item['domain']}",
    ]
    if item.get("accepted_rid"):
        lines.append(f"accepted_rid={item['accepted_rid']}")
    if item.get("review_reason"):
        lines.append(f"review_reason={item['review_reason']}")
    if item.get("hints"):
        lines.append(f"hints={','.join(cast(list[str], item['hints']))}")
    return "\n".join(lines)

def _handle_recall_config(args: argparse.Namespace, store: Anamnesis) -> int:
    if args.recall_config_command != "set":
        raise AssertionError(f"unknown recall-config command {args.recall_config_command!r}")
    model_name = args.model or store.active_embedding_model() or DEFAULT_EMBEDDER_MODEL
    spec = get_model_spec(model_name)
    store.set_recall_config(
        model=spec.name,
        ann_backend=args.ann_backend,
        index_db=args.index_db,
        recall_policy=args.recall_policy,
        ann_candidate_limit=max(0, args.ann_candidate_limit),
        ann_min_keyword_candidates=max(0, args.ann_min_keyword_candidates),
        vector_candidate_limit=max(0, args.vector_candidate_limit),
    )
    payload = _normalized_recall_config(store)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(_format_recall_config(payload))
    return 0


def _handle_synthesis_config(args: argparse.Namespace, store: Anamnesis) -> int:
    if args.synthesis_config_command != "set":
        raise AssertionError(f"unknown synthesis-config command {args.synthesis_config_command!r}")
    payload = store.set_synthesis_config(
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=max(1, args.max_tokens),
        timeout=max(1, args.timeout),
        max_context_chars=max(1, args.max_context_chars),
        max_memory_chars=max(1, args.max_memory_chars),
    )
    normalized = _normalized_synthesis_config(store, payload)
    if args.json:
        print(json.dumps(normalized, sort_keys=True))
    else:
        print(_format_synthesis_config(normalized))
    return 0


def _handle_recall(args: argparse.Namespace, store: Anamnesis) -> int:
    config = _normalized_recall_config(store)
    model_name = args.model or str(config["model"])
    embedder = _embedder_for_backfill(model_name, test_keyword=args.test_keyword_embedder)
    index_db = args.index_db or Path(str(config["index_db"]))
    vector_index: VectorIndex | None = None
    ann_candidate_limit = int(str(config["ann_candidate_limit"]))
    ann_backend = str(config["ann_backend"])
    if ann_candidate_limit > 0:
        vector_index = _create_vector_index(
            backend=ann_backend,
            db_path=index_db,
            model_id=embedder.model_id,
            dimension=embedder.dimension,
        )
    allowed_visibility = set(args.visibility or ["private"])
    results = store.recall(
        args.query,
        owner=args.owner,
        platform=args.platform,
        allowed_visibility=allowed_visibility,
        limit=max(1, args.limit),
        domain=args.domain,
        embedder=embedder,
        vector_candidate_limit=int(str(config["vector_candidate_limit"])),
        vector_index=vector_index,
        ann_candidate_limit=ann_candidate_limit,
        recall_policy=str(config["recall_policy"]),
        ann_min_keyword_candidates=int(str(config["ann_min_keyword_candidates"])),
    )
    result_payload = [
        {
            "rid": result.record.rid,
            "text": result.record.text,
            "score": result.score,
            "reasons": result.reasons,
            "domain": result.record.domain,
        }
        for result in results
    ]
    payload: dict[str, object] = {"query": args.query, "config": config, "results": result_payload}
    if args.explain:
        payload["explain"] = {
            "ann_searched": vector_index is not None and ann_candidate_limit > 0,
            "result_count": len(result_payload),
            "allowed_visibility": sorted(allowed_visibility),
            "owner": args.owner,
            "platform": args.platform,
            "domain": args.domain,
        }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(_format_recall_results(payload))
    return 0


def _handle_synthesize(args: argparse.Namespace, store: Anamnesis) -> int:
    config = _normalized_recall_config(store)
    model_name = args.model or str(config["model"])
    embedder = _embedder_for_backfill(model_name, test_keyword=args.test_keyword_embedder)
    index_db = args.index_db or Path(str(config["index_db"]))
    vector_index: VectorIndex | None = None
    ann_candidate_limit = int(str(config["ann_candidate_limit"]))
    ann_backend = str(config["ann_backend"])
    if ann_candidate_limit > 0:
        vector_index = _create_vector_index(
            backend=ann_backend,
            db_path=index_db,
            model_id=embedder.model_id,
            dimension=embedder.dimension,
        )
    allowed_visibility = set(args.visibility or ["private"])
    results = store.recall(
        args.query,
        owner=args.owner,
        platform=args.platform,
        allowed_visibility=allowed_visibility,
        limit=max(1, args.limit),
        domain=args.domain,
        embedder=embedder,
        vector_candidate_limit=int(str(config["vector_candidate_limit"])),
        vector_index=vector_index,
        ann_candidate_limit=ann_candidate_limit,
        recall_policy=str(config["recall_policy"]),
        ann_min_keyword_candidates=int(str(config["ann_min_keyword_candidates"])),
    )
    llm_config = _local_llm_config_from_args(args, store)
    synthesis = synthesize_from_recall(args.query, results, llm_config)
    payload = {
        "query": args.query,
        "answer": synthesis.answer,
        "model": synthesis.model,
        "memory_ids": synthesis.memory_ids,
        "cited_memory_ids": synthesis.cited_memory_ids,
        "uncited_memory_ids": synthesis.uncited_memory_ids,
        "citation_missing": synthesis.citation_missing,
        "insufficient_evidence": synthesis.insufficient_evidence,
        "retry_count": synthesis.retry_count,
        "packed_char_count": synthesis.packed_char_count,
        "truncated_memory_ids": synthesis.truncated_memory_ids,
        "recall_count": len(results),
        "read_only": True,
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(synthesis.answer)
        print("\nCited memories: " + ", ".join(synthesis.cited_memory_ids or []))
    return 0


def _normalized_recall_config(store: Anamnesis) -> dict[str, object]:
    raw = store.recall_config()
    model_name = raw.get("model") or store.active_embedding_model() or DEFAULT_EMBEDDER_MODEL
    spec = get_model_spec(model_name)
    model_id = spec.model_id
    index_db = raw.get("index_db") or str(_default_vector_index_db(store.db_path, model_id))
    return {
        "model": spec.name,
        "model_id": model_id,
        "dimension": spec.dimension,
        "ann_backend": raw.get("ann_backend", "sqlite-vec"),
        "index_db": index_db,
        "recall_policy": raw.get("recall_policy", "latency_first"),
        "ann_candidate_limit": int(raw.get("ann_candidate_limit", "50")),
        "ann_min_keyword_candidates": int(raw.get("ann_min_keyword_candidates", "50")),
        "vector_candidate_limit": int(raw.get("vector_candidate_limit", "1000")),
    }


def _format_recall_config(payload: dict[str, object]) -> str:
    return "\n".join(f"{key}={value}" for key, value in payload.items())


def _format_recall_results(payload: dict[str, object]) -> str:
    lines = [f"query={payload['query']}"]
    for result in cast(list[dict[str, object]], payload["results"]):
        lines.append(
            f"rid={result['rid']} score={result['score']} reasons={','.join(cast(list[str], result['reasons']))}"
        )
        lines.append(str(result["text"]))
    return "\n".join(lines)


def _handle_embeddings(args: argparse.Namespace, store: Anamnesis) -> int:
    if args.embedding_command == "switch":
        spec = get_model_spec(args.model)
        store.set_active_embedding_model(spec.name)
        status = _status_payload(store, spec.name)
        print(
            f"active_model={spec.name} model_id={spec.model_id} "
            f"dimension={spec.dimension} missing={status['missing']} "
            f"backfill_required={str(status['backfill_required']).lower()}"
        )
        return 0

    if args.embedding_command == "status":
        model_name = args.model or store.active_embedding_model() or DEFAULT_EMBEDDER_MODEL
        payload = _status_payload(store, model_name)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(_format_status(payload))
        return 0

    if args.embedding_command == "backfill":
        model_name = args.model or store.active_embedding_model() or DEFAULT_EMBEDDER_MODEL
        before = _status_payload(store, model_name)
        embedder = _embedder_for_backfill(model_name, test_keyword=args.test_keyword_embedder)
        report = store.embed_missing(embedder)
        after = _status_payload(store, model_name)
        print(
            json.dumps(
                {
                    "model": model_name,
                    "model_id": embedder.model_id,
                    "dimension": embedder.dimension,
                    "before": before,
                    "embedded": report["embedded"],
                    "skipped": report["skipped"],
                    "after": after,
                },
                sort_keys=True,
            )
        )
        return 0

    if args.embedding_command == "index-status":
        model_name = args.model or store.active_embedding_model() or DEFAULT_EMBEDDER_MODEL
        payload = _index_status_payload(store, model_name, index_db=args.index_db)
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(_format_index_status(payload))
        return 0

    if args.embedding_command == "index-rebuild":
        model_name = args.model or store.active_embedding_model() or DEFAULT_EMBEDDER_MODEL
        spec = get_model_spec(model_name)
        embedder = _SpecKeywordEmbedder(
            name=spec.name, model_id_value=spec.model_id, dimension_value=spec.dimension
        )
        index = SQLiteVecVectorIndex(
            db_path=args.index_db or _default_vector_index_db(store.db_path, embedder.model_id),
            model_id=embedder.model_id,
            dimension=embedder.dimension,
        )
        started = time.perf_counter()
        report = store.rebuild_vector_index(embedder, index)
        elapsed = time.perf_counter() - started
        payload = {
            "backend": "sqlite-vec",
            "model": model_name,
            "model_id": embedder.model_id,
            "dimension": embedder.dimension,
            "index_db": str(index.db_path),
            "indexed": report["indexed"],
            "skipped": report["skipped"],
            "rebuild_seconds": round(elapsed, 6),
        }
        print(json.dumps(payload, sort_keys=True))
        return 0

    if args.embedding_command == "benchmark":
        models = [model.strip() for model in args.models.split(",") if model.strip()]
        payload = _run_embedding_benchmark(
            models=models,
            test_keyword=args.test_keyword_embedder,
            repeat=max(1, args.repeat),
            synthetic_count=max(0, args.synthetic_count),
            include_adversarial=args.include_adversarial,
            vector_candidate_limit=args.vector_candidate_limit if args.vector_candidate_limit > 0 else None,
            ann_candidate_limit=max(0, args.ann_candidate_limit),
            ann_backend=args.ann_backend,
            recall_policy=args.recall_policy,
            ann_min_keyword_candidates=max(0, args.ann_min_keyword_candidates),
        )
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(_format_benchmark(payload))
        return 0

    raise AssertionError(f"unknown embeddings command {args.embedding_command!r}")


def _status_payload(store: Anamnesis, model_name: str) -> dict[str, bool | int | str]:
    spec = get_model_spec(model_name)
    embedder = _SpecKeywordEmbedder(
        name=spec.name, model_id_value=spec.model_id, dimension_value=spec.dimension
    )
    status = store.embedding_status(embedder)
    missing = int(status["missing"])
    return {
        "active_model": store.active_embedding_model() or DEFAULT_EMBEDDER_MODEL,
        "inspected_model": spec.name,
        "model_id": spec.model_id,
        "dimension": spec.dimension,
        "total_active": int(status["total_active"]),
        "embedded": int(status["embedded"]),
        "missing": missing,
        "backfill_required": missing > 0,
        "fts_fallback": missing > 0,
    }


def _format_status(payload: dict[str, bool | int | str]) -> str:
    return "\n".join(
        [
            f"active_model={payload['active_model']}",
            f"inspected_model={payload['inspected_model']}",
            f"model_id={payload['model_id']}",
            f"dimension={payload['dimension']}",
            f"semantic_index={payload['embedded']}/{payload['total_active']} ready",
            f"missing={payload['missing']}",
            f"backfill_required={str(payload['backfill_required']).lower()}",
            f"fts_fallback={str(payload['fts_fallback']).lower()}",
        ]
    )


def _index_status_payload(
    store: Anamnesis, model_name: str, *, index_db: Path | None = None
) -> dict[str, bool | int | str]:
    spec = get_model_spec(model_name)
    embedder = _SpecKeywordEmbedder(
        name=spec.name, model_id_value=spec.model_id, dimension_value=spec.dimension
    )
    embedding_status = store.embedding_status(embedder)
    index = SQLiteVecVectorIndex(
        db_path=index_db or _default_vector_index_db(store.db_path, spec.model_id),
        model_id=spec.model_id,
        dimension=spec.dimension,
    )
    embedded = int(embedding_status["embedded"])
    indexed = index.count()
    missing = max(0, embedded - indexed)
    canonical_metadata = store.vector_index_metadata_fingerprints(embedder)
    indexed_metadata = index.metadata_fingerprints()
    stale = sum(
        1
        for rid, indexed_fingerprint in indexed_metadata.items()
        if canonical_metadata.get(rid) != indexed_fingerprint
    )
    return {
        "backend": "sqlite-vec",
        "active_model": store.active_embedding_model() or DEFAULT_EMBEDDER_MODEL,
        "inspected_model": spec.name,
        "model_id": spec.model_id,
        "dimension": spec.dimension,
        "index_db": str(index.db_path),
        "embedded": embedded,
        "indexed": indexed,
        "missing": missing,
        "stale": stale,
        "rebuild_required": missing > 0 or stale > 0,
    }


def _format_index_status(payload: dict[str, bool | int | str]) -> str:
    return "\n".join(
        [
            f"backend={payload['backend']}",
            f"active_model={payload['active_model']}",
            f"inspected_model={payload['inspected_model']}",
            f"model_id={payload['model_id']}",
            f"dimension={payload['dimension']}",
            f"index_db={payload['index_db']}",
            f"vector_index={payload['indexed']}/{payload['embedded']} ready",
            f"missing={payload['missing']}",
            f"stale={payload['stale']}",
            f"rebuild_required={str(payload['rebuild_required']).lower()}",
        ]
    )


def _default_vector_index_db(canonical_db: Path, model_id: str) -> Path:
    safe_model_id = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in model_id)
    return canonical_db.with_name(f"{canonical_db.stem}.vectors.{safe_model_id}.db")


def _run_embedding_benchmark(
    *,
    models: list[str],
    test_keyword: bool,
    repeat: int,
    synthetic_count: int,
    include_adversarial: bool,
    vector_candidate_limit: int | None,
    ann_candidate_limit: int,
    ann_backend: str,
    recall_policy: str,
    ann_min_keyword_candidates: int,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for model_name in models:
        spec = get_model_spec(model_name)
        embedder = _embedder_for_backfill(model_name, test_keyword=test_keyword)
        with tempfile.TemporaryDirectory(prefix="anamnesis-embedding-benchmark-") as tmp:
            store = Anamnesis(Path(tmp) / "benchmark.db")
            store.set_active_embedding_model(model_name)
            cases = _seed_embedding_benchmark_fixture(
                store,
                synthetic_count=synthetic_count,
                include_adversarial=include_adversarial,
            )
            before = store.embedding_status(embedder)
            start = time.perf_counter()
            backfill_report = store.embed_missing(embedder)
            backfill_seconds = time.perf_counter() - start
            after = store.embedding_status(embedder)
            vector_index = None
            ann_rebuild_seconds = 0.0
            ann_indexed = 0
            if ann_candidate_limit > 0:
                vector_index = _create_vector_index(
                    backend=ann_backend,
                    db_path=Path(tmp) / f"{model_name.replace('/', '_')}.sqlite-vec.db",
                    model_id=embedder.model_id,
                    dimension=embedder.dimension,
                )
                ann_started = time.perf_counter()
                ann_report = store.rebuild_vector_index(embedder, vector_index)
                ann_rebuild_seconds = time.perf_counter() - ann_started
                ann_indexed = int(ann_report["indexed"])
            latencies_ms: list[float] = []
            passed = 0
            total = 0
            for _ in range(repeat):
                for case in cases:
                    total += 1
                    started = time.perf_counter()
                    recalled = store.recall(
                        cast(str, case["query"]),
                        owner="benchmark",
                        platform="cli",
                        allowed_visibility={"private"},
                        limit=5,
                        embedder=embedder,
                        vector_candidate_limit=vector_candidate_limit,
                        vector_index=vector_index,
                        ann_candidate_limit=ann_candidate_limit,
                        recall_policy=recall_policy,
                        ann_min_keyword_candidates=ann_min_keyword_candidates,
                    )
                    latencies_ms.append((time.perf_counter() - started) * 1000)
                    recalled_rids = {result.record.rid for result in recalled}
                    expected_rids = set(cast(list[str], case.get("expected_rids", [])))
                    forbidden_rids = set(cast(list[str], case.get("forbidden_rids", [])))
                    expected_ok = expected_rids.issubset(recalled_rids)
                    forbidden_ok = recalled_rids.isdisjoint(forbidden_rids)
                    if expected_ok and forbidden_ok:
                        passed += 1
            score = passed / total if total else 0.0
            db_path = store.db_path
            db_size_bytes = db_path.stat().st_size if db_path.exists() else 0
            backfill_rate = (
                backfill_report["embedded"] / backfill_seconds if backfill_seconds > 0 else 0.0
            )
            results.append(
                {
                    "model": model_name,
                    "model_id": spec.model_id,
                    "dimension": spec.dimension,
                    "total_active": int(after["total_active"]),
                    "embedded": int(after["embedded"]),
                    "missing_before": int(before["missing"]),
                    "missing_after": int(after["missing"]),
                    "backfill_seconds": round(backfill_seconds, 6),
                    "backfill_memories_per_second": round(backfill_rate, 3),
                    "db_size_bytes": db_size_bytes,
                    "embedded_during_backfill": backfill_report["embedded"],
                    "skipped_during_backfill": backfill_report["skipped"],
                    "ann_rebuild_seconds": round(ann_rebuild_seconds, 6),
                    "ann_indexed": ann_indexed,
                    "recall_p50_ms": round(_percentile(latencies_ms, 50), 6),
                    "recall_p95_ms": round(_percentile(latencies_ms, 95), 6),
                    "score": round(score, 4),
                    "passed": passed,
                    "total": total,
                }
            )
    return {
        "cascade": False,
        "models": models,
        "repeat": repeat,
        "synthetic_count": synthetic_count,
        "include_adversarial": include_adversarial,
        "vector_candidate_limit": vector_candidate_limit,
        "ann_candidate_limit": ann_candidate_limit,
        "ann_backend": ann_backend if ann_candidate_limit > 0 else None,
        "recall_policy": recall_policy,
        "ann_min_keyword_candidates": ann_min_keyword_candidates,
        "results": results,
    }


def _create_vector_index(
    *, backend: str, db_path: Path, model_id: str, dimension: int
) -> VectorIndex:
    if backend == "exact":
        return ExactVectorIndex(model_id=model_id, dimension=dimension)
    if backend == "sqlite-vec":
        if not sqlite_vec_available():
            raise RuntimeError("sqlite-vec backend requested but sqlite-vec is not installed")
        return SQLiteVecVectorIndex(db_path=db_path, model_id=model_id, dimension=dimension)
    raise ValueError(f"Unsupported ANN backend: {backend}")


def _seed_embedding_benchmark_fixture(
    store: Anamnesis, *, synthetic_count: int, include_adversarial: bool
) -> list[dict[str, object]]:
    memories = [
        ("local", "Anamnesis keeps local private memory on the user machine."),
        ("devices", "Delegate users can ask questions but cannot control smart-home devices."),
        ("car", "Collaborator handles car washing and pool maintenance."),
        ("model", "Embedding model changes rebuild a search cache instead of migrating memory."),
    ]
    cases: list[dict[str, object]] = []
    for key, text in memories:
        record = store.add_memory(
            text,
            owner="benchmark",
            visibility="private",
            platform_scope="cli",
            domain="benchmark",
            source="embedding-benchmark",
        )
        cases.append({"query": key, "expected_rids": [record.rid], "forbidden_rids": []})

    if synthetic_count:
        _bulk_seed_synthetic_memories(store, synthetic_count)

    if include_adversarial:
        private_other = store.add_memory(
            "Other owner private memory mentions local private machine details.",
            owner="other-owner",
            visibility="private",
            platform_scope="cli",
            domain="benchmark",
            source="embedding-benchmark",
        )
        invalidated = store.add_memory(
            "Benchmark invalidated memory about smart-home device control.",
            owner="benchmark",
            visibility="private",
            platform_scope="cli",
            domain="benchmark",
            source="embedding-benchmark",
        )
        store.invalidate(invalidated.rid, reason="embedding benchmark adversarial fixture")
        cases.extend(
            [
                {
                    "query": "local private machine details",
                    "expected_rids": [],
                    "forbidden_rids": [private_other.rid],
                },
                {
                    "query": "invalidated smart-home device control",
                    "expected_rids": [],
                    "forbidden_rids": [invalidated.rid],
                },
            ]
        )
    return cases


def _bulk_seed_synthetic_memories(store: Anamnesis, count: int) -> None:
    """Seed large benchmark distractor sets in one transaction.

    The public add_memory path opens one SQLite connection per row, which is fine
    for product use but makes 75K+ benchmark fixture seeding hit SQLite file
    churn on macOS temp dirs. Benchmark distractors are deterministic fixtures,
    so insert them as one transaction while preserving FTS/audit invariants.
    """
    now = time.time()
    memory_rows = []
    fts_rows = []
    audit_rows = []
    for idx in range(count):
        rid = str(uuid.uuid4())
        text = f"Synthetic distractor memory {idx} about topic-{idx % 17} and filler-{idx % 31}."
        memory_rows.append(
            (
                rid,
                text,
                "semantic",
                "benchmark",
                "private",
                "cli",
                "read_only",
                "synthetic",
                "embedding-benchmark",
                0.5,
                1.0,
                "active",
                now,
                now,
                None,
                None,
                "{}",
            )
        )
        fts_rows.append((rid, text, "synthetic", "embedding-benchmark", "benchmark"))
        audit_rows.append((rid, "memory_added", "", now, "{}"))
    with store._connect() as conn:  # noqa: SLF001 - benchmark fixture fast path
        conn.executemany(
            """
            INSERT INTO memories (
                rid,text,kind,owner,visibility,platform_scope,action_scope,domain,source,
                importance,confidence,status,created_at,updated_at,last_access,ttl_days,metadata_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            memory_rows,
        )
        conn.executemany(
            "INSERT INTO memory_fts (rid,text,domain,source,owner) VALUES (?,?,?,?,?)",
            fts_rows,
        )
        conn.executemany(
            "INSERT INTO audit_log (rid,event_type,reason,created_at,metadata_json) VALUES (?,?,?,?,?)",
            audit_rows,
        )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    if percentile == 50:
        return float(statistics.median(values))
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100)
    return float(ordered[index])


def _format_benchmark(payload: dict[str, object]) -> str:
    lines = ["cascade=false"]
    for result in payload["results"]:  # type: ignore[index]
        row = result  # type: ignore[assignment]
        lines.append(
            " ".join(
                [
                    f"model={row['model']}",
                    f"dimension={row['dimension']}",
                    f"embedded={row['embedded']}",
                    f"missing_after={row['missing_after']}",
                    f"backfill_seconds={row['backfill_seconds']}",
                    f"backfill_memories_per_second={row['backfill_memories_per_second']}",
                    f"db_size_bytes={row['db_size_bytes']}",
                    f"recall_p50_ms={row['recall_p50_ms']}",
                    f"recall_p95_ms={row['recall_p95_ms']}",
                    f"score={row['score']}",
                ]
            )
        )
    return "\n".join(lines)


def _normalized_synthesis_config(
    store: Anamnesis, raw_override: dict[str, str] | None = None
) -> dict[str, object]:
    raw = raw_override or store.synthesis_config()
    return {
        "base_url": raw.get("base_url", ""),
        "model": raw.get("model", ""),
        "api_key_env": raw.get("api_key_env", ""),
        "temperature": float(raw.get("temperature", "0")),
        "max_tokens": int(raw.get("max_tokens", "512")),
        "timeout": int(raw.get("timeout", "60")),
        "max_context_chars": int(raw.get("max_context_chars", "8000")),
        "max_memory_chars": int(raw.get("max_memory_chars", "1200")),
    }


def _local_llm_config_from_args(args: argparse.Namespace, store: Anamnesis) -> LocalLLMConfig:
    raw = _normalized_synthesis_config(store)
    base_url = args.llm_base_url or str(raw["base_url"])
    model = args.llm_model or str(raw["model"])
    if not base_url or not model:
        raise SystemExit(
            "local LLM synthesis is not configured; run `anamnesis synthesis-config set --base-url ... --model ...`"
        )
    api_key_env = args.api_key_env if args.api_key_env is not None else str(raw["api_key_env"] or "")
    return LocalLLMConfig(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env or None,
        temperature=float(args.temperature if args.temperature is not None else str(raw["temperature"])),
        max_tokens=int(args.max_tokens if args.max_tokens is not None else str(raw["max_tokens"])),
        timeout=int(str(raw["timeout"])),
        max_context_chars=int(
            args.max_context_chars if args.max_context_chars is not None else str(raw["max_context_chars"])
        ),
        max_memory_chars=int(
            args.max_memory_chars if args.max_memory_chars is not None else str(raw["max_memory_chars"])
        ),
    )


def _format_synthesis_config(config: dict[str, object]) -> str:
    return "\n".join(
        [
            f"base_url={config['base_url']}",
            f"model={config['model']}",
            f"api_key_env={config['api_key_env'] or '(none)'}",
            f"temperature={config['temperature']}",
            f"max_tokens={config['max_tokens']}",
            f"timeout={config['timeout']}",
            f"max_context_chars={config['max_context_chars']}",
            f"max_memory_chars={config['max_memory_chars']}",
        ]
    )


def _embedder_for_backfill(model_name: str, *, test_keyword: bool):
    spec = get_model_spec(model_name)
    if test_keyword:
        return _SpecKeywordEmbedder(
            name=spec.name, model_id_value=spec.model_id, dimension_value=spec.dimension
        )
    embedder = load_embedder_by_name(model_name)
    if embedder is None:
        raise RuntimeError(f"embedding model {model_name!r} resolved to no embedder")
    return embedder


if __name__ == "__main__":
    raise SystemExit(main())

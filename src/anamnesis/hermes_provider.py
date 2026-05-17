from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .core import Anamnesis, MemoryInboxItem, MemoryRecord, RecallResult
from .intake import classify_intake, is_platform_local_text

try:  # Imported inside Hermes runtime.
    from agent.memory_provider import MemoryProvider
    from tools.registry import tool_error
except ImportError:  # Allows package tests/imports outside Hermes.

    class MemoryProvider:  # type: ignore[no-redef]
        pass

    def tool_error(message: str) -> str:  # type: ignore[no-redef]
        return json.dumps({"error": message})


REMEMBER_SCHEMA = {
    "name": "anamnesis_remember",
    "description": (
        "Store a durable memory in Anamnesis. Use for stable facts, preferences, "
        "decisions, people, project context, or infrastructure details. Do not use "
        "for temporary task state, commit IDs, PR numbers, process IDs, or current-session TODOs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The durable memory text to store.",
            },
            "importance": {
                "type": "number",
                "description": "0.0-1.0 importance. Default 0.6.",
            },
            "domain": {
                "type": "string",
                "description": "Optional domain tag such as preference, privacy, project, infrastructure.",
            },
            "visibility": {
                "type": "string",
                "description": "Visibility scope. Default private.",
            },
            "platform_scope": {
                "type": "string",
                "description": "Comma-separated recall scope, current, shared, or all. Default all; source/provenance is separate.",
            },
            "source": {
                "type": "string",
                "description": "Source/provenance label. Default hermes.",
            },
            "policy": {
                "type": "string",
                "description": "Write policy: autopilot (default), force_accept, or inbox.",
            },
        },
        "required": ["text"],
    },
}

RECALL_SCHEMA = {
    "name": "anamnesis_recall",
    "description": (
        "Recall memories from Anamnesis with governance filters: owner scope, platform scope, "
        "visibility, invalidated-memory suppression, and operational-junk suppression. Use before answering questions "
        "about remembered user facts, preferences, permissions, projects, or past decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language recall query.",
            },
            "limit": {"type": "integer", "description": "Max results. Default 10."},
            "domain": {
                "type": "string",
                "description": "Optional exact domain filter.",
            },
            "visibility": {
                "type": "string",
                "description": "Optional visibility filter. Default private.",
            },
        },
        "required": ["query"],
    },
}

FORGET_SCHEMA = {
    "name": "anamnesis_forget",
    "description": "Invalidate a memory by rid so it no longer appears in recall while remaining auditable.",
    "parameters": {
        "type": "object",
        "properties": {
            "rid": {"type": "string", "description": "Memory rid to invalidate."},
            "reason": {"type": "string", "description": "Optional reason."},
        },
        "required": ["rid"],
    },
}

CORRECT_SCHEMA = {
    "name": "anamnesis_correct",
    "description": "Correct a memory by invalidating the old rid and creating an audited replacement.",
    "parameters": {
        "type": "object",
        "properties": {
            "rid": {"type": "string", "description": "Active memory rid to correct."},
            "text": {"type": "string", "description": "Replacement memory text."},
            "reason": {"type": "string", "description": "Optional correction reason."},
        },
        "required": ["rid", "text"],
    },
}

STATS_SCHEMA = {
    "name": "anamnesis_stats",
    "description": "Return Anamnesis memory counts by status plus inbox/conflict counts.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

ALL_TOOL_SCHEMAS = [REMEMBER_SCHEMA, RECALL_SCHEMA, FORGET_SCHEMA, CORRECT_SCHEMA, STATS_SCHEMA]


class AnamnesisMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider adapter for Anamnesis."""

    def __init__(self) -> None:
        self._store: Anamnesis | None = None
        self._db_path = ""
        self._owner = "default"
        self._platform = "cli"
        self._visibility = "private"
        self._session_id = ""
        self._prefetch_results: dict[str, str] = {}
        self._init_error: BaseException | None = None

    @property
    def name(self) -> str:
        return "anamnesis"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id
        self._platform = str(
            kwargs.get("platform") or os.environ.get("ANAMNESIS_PLATFORM") or "cli"
        )
        hermes_home = Path(str(kwargs.get("hermes_home") or Path.home() / ".hermes"))
        default_db = hermes_home / "anamnesis" / "anamnesis.db"
        self._db_path = str(os.environ.get("ANAMNESIS_DB_PATH") or default_db)
        self._owner = self._resolve_owner(kwargs)
        # Benchmark/imported DBs usually use owner=default. Keep that as an easy env override.
        self._visibility = str(os.environ.get("ANAMNESIS_VISIBILITY") or "private")
        try:
            self._store = Anamnesis(self._db_path)
            self._init_error = None
        except (
            BaseException
        ) as exc:  # surface via tools/system prompt instead of crashing Hermes.
            self._store = None
            self._init_error = exc


    def _resolve_owner(self, kwargs: dict[str, Any]) -> str:
        """Resolve row-level owner namespace from explicit config or caller identity.

        The database is profile-level; owner is an in-DB namespace. Prefer the
        actual caller/chat identity over agent identity so one shared profile can
        still keep user memories separated when the gateway supplies identities.
        """
        explicit = str(os.environ.get("ANAMNESIS_OWNER") or "").strip()
        if explicit:
            return explicit
        for key in ("memory_owner", "owner"):
            value = str(kwargs.get(key) or "").strip()
            if value:
                return self._canonical_owner(value)
        for key in ("user_id", "chat_id", "sender_id"):
            value = str(kwargs.get(key) or "").strip()
            if value:
                return self._canonical_owner(self._owner_namespace(value))
        agent_identity = str(kwargs.get("agent_identity") or "").strip()
        if agent_identity:
            return self._canonical_owner(self._owner_namespace(agent_identity))
        return "default"


    def _canonical_owner(self, owner: str) -> str:
        aliases = self._owner_aliases()
        return aliases.get(owner, owner)

    @staticmethod
    def _owner_aliases() -> dict[str, str]:
        raw = str(os.environ.get("ANAMNESIS_OWNER_ALIASES") or "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("ANAMNESIS_OWNER_ALIASES must be a JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError("ANAMNESIS_OWNER_ALIASES must be a JSON object")
        return {str(key).strip(): str(value).strip() for key, value in parsed.items() if str(key).strip() and str(value).strip()}

    def _owner_namespace(self, raw: str) -> str:
        if ":" in raw:
            return raw
        platform = (self._platform or "").strip()
        if platform and platform != "cli":
            return f"{platform}:{raw}"
        return raw

    def system_prompt_block(self) -> str:
        if self._init_error is not None:
            return f"Anamnesis memory provider failed to initialize: {type(self._init_error).__name__}: {self._safe_error(self._init_error)}"
        return (
            "Anamnesis memory is active. Use anamnesis_recall before making claims about remembered "
            "user facts, preferences, permissions, projects, or past decisions. Store only durable memories; "
            "skip temporary task state."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        cache_key = session_id or self._session_id
        if cache_key and cache_key in self._prefetch_results:
            return self._prefetch_results.pop(cache_key)
        return self._build_recall_block(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        cache_key = session_id or self._session_id
        if not cache_key:
            return
        self._prefetch_results[cache_key] = self._build_recall_block(query)

    def sync_turn(
        self, user_content: str, assistant_content: str = "", *, session_id: str = ""
    ) -> None:
        if self._store is None:
            return
        text = " ".join((user_content or "").split())
        if not text:
            return
        self._apply_memory_policy(
            text,
            domain="",
            visibility=self._visibility,
            platform_scope=self._sync_turn_platform_scope(text),
            source="hermes_sync_turn",
            importance=0.5,
            metadata={
                "session_id": session_id or self._session_id,
                "source_platform": self._platform,
            },
            policy="autopilot",
        )

    def _sync_turn_platform_scope(self, text: str) -> str:
        decision = classify_intake(text, domain="")
        if decision.lifecycle == "sensitive" or is_platform_local_text(text):
            return self._platform
        return "all"

    def _build_recall_block(self, query: str) -> str:
        if self._store is None or not query.strip():
            return ""
        try:
            results = self._store.recall(
                query,
                owner=self._owner,
                platform=self._platform,
                allowed_visibility={self._visibility},
                limit=8,
            )
        except Exception:
            return ""
        filtered = [result for result in results if self._should_inject_result(result)]
        if not filtered:
            return ""
        lines = ["## Anamnesis Recall"]
        for idx, result in enumerate(filtered[:5], start=1):
            lines.append(f"{idx}. {result.record.text}")
        return "\n".join(lines)

    def _should_inject_result(self, result: RecallResult) -> bool:
        decision = classify_intake(result.record.text, domain=result.record.domain)
        return decision.action == "accept"

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return ALL_TOOL_SCHEMAS

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        if self._store is None:
            return tool_error(
                f"Anamnesis not initialized: {self._safe_error(self._init_error)}"
            )
        try:
            if tool_name == "anamnesis_remember":
                return self._remember(args)
            if tool_name == "anamnesis_recall":
                return self._recall(args)
            if tool_name == "anamnesis_forget":
                rid = str(args.get("rid") or "").strip()
                if not rid:
                    return tool_error("rid is required")
                self._store.invalidate(rid, reason=str(args.get("reason") or ""))
                return json.dumps({"success": True, "rid": rid})
            if tool_name == "anamnesis_correct":
                rid = str(args.get("rid") or "").strip()
                text = str(args.get("text") or "").strip()
                if not rid:
                    return tool_error("rid is required")
                if not text:
                    return tool_error("text is required")
                replacement = self._store.correct_memory(
                    rid,
                    text,
                    reason=str(args.get("reason") or ""),
                    metadata={"source_platform": self._platform},
                )
                old = self._store.get_memory(rid)
                return json.dumps(
                    {
                        "success": True,
                        "old": self._record_dict(old),
                        "replacement": self._record_dict(replacement),
                    },
                    ensure_ascii=False,
                )
            if tool_name == "anamnesis_stats":
                return self._stats()
            return tool_error(f"Unknown Anamnesis tool: {tool_name}")
        except Exception as exc:
            return tool_error(f"Anamnesis {tool_name} failed: {self._safe_error(exc)}")

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if (
            self._store is None
            or action not in {"add", "replace"}
            or not content.strip()
        ):
            return
        text = " ".join(content.split())
        enriched_metadata = {
            **(metadata or {}),
            "source_platform": self._platform,
        }
        self._apply_memory_policy(
            text,
            domain=target,
            visibility=self._visibility,
            platform_scope=self._sync_turn_platform_scope(text),
            source="hermes_memory_tool",
            importance=0.6,
            metadata=enriched_metadata,
            policy="autopilot",
        )


    def _normalize_platform_scope(self, scope: str) -> str:
        normalized = ",".join(part.strip() for part in scope.split(",") if part.strip())
        key = normalized.lower()
        if key in {"", "all", "shared", "global", "cross-platform", "cross_platform"}:
            return "all"
        if key in {"current", "this", "this-platform", "this_platform"}:
            return self._platform
        return normalized

    def _remember(self, args: dict[str, Any]) -> str:
        assert self._store is not None
        text = str(args.get("text") or "").strip()
        if not text:
            return tool_error("text is required")
        return self._apply_memory_policy(
            text,
            domain=str(args.get("domain") or ""),
            visibility=str(args.get("visibility") or self._visibility),
            platform_scope=self._normalize_platform_scope(str(args.get("platform_scope") or "all")),
            source=str(args.get("source") or "hermes"),
            importance=float(
                args.get("importance") if args.get("importance") is not None else 0.6
            ),
            metadata={},
            policy=str(args.get("policy") or "autopilot"),
        )


    def _apply_memory_policy(
        self,
        text: str,
        *,
        domain: str,
        visibility: str,
        platform_scope: str,
        source: str,
        importance: float,
        metadata: dict[str, Any],
        policy: str = "autopilot",
    ) -> str:
        assert self._store is not None
        policy_key = (policy or "autopilot").strip().lower()
        decision = classify_intake(text, domain=domain)
        enriched_metadata = {
            **metadata,
            "intake_reasons": decision.reasons,
            "intake_lifecycle": decision.lifecycle,
            "intake_action": decision.action,
            "policy": policy_key,
        }

        if policy_key == "force_accept" or (policy_key == "autopilot" and decision.action == "accept"):
            record = self._store.add_memory(
                text,
                owner=self._owner,
                visibility=visibility,
                platform_scope=platform_scope,
                domain=domain or decision.lifecycle,
                source=source,
                importance=importance,
                confidence=decision.confidence,
                metadata=enriched_metadata,
            )
            return json.dumps(
                {
                    "success": True,
                    "action": "accepted",
                    "intake": self._intake_dict(decision),
                    "memory": self._record_dict(record),
                },
                ensure_ascii=False,
            )

        if policy_key == "inbox" or (policy_key == "autopilot" and decision.action == "inbox"):
            item = self._store.propose_memory(
                text,
                source_snippet=text[:500],
                proposed_kind="semantic",
                owner=self._owner,
                visibility=visibility,
                platform_scope=platform_scope,
                action_scope="all",
                domain=domain or decision.lifecycle,
                source=source,
                confidence=decision.confidence,
                why_save=", ".join(decision.reasons),
                suggested_lifecycle=decision.lifecycle,
            )
            return json.dumps(
                {
                    "success": True,
                    "action": "inboxed",
                    "intake": self._intake_dict(decision),
                    "inbox_item": self._inbox_item_dict(item),
                },
                ensure_ascii=False,
            )

        if policy_key not in {"autopilot", "force_accept", "inbox"}:
            raise ValueError("policy must be autopilot, force_accept, or inbox")

        return json.dumps(
            {
                "success": False,
                "action": "rejected",
                "intake": self._intake_dict(decision),
            },
            ensure_ascii=False,
        )

    def _recall(self, args: dict[str, Any]) -> str:
        assert self._store is not None
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        limit = int(args.get("limit") or 10)
        visibility = str(args.get("visibility") or self._visibility)
        results = self._store.recall(
            query,
            owner=self._owner,
            platform=self._platform,
            allowed_visibility={visibility},
            limit=max(1, min(50, limit)),
            domain=str(args.get("domain") or "") or None,
        )
        return json.dumps(
            {"results": [self._result_dict(r) for r in results]}, ensure_ascii=False
        )

    @staticmethod
    def _increment_counter(counter: dict[str, int], key: str) -> None:
        normalized = key.strip() or "unknown"
        counter[normalized] = counter.get(normalized, 0) + 1

    def _stats(self) -> str:
        assert self._store is not None
        with self._store._connect() as conn:  # noqa: SLF001 - provider adapter stats wrapper.
            statuses = {
                row["status"]: row["n"]
                for row in conn.execute(
                    "SELECT status, count(*) AS n FROM memories GROUP BY status"
                )
            }
            inbox = conn.execute(
                "SELECT decision, count(*) AS n FROM memory_inbox GROUP BY decision"
            ).fetchall()
            conflicts = conn.execute(
                "SELECT status, count(*) AS n FROM contradictions GROUP BY status"
            ).fetchall()
            rows = conn.execute(
                """
                SELECT owner, platform_scope, source, metadata_json
                FROM memories
                WHERE status='active'
                """
            ).fetchall()
        owners: dict[str, int] = {}
        platform_scopes: dict[str, int] = {}
        sources: dict[str, int] = {}
        source_platforms: dict[str, int] = {}
        for row in rows:
            self._increment_counter(owners, str(row["owner"] or "unknown"))
            self._increment_counter(sources, str(row["source"] or "unknown"))
            for scope in str(row["platform_scope"] or "all").split(","):
                self._increment_counter(platform_scopes, scope)
            metadata = json.loads(row["metadata_json"] or "{}")
            self._increment_counter(
                source_platforms, str(metadata.get("source_platform") or "unknown")
            )
        return json.dumps(
            {
                "db_path": self._db_path,
                "owner": self._owner,
                "platform": self._platform,
                "memories": statuses,
                "inbox": {row["decision"]: row["n"] for row in inbox},
                "contradictions": {row["status"]: row["n"] for row in conflicts},
                "owners": owners,
                "platform_scopes": platform_scopes,
                "sources": sources,
                "source_platforms": source_platforms,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _record_dict(record: MemoryRecord) -> dict[str, Any]:
        return {
            "rid": record.rid,
            "text": record.text,
            "kind": record.kind,
            "owner": record.owner,
            "visibility": record.visibility,
            "platform_scope": record.platform_scope,
            "domain": record.domain,
            "source": record.source,
            "importance": record.importance,
            "confidence": record.confidence,
            "status": record.status,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "metadata": record.metadata,
        }

    @staticmethod
    def _inbox_item_dict(item: MemoryInboxItem) -> dict[str, Any]:
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

    @staticmethod
    def _intake_dict(decision: Any) -> dict[str, Any]:
        return {
            "action": decision.action,
            "lifecycle": decision.lifecycle,
            "reasons": decision.reasons,
            "confidence": decision.confidence,
        }

    @classmethod
    def _result_dict(cls, result: RecallResult) -> dict[str, Any]:
        data = cls._record_dict(result.record)
        data["score"] = result.score
        data["reasons"] = result.reasons
        return data

    @staticmethod
    def _safe_error(exc: BaseException | None) -> str:
        if exc is None:
            return "not initialized"
        return " ".join(str(exc).split())[:240]


def register(ctx: Any) -> None:
    ctx.register_memory_provider(AnamnesisMemoryProvider())

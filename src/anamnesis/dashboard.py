from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .core import Anamnesis, MemoryInboxItem, MemoryRecord
from .embedding_models import OFFICIAL_EMBEDDING_MODELS, get_model_spec, load_embedder_by_name
from .embeddings import normalize
from .intake import classify_intake, is_platform_local_text

JsonDict = dict[str, Any]


class DashboardAPI:
    """Small JSON API for the local Anamnesis dashboard.

    The class is deliberately framework-free so the dashboard backend stays local,
    dependency-light, and easy to test. `handle()` is the pure API surface used by
    both tests and the stdlib HTTP server wrapper below.
    """

    def __init__(self, store: Anamnesis):
        self.store = store

    def handle(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        parsed = urlparse(path)
        route = parsed.path.rstrip("/") or "/"
        query = {key: values[-1] for key, values in parse_qs(parsed.query).items() if values}
        headers = headers or {}
        try:
            if self._requires_auth(route) and not self._is_authenticated(headers):
                return self._json(401, {"error": "unauthorized", "message": "password required"})
            if method == "GET" and route == "/":
                return self._asset("dashboard.html", "text/html; charset=utf-8")
            if method == "GET" and route == "/static/dashboard.css":
                return self._asset("dashboard.css", "text/css; charset=utf-8")
            if method == "GET" and route == "/static/dashboard.js":
                return self._asset("dashboard.js", "application/javascript; charset=utf-8")
            if method == "GET" and route == "/api/auth/status":
                return self._json(200, self.auth_status(headers))
            if method == "GET" and route == "/api/overview":
                return self._json(200, self.overview())
            if method == "GET" and route == "/api/facets":
                return self._json(200, self.facets())
            if method == "GET" and route == "/api/memories":
                return self._json(200, self.memories(query))
            if method == "GET" and route == "/api/inbox":
                return self._json(200, self.inbox(query))
            if method == "GET" and route.startswith("/api/audit/"):
                rid = unquote(route.removeprefix("/api/audit/"))
                return self._json(200, self.audit(rid))
            if method == "POST" and route == "/api/preview-turn":
                return self._json(200, self.preview_turn(self._parse_body(body)))
            if method == "POST" and route == "/api/preview-memory-write":
                return self._json(200, self.preview_memory_write(self._parse_body(body)))
            if method == "POST" and route == "/api/inbox/accept":
                return self._json(200, self.accept_inbox(self._parse_body(body)))
            if method == "POST" and route == "/api/inbox/reject":
                return self._json(200, self.reject_inbox(self._parse_body(body)))
            if method == "POST" and route == "/api/inbox/batch":
                return self._json(200, self.batch_inbox(self._parse_body(body)))
            if method == "POST" and route == "/api/memories/batch":
                return self._json(200, self.batch_memories(self._parse_body(body)))
            if method == "POST" and route == "/api/correct":
                return self._json(200, self.correct(self._parse_body(body)))
            if method == "GET" and route == "/api/settings":
                return self._json(200, self.settings())
            if method == "GET" and route == "/api/embedding/status":
                return self._json(200, self.embedding_status())
            if method == "POST" and route == "/api/embedding/backfill":
                return self._json(200, self.embedding_backfill(self._parse_body(body)))
            if method == "POST" and route == "/api/recall/simulate":
                return self._json(200, self.recall_simulate(self._parse_body(body)))
            if method == "GET" and route == "/api/runtime/status":
                return self._json(200, self.runtime_status())
            if method == "POST" and route == "/api/runtime/test-recall":
                return self._json(200, self.runtime_test_recall(self._parse_body(body)))
            if method == "POST" and route == "/api/maintenance/autopilot":
                return self._json(200, self.maintenance_autopilot(self._parse_body(body)))
            if method == "POST" and route == "/api/settings/dashboard-password":
                return self._clear_session_response(self.set_dashboard_password(self._parse_body(body)))
            if method == "POST" and route == "/api/settings/dashboard-password/clear":
                return self._clear_session_response(self.clear_dashboard_password())
            if method == "POST" and route == "/api/settings/embedding":
                return self._json(200, self.set_embedding_model(self._parse_body(body)))
            if method == "POST" and route == "/api/settings/synthesis":
                return self._json(200, self.set_synthesis(self._parse_body(body)))
            if method == "POST" and route == "/api/auth/login":
                return self._login_response(self._parse_body(body))
        except KeyError as exc:
            return self._json(404, {"error": "not_found", "message": str(exc.args[0])})
        except ValueError as exc:
            return self._json(400, {"error": "bad_request", "message": str(exc)})
        return self._json(404, {"error": "not_found", "message": route})

    def auth_status(self, headers: dict[str, str] | None = None) -> JsonDict:
        password_set = self._password_set()
        return {
            "password_required": password_set,
            "authenticated": (not password_set) or self._is_authenticated(headers or {}),
        }

    def _requires_auth(self, route: str) -> bool:
        if not route.startswith("/api/"):
            return False
        if route in {"/api/auth/login", "/api/auth/status"}:
            return False
        return self._password_set()

    def _password_set(self) -> bool:
        with self.store._connect() as conn:  # noqa: SLF001
            row = conn.execute("SELECT 1 FROM settings WHERE key='dashboard.password_hash'").fetchone()
        return row is not None

    def _is_authenticated(self, headers: dict[str, str]) -> bool:
        cookie_header = headers.get("cookie") or headers.get("Cookie") or ""
        cookie = SimpleCookie(cookie_header)
        token = cookie.get("anamnesis_session")
        if token is None:
            return False
        token_hash = _sha256(token.value)
        with self.store._connect() as conn:  # noqa: SLF001
            row = conn.execute(
                "SELECT value FROM settings WHERE key='dashboard.session_hash'"
            ).fetchone()
            expires_row = conn.execute(
                "SELECT value FROM settings WHERE key='dashboard.session_expires'"
            ).fetchone()
        if row is None or expires_row is None:
            return False
        try:
            expires_at = float(expires_row["value"])
        except (TypeError, ValueError):
            return False
        if expires_at < time.time():
            return False
        return hmac.compare_digest(token_hash, str(row["value"]))

    def facets(self) -> JsonDict:
        with self.store._connect() as conn:  # noqa: SLF001 - dashboard read model.
            memory_owners = _facet_rows(conn, "memories", "owner")
            memory_domains = _facet_rows(conn, "memories", "domain")
            memory_sources = _facet_rows(conn, "memories", "source")
            memory_platforms = _facet_rows(conn, "memories", "platform_scope")
            inbox_owners = _facet_rows(conn, "memory_inbox", "owner")
            inbox_domains = _facet_rows(conn, "memory_inbox", "domain")
            inbox_sources = _facet_rows(conn, "memory_inbox", "source")
            inbox_platforms = _facet_rows(conn, "memory_inbox", "platform_scope")
        return {
            "memories": {"owners": memory_owners, "domains": memory_domains, "sources": memory_sources, "platforms": memory_platforms},
            "inbox": {"owners": inbox_owners, "domains": inbox_domains, "sources": inbox_sources, "platforms": inbox_platforms},
        }

    def overview(self) -> JsonDict:
        with self.store._connect() as conn:  # noqa: SLF001 - dashboard read model.
            memory_counts = {
                row["status"]: int(row["count"])
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS count FROM memories GROUP BY status"
                ).fetchall()
            }
            inbox_counts = {
                row["decision"]: int(row["count"])
                for row in conn.execute(
                    "SELECT decision, COUNT(*) AS count FROM memory_inbox GROUP BY decision"
                ).fetchall()
            }
            recent_memory_rows = conn.execute(
                "SELECT * FROM memories WHERE status='active' ORDER BY updated_at DESC LIMIT 8"
            ).fetchall()
            recent_inbox_rows = conn.execute(
                "SELECT * FROM memory_inbox ORDER BY updated_at DESC LIMIT 8"
            ).fetchall()
            recent_audit_rows = conn.execute(
                "SELECT rid,event_type,reason,created_at,metadata_json FROM audit_log ORDER BY id DESC LIMIT 8"
            ).fetchall()
        return {
            "generated_at": time.time(),
            "counts": {
                "memories": _filled_counts(memory_counts, ["active", "superseded", "invalidated"]),
                "inbox": _filled_counts(
                    inbox_counts, ["pending", "accepted", "rejected", "expired"]
                ),
            },
            "recent_memories": [
                memory_record_dict(self.store._row_to_record(row))  # noqa: SLF001
                for row in recent_memory_rows
            ],
            "recent_inbox": [
                inbox_item_dict(self.store._row_to_inbox_item(row))  # noqa: SLF001
                for row in recent_inbox_rows
            ],
            "recent_audit": [_audit_event_dict(row) for row in recent_audit_rows],
        }

    def memories(self, query: dict[str, str]) -> JsonDict:
        limit = _limit(query.get("limit"), default=50, maximum=200)
        offset = _offset(query.get("offset"))
        clauses: list[str] = []
        params: list[Any] = []
        for field in ("owner", "status", "domain", "source", "visibility", "platform_scope"):
            value = query.get(field)
            if value:
                clauses.append(f"{field}=?")
                params.append(value)
        search = (query.get("q") or "").strip()
        if search:
            clauses.append("text LIKE ?")
            params.append(f"%{search}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.store._connect() as conn:  # noqa: SLF001 - dashboard read model.
            total = int(conn.execute(f"SELECT COUNT(*) FROM memories {where}", tuple(params)).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM memories {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return {
            "items": [memory_record_dict(self.store._row_to_record(row)) for row in rows],  # noqa: SLF001
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_next": offset + limit < total,
            "has_prev": offset > 0,
            "filters": query,
        }

    def inbox(self, query: dict[str, str]) -> JsonDict:
        decision = query.get("decision") or "pending"
        limit = _limit(query.get("limit"), default=50, maximum=200)
        offset = _offset(query.get("offset"))
        clauses = ["decision=?"]
        params: list[Any] = [decision]
        for field in ("owner", "domain", "source", "visibility", "platform_scope"):
            value = query.get(field)
            if value:
                clauses.append(f"{field}=?")
                params.append(value)
        min_confidence = _optional_float(query.get("min_confidence"))
        if min_confidence is not None:
            clauses.append("confidence>=?")
            params.append(min_confidence)
        max_confidence = _optional_float(query.get("max_confidence"))
        if max_confidence is not None:
            clauses.append("confidence<?")
            params.append(max_confidence)
        search = (query.get("q") or "").strip()
        if search:
            clauses.append("(proposed_text LIKE ? OR source_snippet LIKE ? OR why_save LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        where = "WHERE " + " AND ".join(clauses)
        with self.store._connect() as conn:  # noqa: SLF001 - dashboard read model.
            total = int(conn.execute(f"SELECT COUNT(*) FROM memory_inbox {where}", tuple(params)).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM memory_inbox {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return {
            "decision": decision,
            "limit": limit,
            "offset": offset,
            "total": total,
            "has_next": offset + limit < total,
            "has_prev": offset > 0,
            "items": [inbox_item_dict(self.store._row_to_inbox_item(row)) for row in rows],  # noqa: SLF001
        }

    def audit(self, rid: str) -> JsonDict:
        record = self.store.get_memory(rid)
        events = self.store.audit_events(rid)
        chain: dict[str, str] = {}
        for event in events:
            metadata = event.get("metadata", {})
            if event.get("event_type") == "memory_corrected_from" and metadata.get("replacement_rid"):
                chain["replacement_rid"] = str(metadata["replacement_rid"])
            if event.get("event_type") == "memory_corrected_to" and metadata.get("old_rid"):
                chain["old_rid"] = str(metadata["old_rid"])
        if "corrects_rid" in record.metadata:
            chain.setdefault("old_rid", str(record.metadata["corrects_rid"]))
        return {
            "rid": rid,
            "memory": memory_record_dict(record),
            "events": events,
            "correction_chain": chain,
        }

    def preview_turn(self, payload: JsonDict) -> JsonDict:
        text = _required_str(payload, "text")
        owner = str(payload.get("owner") or "default")
        platform = str(payload.get("platform") or "local")
        visibility = _visibility_list(payload.get("visibility"))
        domain = str(payload.get("domain") or "")
        limit = int(payload.get("limit") or 5)
        return {
            "mode": "preview",
            **self._preview_payload(
                text=text,
                owner=owner,
                platform=platform,
                visibility=visibility,
                domain=domain,
                limit=limit,
            ),
        }

    def preview_memory_write(self, payload: JsonDict) -> JsonDict:
        text = _required_str(payload, "text")
        owner = str(payload.get("owner") or "default")
        platform = str(payload.get("platform") or "local")
        target = str(payload.get("target") or payload.get("domain") or "memory")
        origin = str(payload.get("origin") or "")
        visibility = str(payload.get("visibility") or "private")
        apply = bool(payload.get("apply", False))
        preview = self._preview_payload(
            text=text,
            owner=owner,
            platform=platform,
            visibility=[visibility],
            domain=target,
            limit=int(payload.get("limit") or 1),
        )
        preview["mode"] = "preview_memory_write"
        preview["input"].update({"target": target, "origin": origin, "source": "hermes_memory_tool"})
        applied = None
        if apply:
            applied = self._apply_preview_write(
                text=text,
                owner=owner,
                visibility=visibility,
                domain=target,
                payload=preview,
                source="hermes_memory_tool",
                metadata={"origin": origin, "preview_memory_write_applied": True},
            )
        return {**preview, "apply": apply, "applied": applied}

    def correct(self, payload: JsonDict) -> JsonDict:
        rid = _required_str(payload, "rid")
        text = _required_str(payload, "text")
        reason = str(payload.get("reason") or "")
        replacement = self.store.correct_memory(rid, text, reason=reason)
        old = self.store.get_memory(rid)
        return {
            "old": memory_record_dict(old),
            "replacement": memory_record_dict(replacement),
            "reason": reason,
        }

    def settings(self) -> JsonDict:
        recall = self.store.recall_config()
        synthesis = self.store.synthesis_config()
        active_model = self.store.active_embedding_model()
        with self.store._connect() as conn:  # noqa: SLF001
            row = conn.execute("SELECT value FROM settings WHERE key='dashboard.password_hash'").fetchone()
            password_set = row is not None
        return {
            "dashboard_password_set": password_set,
            "embedding": {
                "enabled": _setting_bool(recall.get("embedding_enabled"), default=bool(active_model)),
                "active_model": active_model or "",
                "available_models": [
                    {"name": spec.name, "dimension": spec.dimension}
                    for spec in OFFICIAL_EMBEDDING_MODELS.values()
                ],
            },
            "recall": recall,
            "synthesis": {
                "enabled": _setting_bool(synthesis.get("enabled"), default=bool(synthesis.get("base_url") and synthesis.get("model"))),
                "base_url": synthesis.get("base_url", ""),
                "model": synthesis.get("model", ""),
                "api_key_env": synthesis.get("api_key_env", ""),
                "temperature": float(synthesis.get("temperature", "0.0") or "0.0"),
                "max_tokens": int(synthesis.get("max_tokens", "512") or "512"),
                "timeout": int(synthesis.get("timeout", "60") or "60"),
            },
        }

    def set_dashboard_password(self, payload: JsonDict) -> JsonDict:
        password = _required_str(payload, "password")
        password_hash = _sha256(password)
        now = time.time()
        with self.store._connect() as conn:  # noqa: SLF001
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES ('dashboard.password_hash', ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (password_hash, now),
            )
            conn.execute("DELETE FROM settings WHERE key IN ('dashboard.session_hash', 'dashboard.session_expires')")
        return {"dashboard_password_set": True, "session_cleared": True}

    def clear_dashboard_password(self) -> JsonDict:
        with self.store._connect() as conn:  # noqa: SLF001
            conn.execute(
                "DELETE FROM settings WHERE key IN ('dashboard.password_hash', 'dashboard.session_hash', 'dashboard.session_expires')"
            )
        return {"dashboard_password_set": False, "session_cleared": True}

    def _clear_session_response(self, payload: JsonDict) -> tuple[int, dict[str, str], bytes]:
        return self._json(200, payload, {"set-cookie": _clear_cookie_header()})

    def set_embedding_model(self, payload: JsonDict) -> JsonDict:
        model_name = _required_str(payload, "model")
        enabled = bool(payload.get("enabled", True))
        if model_name not in OFFICIAL_EMBEDDING_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {', '.join(OFFICIAL_EMBEDDING_MODELS.keys())}")
        self.store.set_active_embedding_model(model_name)
        self.store.set_recall_config(embedding_enabled=str(enabled).lower())
        return {"active_model": model_name, "enabled": enabled}

    def set_synthesis(self, payload: JsonDict) -> JsonDict:
        values: dict[str, str | int | float] = {}
        if "enabled" in payload:
            values["enabled"] = str(bool(payload["enabled"])).lower()
        for key in ("base_url", "model", "api_key_env"):
            if key in payload:
                values[key] = str(payload[key])
        for key in ("temperature", "max_tokens", "timeout"):
            if key in payload:
                values[key] = payload[key]
        if not values:
            raise ValueError("No settings provided")
        return self.store.set_synthesis_config(**values)

    def login(self, payload: JsonDict) -> JsonDict:
        password = _required_str(payload, "password")
        with self.store._connect() as conn:  # noqa: SLF001
            row = conn.execute("SELECT value FROM settings WHERE key='dashboard.password_hash'").fetchone()
            if row is None:
                return {"authenticated": True}
            expected = row["value"]
        password_hash = _sha256(password)
        if not hmac.compare_digest(password_hash, str(expected)):
            raise ValueError("Incorrect password")
        return {"authenticated": True}

    def _login_response(self, payload: JsonDict) -> tuple[int, dict[str, str], bytes]:
        result = self.login(payload)
        if not self._password_set():
            return self._json(200, result)
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + 60 * 60 * 24 * 30
        with self.store._connect() as conn:  # noqa: SLF001
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES ('dashboard.session_hash', ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (_sha256(token), time.time()),
            )
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES ('dashboard.session_expires', ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (str(expires_at), time.time()),
            )
        return self._json(
            200,
            result,
            {
                "set-cookie": f"anamnesis_session={token}; Max-Age=2592000; Path=/; HttpOnly; SameSite=Strict"
            },
        )

    def embedding_status(self) -> JsonDict:
        model_name = self.store.active_embedding_model() or "potion-base-2M"
        embedder = _dashboard_embedder(model_name)
        status = self.store.embedding_status(embedder)
        recall = self.store.recall_config()
        status.update(
            {
                "enabled": _setting_bool(recall.get("embedding_enabled"), default=False),
                "active_model": model_name,
                "backfill_required": int(status["missing"]) > 0,
                "fts_fallback": True,
            }
        )
        return status

    def embedding_backfill(self, payload: JsonDict) -> JsonDict:
        model_name = str(payload.get("model") or self.store.active_embedding_model() or "potion-base-2M")
        status_embedder = _dashboard_embedder(model_name)
        before = self.store.embedding_status(status_embedder)
        started = time.time()
        embedder = _dashboard_backfill_embedder(model_name)
        report = self.store.embed_missing(embedder)
        after = self.store.embedding_status(status_embedder)
        payload_out = {
            "model": model_name,
            "before": before,
            "after": after,
            "embedded": report["embedded"],
            "skipped": report["skipped"],
            "seconds": round(time.time() - started, 3),
        }
        self._record_audit("embedding_backfill", payload_out)
        return payload_out

    def recall_simulate(self, payload: JsonDict) -> JsonDict:
        query = _required_str(payload, "query")
        owner = str(payload.get("owner") or "primary")
        platform = str(payload.get("platform") or "whatsapp")
        domain = str(payload.get("domain") or "") or None
        limit = int(payload.get("limit") or 10)
        visibility = _visibility_list(payload.get("visibility") or ["private"])
        return self.store.simulate_recall(
            query,
            owner=owner,
            platform=platform,
            allowed_visibility=set(visibility),
            limit=max(1, min(50, limit)),
            domain=domain,
            sample_limit=200,
        )

    def maintenance_autopilot(self, payload: JsonDict) -> JsonDict:
        apply = bool(payload.get("apply", False))  # noqa: A001 - JSON API field.
        owner = str(payload.get("owner") or "") or None
        domain = str(payload.get("domain") or "") or None
        max_age_days = max(0, int(payload.get("max_inbox_age_days") or 30))
        threshold = float(payload.get("duplicate_threshold") or 0.9)
        cutoff = time.time() - max_age_days * 86400
        clauses = ["decision='pending'", "created_at < ?"]
        params: list[Any] = [cutoff]
        if owner is not None:
            clauses.append("owner=?")
            params.append(owner)
        if domain is not None:
            clauses.append("domain=?")
            params.append(domain)
        where = " AND ".join(clauses)
        with self.store._connect() as conn:  # noqa: SLF001
            pending_stale = int(
                conn.execute(f"SELECT COUNT(*) FROM memory_inbox WHERE {where}", tuple(params)).fetchone()[0]
            )
            stale_examples = [
                inbox_item_dict(self.store._row_to_inbox_item(row))  # noqa: SLF001
                for row in conn.execute(
                    f"SELECT * FROM memory_inbox WHERE {where} ORDER BY created_at ASC LIMIT 10",
                    tuple(params),
                ).fetchall()
            ]
        duplicate_preview = self.store.preview_duplicate_supersession(
            owner=owner,
            domain=domain,
            threshold=threshold,
            example_limit=10,
        )
        if not apply:
            return {
                "mode": "dry_run",
                "summary": {
                    "stale_pending_inbox": pending_stale,
                    "active_memories_considered": duplicate_preview["active_memories_considered"],
                    "duplicate_pairs_compared": duplicate_preview["compared_pairs"],
                    "would_supersede_duplicates": duplicate_preview["would_supersede_count"],
                    "shown_duplicate_examples": len(duplicate_preview["examples"]),
                    "truncated_duplicate_examples": duplicate_preview["examples_truncated"],
                },
                "would_expire_inbox": stale_examples,
                "would_supersede_duplicates": duplicate_preview["examples"],
                "settings": {"owner": owner, "domain": domain, "max_inbox_age_days": max_age_days, "duplicate_threshold": threshold},
            }
        expired = self.store.expire_pending_inbox_items(
            max_age_days=max_age_days,
            reason="dashboard autopilot stale pending expiry",
            owner=owner,
            domain=domain,
        )
        superseded = self.store.supersede_duplicate_memories(owner=owner, domain=domain, threshold=threshold)
        out = {
            "mode": "applied",
            "expired_inbox": [inbox_item_dict(item) for item in expired],
            "superseded_duplicates": superseded,
            "summary": {"expired_inbox": len(expired), "superseded_duplicates": len(superseded)},
        }
        self._record_audit("maintenance_autopilot", out)
        return out

    def runtime_status(self) -> JsonDict:
        recall = self.store.recall_config()
        synthesis = self.store.synthesis_config()
        active_model = self.store.active_embedding_model() or ""
        with self.store._connect() as conn:  # noqa: SLF001
            db_counts = {
                "memories": int(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]),
                "inbox": int(conn.execute("SELECT COUNT(*) FROM memory_inbox").fetchone()[0]),
                "audit_events": int(conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]),
            }
            last_recall = conn.execute(
                "SELECT rid,event_type,reason,created_at,metadata_json FROM audit_log WHERE event_type='recall_query' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return {
            "db_path": str(self.store.db_path),
            "dashboard_host": "local stdlib server",
            "runtime_injection": "AnamnesisMemoryProvider is available; prefetch/tool recall calls store.recall and writes recall_query audit events.",
            "embedding": {"enabled": _setting_bool(recall.get("embedding_enabled"), default=False), "active_model": active_model},
            "synthesis": {"enabled": _setting_bool(synthesis.get("enabled"), default=False), "base_url": synthesis.get("base_url", ""), "model": synthesis.get("model", "")},
            "counts": db_counts,
            "last_recall": _audit_event_dict(last_recall) if last_recall else None,
            "last_recall_event": _audit_event_dict(last_recall) if last_recall else None,
        }

    def runtime_test_recall(self, payload: JsonDict) -> JsonDict:
        query = _required_str(payload, "query")
        owner = str(payload.get("owner") or "primary")
        platform = str(payload.get("platform") or "whatsapp")
        domain = str(payload.get("domain") or "") or None
        visibility = _visibility_list(payload.get("visibility") or ["private"])
        limit = max(1, min(50, int(payload.get("limit") or 10)))
        results = self.store.recall(
            query,
            owner=owner,
            platform=platform,
            allowed_visibility=set(visibility),
            limit=limit,
            domain=domain,
        )
        with self.store._connect() as conn:  # noqa: SLF001
            event = conn.execute(
                "SELECT rid,event_type,reason,created_at,metadata_json FROM audit_log WHERE event_type='recall_query' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return {
            "mode": "runtime_test_recall",
            "query": query,
            "owner": owner,
            "platform": platform,
            "allowed_visibility": visibility,
            "domain": domain,
            "limit": limit,
            "result_count": len(results),
            "included": [
                {
                    "rid": result.record.rid,
                    "text": result.record.text,
                    "score": result.score,
                    "reasons": result.reasons,
                    "domain": result.record.domain,
                    "platform_scope": result.record.platform_scope,
                }
                for result in results
            ],
            "context_preview": "\n".join(result.record.text for result in results),
            "audit_event": _audit_event_dict(event) if event else None,
        }

    def _record_audit(self, event_type: str, payload: JsonDict) -> None:
        with self.store._connect() as conn:  # noqa: SLF001
            conn.execute(
                "INSERT INTO audit_log (rid,event_type,reason,created_at,metadata_json) VALUES (?,?,?,?,?)",
                ("dashboard", event_type, event_type, time.time(), json.dumps(payload, sort_keys=True)),
            )

    def accept_inbox(self, payload: JsonDict) -> JsonDict:
        cid = _required_str(payload, "cid")
        record = self.store.accept_inbox_item(cid)
        item = self.store.get_inbox_item(cid)
        return {"item": inbox_item_dict(item), "memory": memory_record_dict(record)}

    def reject_inbox(self, payload: JsonDict) -> JsonDict:
        cid = _required_str(payload, "cid")
        reason = str(payload.get("reason") or "dashboard rejection")
        item = self.store.reject_inbox_item(cid, reason=reason)
        return {"item": inbox_item_dict(item), "reason": reason}

    def batch_inbox(self, payload: JsonDict) -> JsonDict:
        action = _required_str(payload, "action")
        cids = _required_str_list(payload, "cids")
        results: list[JsonDict] = []
        errors: list[JsonDict] = []
        for cid in cids:
            try:
                if action == "accept":
                    record = self.store.accept_inbox_item(cid)
                    item = self.store.get_inbox_item(cid)
                    results.append({"cid": cid, "item": inbox_item_dict(item), "memory": memory_record_dict(record)})
                elif action == "reject":
                    item = self.store.reject_inbox_item(
                        cid, reason=str(payload.get("reason") or "dashboard batch rejection")
                    )
                    results.append({"cid": cid, "item": inbox_item_dict(item)})
                else:
                    raise ValueError("action must be accept or reject")
            except (KeyError, ValueError) as exc:
                errors.append({"cid": cid, "error": str(exc)})
        return {"action": action, "requested": len(cids), "changed": len(results), "errors": errors, "results": results}

    def batch_memories(self, payload: JsonDict) -> JsonDict:
        action = _required_str(payload, "action")
        rids = _required_str_list(payload, "rids")
        if action != "invalidate":
            raise ValueError("action must be invalidate")
        results: list[JsonDict] = []
        errors: list[JsonDict] = []
        reason = str(payload.get("reason") or "dashboard batch invalidate")
        for rid in rids:
            try:
                self.store.invalidate(rid, reason=reason)
                record = self.store.get_memory(rid)
                results.append({"rid": rid, "memory": memory_record_dict(record)})
            except (KeyError, ValueError) as exc:
                errors.append({"rid": rid, "error": str(exc)})
        return {"action": action, "requested": len(rids), "changed": len(results), "errors": errors, "results": results}

    def _preview_payload(
        self,
        *,
        text: str,
        owner: str,
        platform: str,
        visibility: list[str],
        domain: str,
        limit: int,
    ) -> JsonDict:
        decision = classify_intake(text, domain=domain)
        platform_scope = platform if decision.lifecycle == "sensitive" or is_platform_local_text(text) else "all"
        simulation = self.store.simulate_recall(
            text,
            owner=owner,
            platform=platform,
            allowed_visibility=set(visibility or ["private"]),
            limit=max(1, limit),
            domain=domain or None,
        )
        return {
            "input": {"text": text, "owner": owner, "platform": platform, "domain": domain},
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

    def _apply_preview_write(
        self,
        *,
        text: str,
        owner: str,
        visibility: str,
        domain: str,
        payload: JsonDict,
        source: str,
        metadata: dict[str, object],
    ) -> dict[str, str] | None:
        would_write = payload["would_write"]
        action = str(would_write["action"])
        write_metadata = {
            "source_platform": would_write["source_platform"],
            "intake_reasons": would_write["reasons"],
            "intake_lifecycle": would_write["lifecycle"],
            **metadata,
        }
        if action == "accept":
            record = self.store.add_memory(
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
            item = self.store.propose_memory(
                text,
                source_snippet=text[:500],
                owner=owner,
                visibility=visibility,
                platform_scope=str(would_write["platform_scope"]),
                domain=domain or str(would_write["lifecycle"]),
                source=source,
                confidence=float(would_write["confidence"]),
                why_save=", ".join(would_write["reasons"]),
                suggested_lifecycle=str(would_write["lifecycle"]),
            )
            return {"cid": item.cid, "kind": "inbox_item"}
        return None

    @staticmethod
    def _parse_body(body: bytes) -> JsonDict:
        if not body:
            return {}
        try:
            payload = json.loads(body.decode())
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    @staticmethod
    def _asset(filename: str, content_type: str) -> tuple[int, dict[str, str], bytes]:
        content = resources.files("anamnesis.static").joinpath(filename).read_bytes()
        return 200, {"content-type": content_type}, content

    @staticmethod
    def _json(
        status: int,
        payload: JsonDict,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        headers = {"content-type": "application/json; charset=utf-8"}
        if extra_headers:
            headers.update(extra_headers)
        return (
            status,
            headers,
            json.dumps(payload, sort_keys=True).encode(),
        )


def memory_record_dict(record: MemoryRecord) -> JsonDict:
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
        "last_access": record.last_access,
        "ttl_days": record.ttl_days,
        "metadata": record.metadata,
    }


def inbox_item_dict(item: MemoryInboxItem) -> JsonDict:
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


def _audit_event_dict(row: Any) -> JsonDict:
    return {
        "rid": row["rid"],
        "event_type": row["event_type"],
        "reason": row["reason"],
        "created_at": float(row["created_at"]),
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _clear_cookie_header() -> str:
    return "anamnesis_session=; Max-Age=0; Path=/; HttpOnly; SameSite=Strict"


class _DashboardEmbedder:
    def __init__(self, model_id: str, dimension: int):
        self._model_id = model_id
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in text.lower().replace("-", " ").split():
            idx = sum(ord(ch) for ch in token) % self._dimension
            vector[idx] += 1.0
        return normalize(vector)


def _dashboard_embedder(model_name: str) -> _DashboardEmbedder:
    spec = get_model_spec(model_name)
    return _DashboardEmbedder(model_id=spec.model_id, dimension=spec.dimension)


def _dashboard_backfill_embedder(model_name: str) -> Any:
    embedder = load_embedder_by_name(model_name)
    if embedder is None:
        raise ValueError(f"embedding model {model_name!r} resolved to no embedder")
    return embedder


def _filled_counts(counts: dict[str, int], keys: list[str]) -> dict[str, int]:
    return {key: int(counts.get(key, 0)) for key in keys}


def _setting_bool(value: str | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _facet_rows(conn: Any, table: str, field: str) -> list[JsonDict]:
    if table not in {"memories", "memory_inbox"}:
        raise ValueError("unsupported facet table")
    if field not in {"owner", "domain", "source", "platform_scope"}:
        raise ValueError("unsupported facet field")
    rows = conn.execute(
        f"SELECT {field} AS value, COUNT(*) AS count FROM {table} WHERE {field} != '' "
        f"GROUP BY {field} ORDER BY count DESC, value ASC LIMIT 100"
    ).fetchall()
    return [{"value": row["value"], "count": int(row["count"])} for row in rows]


def _offset(value: str | None) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except ValueError as exc:
        raise ValueError("offset must be an integer") from exc


def _limit(value: str | None, *, default: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        return max(1, min(maximum, int(value)))
    except ValueError as exc:
        raise ValueError("limit must be an integer") from exc


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("confidence filter must be numeric") from exc


def _visibility_list(value: object) -> list[str]:
    if value is None:
        return ["private"]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()] or ["private"]
    if isinstance(value, list):
        return [str(part) for part in value if str(part).strip()] or ["private"]
    raise ValueError("visibility must be a string or list")


def _required_str(payload: JsonDict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()


def _required_str_list(payload: JsonDict, key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        raise ValueError(f"{key} must not be empty")
    if len(items) > 200:
        raise ValueError(f"{key} is limited to 200 items")
    return items


def make_handler(store: Anamnesis) -> type[BaseHTTPRequestHandler]:
    api = DashboardAPI(store)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib hook.
            self._send(api.handle("GET", self.path, headers=dict(self.headers.items())))

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook.
            length = int(self.headers.get("content-length") or 0)
            self._send(api.handle("POST", self.path, self.rfile.read(length), dict(self.headers.items())))

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature.
            return

        def _send(self, result: tuple[int, dict[str, str], bytes]) -> None:
            status, headers, body = result
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(db_path: str | Path, *, host: str = "127.0.0.1", port: int = 8766) -> None:
    store = Anamnesis(db_path)
    server = ThreadingHTTPServer((host, port), make_handler(store))
    server.serve_forever()

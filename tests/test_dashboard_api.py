from __future__ import annotations

import json

from anamnesis import Anamnesis
from anamnesis import dashboard as dashboard_module
from anamnesis.dashboard import DashboardAPI


def _json(api: DashboardAPI, method: str, path: str, payload: dict[str, object] | None = None):
    status, payload_json, _ = _json_with_headers(api, method, path, payload)
    return status, payload_json


def _json_with_headers(
    api: DashboardAPI,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
):
    status, response_headers, body = api.handle(
        method,
        path,
        json.dumps(payload).encode() if payload is not None else b"",
        headers=headers,
    )
    assert response_headers["content-type"] == "application/json; charset=utf-8"
    return status, json.loads(body.decode()), response_headers


def test_dashboard_overview_memories_inbox_and_audit(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    first = store.add_memory(
        "Primary user prefers concise updates with concrete next steps.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        domain="preference",
        source="test",
        importance=0.8,
        confidence=0.9,
    )
    second = store.add_memory(
        "Temporary process 12345 is running.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="ops",
        source="test",
    )
    store.invalidate(second.rid, reason="temporary")
    inbox = store.propose_memory(
        "Potential preference needs review.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        domain="preference",
        source="test",
        why_save="ambiguous",
        confidence=0.92,
    )
    low_confidence_inbox = store.propose_memory(
        "Weak imported candidate.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        domain="preference",
        source="test",
        confidence=0.35,
    )

    api = DashboardAPI(store)

    status, overview = _json(api, "GET", "/api/overview")
    assert status == 200
    assert overview["counts"]["memories"]["active"] == 1
    assert overview["counts"]["memories"]["invalidated"] == 1
    assert overview["counts"]["inbox"]["pending"] == 2
    assert overview["recent_memories"][0]["rid"] == first.rid
    assert {item["cid"] for item in overview["recent_inbox"]} >= {inbox.cid, low_confidence_inbox.cid}

    status, memories = _json(api, "GET", "/api/memories?owner=primary&status=active&limit=10")
    assert status == 200
    assert memories["total"] == 1
    assert memories["offset"] == 0
    assert memories["has_next"] is False
    assert [item["rid"] for item in memories["items"]] == [first.rid]
    assert memories["items"][0]["metadata"] == {}

    status, search_memories = _json(api, "GET", "/api/memories?status=active&q=concise")
    assert status == 200
    assert search_memories["total"] == 1

    status, facets = _json(api, "GET", "/api/facets")
    assert status == 200
    assert facets["memories"]["owners"] == [{"count": 2, "value": "primary"}]
    assert facets["memories"]["platforms"] == [{"count": 1, "value": "all"}, {"count": 1, "value": "whatsapp"}]
    assert facets["inbox"]["owners"] == [{"count": 2, "value": "primary"}]
    assert facets["inbox"]["platforms"] == [{"count": 2, "value": "all"}]

    status, inbox_payload = _json(api, "GET", "/api/inbox?decision=pending&limit=10&offset=0")
    assert status == 200
    assert {item["cid"] for item in inbox_payload["items"]} >= {inbox.cid, low_confidence_inbox.cid}
    assert inbox_payload["items"][0]["decision"] == "pending"

    status, high_confidence = _json(api, "GET", "/api/inbox?decision=pending&min_confidence=0.9")
    assert status == 200
    assert [item["cid"] for item in high_confidence["items"]] == [inbox.cid]

    status, low_confidence = _json(api, "GET", "/api/inbox?decision=pending&max_confidence=0.5")
    assert status == 200
    assert [item["cid"] for item in low_confidence["items"]] == [low_confidence_inbox.cid]

    status, inbox_search = _json(api, "GET", "/api/inbox?decision=pending&q=Potential")
    assert status == 200
    assert inbox_search["total"] == 1

    status, accepted = _json(api, "POST", "/api/inbox/accept", {"cid": inbox.cid})
    assert status == 200
    assert accepted["item"]["decision"] == "accepted"
    assert accepted["item"]["accepted_rid"] == accepted["memory"]["rid"]
    assert accepted["memory"]["text"] == "Potential preference needs review."

    rejected_inbox = store.propose_memory(
        "Reject this imported candidate.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        domain="preference",
        source="test",
    )
    status, rejected = _json(
        api,
        "POST",
        "/api/inbox/reject",
        {"cid": rejected_inbox.cid, "reason": "not useful"},
    )
    assert status == 200
    assert rejected["item"]["decision"] == "rejected"
    assert rejected["item"]["review_reason"] == "not useful"

    batch_a = store.propose_memory(
        "Batch accept me.", owner="primary", visibility="private", platform_scope="all", domain="test", source="test"
    )
    batch_b = store.propose_memory(
        "Batch reject me.", owner="primary", visibility="private", platform_scope="all", domain="test", source="test"
    )
    status, batch_accepted = _json(api, "POST", "/api/inbox/batch", {"action": "accept", "cids": [batch_a.cid]})
    assert status == 200
    assert batch_accepted["changed"] == 1
    status, batch_rejected = _json(api, "POST", "/api/inbox/batch", {"action": "reject", "cids": [batch_b.cid]})
    assert status == 200
    assert batch_rejected["changed"] == 1

    status, batch_memory = _json(api, "POST", "/api/memories/batch", {"action": "invalidate", "rids": [first.rid]})
    assert status == 200
    assert batch_memory["changed"] == 1
    assert batch_memory["results"][0]["memory"]["status"] == "invalidated"

    status, audit = _json(api, "GET", f"/api/audit/{first.rid}")
    assert status == 200
    assert audit["memory"]["rid"] == first.rid
    assert audit["events"][0]["event_type"] == "memory_added"


def test_dashboard_preview_and_correction_routes_are_dry_run_by_default(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    original = store.add_memory(
        "Primary user prefers verbose weekly reports.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        domain="preference",
        source="test",
    )
    api = DashboardAPI(store)

    status, preview = _json(
        api,
        "POST",
        "/api/preview-memory-write",
        {
            "text": "How are U considering what's good to store or not without an llm",
            "owner": "primary",
            "platform": "whatsapp",
            "target": "memory",
            "origin": "background_review",
        },
    )
    assert status == 200
    assert preview["mode"] == "preview_memory_write"
    assert preview["would_write"]["action"] == "reject"
    assert preview["input"]["source"] == "hermes_memory_tool"
    assert preview["apply"] is False
    with store._connect() as conn:  # noqa: SLF001 - assert dry-run boundary.
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1

    status, turn = _json(
        api,
        "POST",
        "/api/preview-turn",
        {"text": "Primary user prefers concise updates.", "owner": "primary", "platform": "whatsapp"},
    )
    assert status == 200
    assert turn["mode"] == "preview"
    assert turn["would_write"]["platform_scope"] == "all"

    status, corrected = _json(
        api,
        "POST",
        "/api/correct",
        {"rid": original.rid, "text": "Primary user prefers concise updates.", "reason": "user correction"},
    )
    assert status == 200
    assert corrected["old"]["status"] == "invalidated"
    assert corrected["replacement"]["text"] == "Primary user prefers concise updates."


def test_dashboard_returns_json_errors(tmp_path):
    api = DashboardAPI(Anamnesis(tmp_path / "anamnesis.db"))

    status, payload = _json(api, "GET", "/api/audit/missing")
    assert status == 404
    assert payload == {"error": "not_found", "message": "missing"}

    status, payload = _json(api, "POST", "/api/preview-turn", {"owner": "primary"})
    assert status == 400
    assert payload["error"] == "bad_request"
    assert "text" in payload["message"]


def test_dashboard_operational_surfaces(tmp_path, monkeypatch):
    store = Anamnesis(tmp_path / "anamnesis.db")
    store.add_memory(
        "Primary user prefers local-first memory dashboards.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        domain="preference",
    )
    store.add_memory(
        "Primary user prefers local-first memory dashboards.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        domain="preference",
        confidence=0.4,
    )
    inbox_item = store.propose_memory(
        "Old pending proposal.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        domain="preference",
        source="test",
    )
    with store._connect() as conn:  # noqa: SLF001
        conn.execute("UPDATE memory_inbox SET created_at=? WHERE cid=?", (0, inbox_item.cid))
    api = DashboardAPI(store)
    monkeypatch.setattr(
        dashboard_module,
        "_dashboard_backfill_embedder",
        lambda model_name: dashboard_module._dashboard_embedder(model_name),
    )

    status, embedding = _json(api, "GET", "/api/embedding/status")
    assert status == 200
    assert embedding["total_active"] == 2
    assert embedding["fts_fallback"] is True

    status, backfill = _json(api, "POST", "/api/embedding/backfill", {"model": "potion-base-2M"})
    assert status == 200
    assert backfill["after"]["embedded"] == 2

    status, sim = _json(
        api,
        "POST",
        "/api/recall/simulate",
        {"query": "local-first dashboards", "owner": "primary", "platform": "whatsapp"},
    )
    assert status == 200
    assert sim["query"] == "local-first dashboards"
    assert "included" in sim
    assert "excluded" in sim

    status, dry_run = _json(
        api,
        "POST",
        "/api/maintenance/autopilot",
        {"apply": False, "owner": "primary", "domain": "preference", "max_inbox_age_days": 0},
    )
    assert status == 200
    assert dry_run["mode"] == "dry_run"
    assert dry_run["summary"]["stale_pending_inbox"] == 1
    assert dry_run["summary"]["would_supersede_duplicates"] == 1
    assert dry_run["would_expire_inbox"][0]["proposed_text"] == "Old pending proposal."
    assert dry_run["would_supersede_duplicates"][0]["canonical_text"] == "Primary user prefers local-first memory dashboards."
    assert dry_run["would_supersede_duplicates"][0]["superseded_text"] == "Primary user prefers local-first memory dashboards."

    store.recall("local-first dashboards", owner="primary", platform="whatsapp", allowed_visibility={"private"})
    status, runtime_test = _json(
        api,
        "POST",
        "/api/runtime/test-recall",
        {"query": "local-first dashboards", "owner": "primary", "platform": "whatsapp", "limit": 5},
    )
    assert status == 200
    assert runtime_test["mode"] == "runtime_test_recall"
    assert runtime_test["query"] == "local-first dashboards"
    assert runtime_test["owner"] == "primary"
    assert runtime_test["platform"] == "whatsapp"
    assert runtime_test["result_count"] == 2
    assert [row["text"] for row in runtime_test["included"]] == [
        "Primary user prefers local-first memory dashboards.",
        "Primary user prefers local-first memory dashboards.",
    ]
    assert runtime_test["audit_event"]["event_type"] == "recall_query"
    assert runtime_test["audit_event"]["metadata"]["query"] == "local-first dashboards"
    assert [row["rid"] for row in runtime_test["included"]] == [row["rid"] for row in sim["included"]]

    status, runtime = _json(api, "GET", "/api/runtime/status")
    assert status == 200
    assert runtime["counts"]["memories"] == 2
    assert "AnamnesisMemoryProvider" in runtime["runtime_injection"]
    assert runtime["last_recall"]["event_type"] == "recall_query"
    assert runtime["last_recall"]["metadata"]["query"] == "local-first dashboards"
    assert runtime["last_recall"]["metadata"]["result_count"] == 2


def test_dashboard_settings_read_and_write(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    api = DashboardAPI(store)

    status, settings = _json(api, "GET", "/api/settings")
    assert status == 200
    assert settings["dashboard_password_set"] is False
    assert settings["embedding"]["active_model"] == ""
    assert settings["embedding"]["enabled"] is False
    assert len(settings["embedding"]["available_models"]) >= 3
    assert settings["synthesis"]["base_url"] == ""
    assert settings["synthesis"]["enabled"] is False

    status, result = _json(api, "POST", "/api/settings/embedding", {"model": "potion-base-32M", "enabled": True})
    assert status == 200
    assert result["active_model"] == "potion-base-32M"
    assert result["enabled"] is True

    status, settings2 = _json(api, "GET", "/api/settings")
    assert settings2["embedding"]["active_model"] == "potion-base-32M"
    assert settings2["embedding"]["enabled"] is True

    status, result = _json(api, "POST", "/api/settings/synthesis", {"enabled": True, "base_url": "http://127.0.0.1:8060/v1", "model": "test-model", "temperature": 0.3})
    assert status == 200

    status, settings3 = _json(api, "GET", "/api/settings")
    assert settings3["synthesis"]["base_url"] == "http://127.0.0.1:8060/v1"
    assert settings3["synthesis"]["enabled"] is True
    assert settings3["synthesis"]["model"] == "test-model"
    assert settings3["synthesis"]["temperature"] == 0.3

    status, result, set_headers = _json_with_headers(api, "POST", "/api/settings/dashboard-password", {"password": "secret123"})
    assert status == 200
    assert result["dashboard_password_set"] is True
    assert result["session_cleared"] is True
    assert "Max-Age=0" in set_headers["set-cookie"]

    status, blocked = _json(api, "GET", "/api/settings")
    assert status == 401
    assert blocked["error"] == "unauthorized"

    status, login_ok, headers = _json_with_headers(api, "POST", "/api/auth/login", {"password": "secret123"})
    assert status == 200
    assert login_ok["authenticated"] is True
    assert "anamnesis_session=" in headers["set-cookie"]
    cookie = headers["set-cookie"].split(";", 1)[0]

    status, settings4, _ = _json_with_headers(api, "GET", "/api/settings", headers={"Cookie": cookie})
    assert status == 200
    assert settings4["dashboard_password_set"] is True

    status, status_payload = _json(api, "GET", "/api/auth/status")
    assert status == 200
    assert status_payload == {"authenticated": False, "password_required": True}

    status, authed_status, _ = _json_with_headers(api, "GET", "/api/auth/status", headers={"Cookie": cookie})
    assert status == 200
    assert authed_status == {"authenticated": True, "password_required": True}

    status, changed, change_headers = _json_with_headers(
        api,
        "POST",
        "/api/settings/dashboard-password",
        {"password": "new-secret"},
        headers={"Cookie": cookie},
    )
    assert status == 200
    assert changed == {"dashboard_password_set": True, "session_cleared": True}
    assert "Max-Age=0" in change_headers["set-cookie"]

    status, old_cookie_blocked, _ = _json_with_headers(api, "GET", "/api/settings", headers={"Cookie": cookie})
    assert status == 401
    assert old_cookie_blocked["error"] == "unauthorized"

    status, login_new, new_headers = _json_with_headers(api, "POST", "/api/auth/login", {"password": "new-secret"})
    assert status == 200
    assert login_new["authenticated"] is True
    new_cookie = new_headers["set-cookie"].split(";", 1)[0]

    status, cleared, clear_headers = _json_with_headers(
        api,
        "POST",
        "/api/settings/dashboard-password/clear",
        {},
        headers={"Cookie": new_cookie},
    )
    assert status == 200
    assert cleared == {"dashboard_password_set": False, "session_cleared": True}
    assert "Max-Age=0" in clear_headers["set-cookie"]

    status, settings5 = _json(api, "GET", "/api/settings")
    assert status == 200
    assert settings5["dashboard_password_set"] is False

    status, _ = _json(api, "POST", "/api/auth/login", {"password": "wrong"})
    assert status == 200

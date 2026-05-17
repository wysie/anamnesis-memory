from __future__ import annotations

import json

from anamnesis import Anamnesis
from anamnesis.hermes_provider import AnamnesisMemoryProvider


def test_hermes_provider_recall_enforces_scope_and_invalidates(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    visible = store.add_memory(
        "Dashboard lives at 127.0.0.1:8765.",
        owner="default",
        platform_scope="cli",
        domain="dashboard",
    )
    store.add_memory(
        "Other owner secret token scope-leak-alpha.",
        owner="other",
        platform_scope="cli",
    )
    invalidated = store.add_memory(
        "Archived project code obsolete-raven.",
        owner="default",
        platform_scope="cli",
    )
    store.invalidate(invalidated.rid, reason="test")

    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "default")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="cli")

    raw = provider.handle_tool_call(
        "anamnesis_recall",
        {"query": "dashboard 8765 secret obsolete-raven", "limit": 10},
    )
    payload = json.loads(raw)
    texts = [row["text"] for row in payload["results"]]

    assert visible.text in texts
    assert not any("scope-leak-alpha" in text for text in texts)
    assert not any("obsolete-raven" in text for text in texts)


def test_hermes_provider_remember_and_stats(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "default")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="cli")

    remembered = json.loads(
        provider.handle_tool_call(
            "anamnesis_remember",
            {
                "text": "User prefers understated benchmark reporting.",
                "domain": "preference",
                "importance": 0.8,
            },
        )
    )
    assert remembered["success"] is True
    assert remembered["memory"]["domain"] == "preference"

    recalled = json.loads(
        provider.handle_tool_call(
            "anamnesis_recall", {"query": "understated benchmark reporting"}
        )
    )
    assert recalled["results"][0]["rid"] == remembered["memory"]["rid"]

    stats = json.loads(provider.handle_tool_call("anamnesis_stats", {}))
    assert stats["memories"]["active"] == 1


def test_sync_turn_autopilot_accepts_inboxes_and_rejects_by_policy(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "default")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="whatsapp")

    provider.sync_turn("Temporary debug port is 54321.", "assistant summary")
    provider.sync_turn("Maybe the dashboard should live on port 8765.", "assistant summary")
    provider.sync_turn(
        "Primary user wants Anamnesis work scoped to the Anamnesis repo only.",
        "assistant summary",
    )

    store = Anamnesis(db_path)
    recalled = store.recall(
        "Anamnesis scoped repo only temporary dashboard port 8765",
        owner="default",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=10,
    )
    texts = [result.record.text for result in recalled]
    pending = store.inbox_items(decision="pending")

    assert "Primary user wants Anamnesis work scoped to the Anamnesis repo only." in texts
    assert [item.proposed_text for item in pending] == ["Maybe the dashboard should live on port 8765."]
    assert not any("Temporary debug port" in text for text in texts)
    assert not any("Maybe the dashboard" in text for text in texts)
    assert store.inbox_items(decision="rejected") == []


def test_sync_turn_defaults_to_all_scope_and_preserves_source_platform(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    aliases = {
        "whatsapp:primary-user@example": "primary",
        "telegram:17847389": "primary",
    }
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.delenv("ANAMNESIS_OWNER", raising=False)
    monkeypatch.setenv("ANAMNESIS_OWNER_ALIASES", json.dumps(aliases))

    whatsapp_provider = AnamnesisMemoryProvider()
    whatsapp_provider.initialize(
        "wa-session",
        hermes_home=str(tmp_path),
        platform="whatsapp",
        user_id="whatsapp:primary-user@example",
    )
    telegram_provider = AnamnesisMemoryProvider()
    telegram_provider.initialize(
        "tg-session",
        hermes_home=str(tmp_path),
        platform="telegram",
        user_id="17847389",
    )

    whatsapp_provider.sync_turn(
        "Primary user prefers Anamnesis memories shared across platforms by default.",
        "assistant summary",
    )

    store = Anamnesis(db_path)
    whatsapp_results = store.recall(
        "Anamnesis memories shared across platforms",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=10,
    )
    telegram_results = store.recall(
        "Anamnesis memories shared across platforms",
        owner="primary",
        platform="telegram",
        allowed_visibility={"private"},
        limit=10,
    )

    assert whatsapp_results
    record = whatsapp_results[0].record
    assert record.text == "Primary user prefers Anamnesis memories shared across platforms by default."
    assert record.platform_scope == "all"
    assert record.source == "hermes_sync_turn"
    assert record.metadata["source_platform"] == "whatsapp"
    assert [result.record.rid for result in telegram_results] == [record.rid]


def test_sync_turn_keeps_platform_local_memory_current_platform(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "primary")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="whatsapp")

    provider.sync_turn(
        "Primary user wants Trusted contact wording only on WhatsApp.",
        "assistant summary",
    )

    store = Anamnesis(db_path)
    record = store.recall(
        "Trusted contact wording WhatsApp",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=1,
    )[0].record

    assert record.platform_scope == "whatsapp"
    assert record.metadata["source_platform"] == "whatsapp"


def test_sync_turn_routes_sensitive_facts_to_current_platform_inbox(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "primary")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="telegram")

    provider.sync_turn(
        "Primary user password for the test service is sample-secret-123.",
        "assistant summary",
    )

    store = Anamnesis(db_path)
    inbox_items = store.inbox_items(decision="pending")

    assert [item.proposed_text for item in inbox_items] == [
        "Primary user password for the test service is sample-secret-123."
    ]
    assert inbox_items[0].platform_scope == "telegram"
    assert inbox_items[0].why_save == "sensitive_content"


def test_hermes_provider_stats_reports_source_and_scope_distribution(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "primary")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="whatsapp")

    provider.sync_turn(
        "Primary user prefers cross-platform Anamnesis memory provenance stats.",
        "assistant summary",
    )
    provider.handle_tool_call(
        "anamnesis_remember",
        {
            "text": "Primary user keeps WhatsApp wording preferences local to WhatsApp.",
            "domain": "preference",
            "platform_scope": "current",
            "policy": "force_accept",
        },
    )

    stats = json.loads(provider.handle_tool_call("anamnesis_stats", {}))

    assert stats["platform_scopes"] == {"all": 1, "whatsapp": 1}
    assert stats["sources"] == {"hermes": 1, "hermes_sync_turn": 1}
    assert stats["source_platforms"] == {"unknown": 1, "whatsapp": 1}
    assert stats["owners"] == {"primary": 2}


def test_hermes_provider_remember_uses_autopilot_policy(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "default")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="cli")

    durable = json.loads(
        provider.handle_tool_call(
            "anamnesis_remember",
            {"text": "User prefers understated benchmark reporting.", "domain": "preference"},
        )
    )
    ambiguous = json.loads(
        provider.handle_tool_call(
            "anamnesis_remember",
            {"text": "Maybe the dashboard should live on port 8765.", "domain": "project"},
        )
    )
    rejected = json.loads(
        provider.handle_tool_call(
            "anamnesis_remember",
            {"text": "Temporary debug port is 54321.", "domain": "debug"},
        )
    )

    assert durable["success"] is True
    assert durable["action"] == "accepted"
    assert ambiguous["success"] is True
    assert ambiguous["action"] == "inboxed"
    assert ambiguous["inbox_item"]["decision"] == "pending"
    assert rejected["success"] is False
    assert rejected["action"] == "rejected"

    store = Anamnesis(db_path)
    assert len(store.inbox_items(decision="pending")) == 1
    recalled = store.recall(
        "understated dashboard temporary",
        owner="default",
        platform="cli",
        allowed_visibility={"private"},
        limit=10,
    )
    texts = [result.record.text for result in recalled]
    assert "User prefers understated benchmark reporting." in texts
    assert not any("Maybe the dashboard" in text for text in texts)
    assert not any("Temporary debug" in text for text in texts)


def test_prefetch_context_filters_junk_and_keeps_relevant_facts(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    store.add_memory("what’s next", owner="default", platform_scope="whatsapp", domain="conversation")
    store.add_memory(
        "[System note: Your previous turn was interrupted before tool results were processed.]",
        owner="default",
        platform_scope="whatsapp",
        domain="conversation",
    )
    store.add_memory(
        "Question: Where does the dashboard live and what URL/port does it use?",
        owner="default",
        platform_scope="whatsapp",
        domain="benchmark",
    )
    store.add_memory(
        "Primary user wants Anamnesis work scoped to the Anamnesis repo only.",
        owner="default",
        platform_scope="whatsapp",
        domain="preference",
    )

    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "default")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="whatsapp")
    provider.queue_prefetch("what are we doing next for Anamnesis", session_id="session")
    block = provider.prefetch("what are we doing next for Anamnesis", session_id="session")

    assert "Anamnesis repo only" in block
    assert "what’s next" not in block
    assert "System note" not in block
    assert "Question:" not in block


def test_provider_owner_prefers_chat_identity_over_agent_identity(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.delenv("ANAMNESIS_OWNER", raising=False)
    monkeypatch.delenv("ANAMNESIS_OWNER_ALIASES", raising=False)
    provider = AnamnesisMemoryProvider()
    provider.initialize(
        "session",
        hermes_home=str(tmp_path),
        platform="whatsapp",
        agent_identity="agent-hammy",
        user_id="whatsapp:primary-user@example",
    )

    remembered = json.loads(
        provider.handle_tool_call(
            "anamnesis_remember",
            {"text": "User prefers understated benchmark reporting.", "domain": "preference"},
        )
    )
    stats = json.loads(provider.handle_tool_call("anamnesis_stats", {}))

    assert remembered["memory"]["owner"] == "whatsapp:primary-user@example"
    assert stats["owner"] == "whatsapp:primary-user@example"


def test_provider_owner_aliases_merge_same_human_across_platforms(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    aliases = {
        "whatsapp:primary-user@example": "primary",
        "telegram:17847389": "primary",
    }
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.delenv("ANAMNESIS_OWNER", raising=False)
    monkeypatch.setenv("ANAMNESIS_OWNER_ALIASES", json.dumps(aliases))

    whatsapp_provider = AnamnesisMemoryProvider()
    whatsapp_provider.initialize(
        "wa-session",
        hermes_home=str(tmp_path),
        platform="whatsapp",
        user_id="whatsapp:primary-user@example",
    )
    telegram_provider = AnamnesisMemoryProvider()
    telegram_provider.initialize(
        "tg-session",
        hermes_home=str(tmp_path),
        platform="telegram",
        user_id="17847389",
    )

    wa_stats = json.loads(whatsapp_provider.handle_tool_call("anamnesis_stats", {}))
    tg_stats = json.loads(telegram_provider.handle_tool_call("anamnesis_stats", {}))

    assert wa_stats["owner"] == "primary"
    assert tg_stats["owner"] == "primary"


def test_provider_owner_aliases_keep_unknown_senders_namespaced(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.delenv("ANAMNESIS_OWNER", raising=False)
    monkeypatch.setenv(
        "ANAMNESIS_OWNER_ALIASES",
        json.dumps({"whatsapp:primary-user@example": "primary"}),
    )
    provider = AnamnesisMemoryProvider()
    provider.initialize(
        "session",
        hermes_home=str(tmp_path),
        platform="whatsapp",
        user_id="unknown@s.whatsapp.net",
    )

    stats = json.loads(provider.handle_tool_call("anamnesis_stats", {}))

    assert stats["owner"] == "whatsapp:unknown@s.whatsapp.net"


def test_provider_owner_env_override_wins_for_profile_level_memory(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "primary-profile")
    provider = AnamnesisMemoryProvider()
    provider.initialize(
        "session",
        hermes_home=str(tmp_path),
        platform="whatsapp",
        user_id="whatsapp:primary-user@example",
    )

    stats = json.loads(provider.handle_tool_call("anamnesis_stats", {}))

    assert stats["owner"] == "primary-profile"


def test_canonical_owner_still_respects_platform_scope(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    aliases = {
        "whatsapp:primary-user@example": "primary",
        "telegram:17847389": "primary",
    }
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.delenv("ANAMNESIS_OWNER", raising=False)
    monkeypatch.setenv("ANAMNESIS_OWNER_ALIASES", json.dumps(aliases))

    whatsapp_provider = AnamnesisMemoryProvider()
    whatsapp_provider.initialize(
        "wa-session",
        hermes_home=str(tmp_path),
        platform="whatsapp",
        user_id="whatsapp:primary-user@example",
    )
    telegram_provider = AnamnesisMemoryProvider()
    telegram_provider.initialize(
        "tg-session",
        hermes_home=str(tmp_path),
        platform="telegram",
        user_id="17847389",
    )

    whatsapp_only = json.loads(
        whatsapp_provider.handle_tool_call(
            "anamnesis_remember",
            {
                "text": "Primary user WhatsApp-only codename is sample-river.",
                "domain": "preference",
                "platform_scope": "whatsapp",
                "policy": "force_accept",
            },
        )
    )
    shared = json.loads(
        whatsapp_provider.handle_tool_call(
            "anamnesis_remember",
            {
                "text": "Primary user cross-platform codename is sample-lantern.",
                "domain": "preference",
                "platform_scope": "shared",
                "policy": "force_accept",
            },
        )
    )

    telegram_recall = json.loads(
        telegram_provider.handle_tool_call(
            "anamnesis_recall",
            {"query": "codename sample-river sample-lantern", "domain": "preference"},
        )
    )
    whatsapp_recall = json.loads(
        whatsapp_provider.handle_tool_call(
            "anamnesis_recall",
            {"query": "codename sample-river sample-lantern", "domain": "preference"},
        )
    )

    telegram_texts = [row["text"] for row in telegram_recall["results"]]
    whatsapp_texts = [row["text"] for row in whatsapp_recall["results"]]

    assert whatsapp_only["memory"]["owner"] == "primary"
    assert shared["memory"]["owner"] == "primary"
    assert shared["memory"]["platform_scope"] == "all"
    assert "Primary user cross-platform codename is sample-lantern." in telegram_texts
    assert "Primary user WhatsApp-only codename is sample-river." not in telegram_texts
    assert "Primary user cross-platform codename is sample-lantern." in whatsapp_texts
    assert "Primary user WhatsApp-only codename is sample-river." in whatsapp_texts


def test_on_memory_write_uses_intake_policy_not_blind_mirror(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "primary")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="whatsapp")

    provider.on_memory_write(
        "add",
        "memory",
        "Review the conversation above and consider saving to memory if appropriate.",
        {"origin": "background_review"},
    )
    provider.on_memory_write(
        "add",
        "memory",
        "How are U considering what's good to store or not without an llm",
        {"origin": "main_turn"},
    )
    provider.on_memory_write(
        "add",
        "memory",
        "Primary user prefers concise next-step suggestions after shipped work.",
        {"origin": "background_review"},
    )

    store = Anamnesis(db_path)
    recalled = store.recall(
        "conversation memory LLM concise next-step suggestions",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=10,
    )
    texts = [result.record.text for result in recalled]

    assert texts == ["Primary user prefers concise next-step suggestions after shipped work."]
    assert store.inbox_items(decision="pending") == []


def test_on_memory_write_preserves_source_platform_and_current_scope_exceptions(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "primary")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="telegram")

    provider.on_memory_write(
        "add",
        "preference",
        "Primary user wants this Telegram-only testing preference kept local to Telegram.",
        {"origin": "memory_tool"},
    )

    store = Anamnesis(db_path)
    record = store.recall(
        "Telegram-only testing preference",
        owner="primary",
        platform="telegram",
        allowed_visibility={"private"},
        limit=1,
    )[0].record

    assert record.platform_scope == "telegram"
    assert record.source == "hermes_memory_tool"
    assert record.metadata["source_platform"] == "telegram"
    assert record.metadata["origin"] == "memory_tool"


def test_hermes_provider_correct_tool_replaces_memory(tmp_path, monkeypatch):
    db_path = tmp_path / "anamnesis.db"
    monkeypatch.setenv("ANAMNESIS_DB_PATH", str(db_path))
    monkeypatch.setenv("ANAMNESIS_OWNER", "primary")
    provider = AnamnesisMemoryProvider()
    provider.initialize("session", hermes_home=str(tmp_path), platform="whatsapp")

    remembered = json.loads(
        provider.handle_tool_call(
            "anamnesis_remember",
            {"text": "Primary user prefers verbose updates.", "domain": "preference", "policy": "force_accept"},
        )
    )
    corrected = json.loads(
        provider.handle_tool_call(
            "anamnesis_correct",
            {
                "rid": remembered["memory"]["rid"],
                "text": "Primary user prefers concise updates with next steps.",
                "reason": "user correction",
            },
        )
    )

    assert corrected["success"] is True
    assert corrected["old"]["status"] == "invalidated"
    assert corrected["replacement"]["text"] == "Primary user prefers concise updates with next steps."
    assert corrected["replacement"]["metadata"]["corrects_rid"] == remembered["memory"]["rid"]

import json
import sqlite3

from anamnesis import Anamnesis, MemoryInboxItem, MemoryRecord, RecallResult


def table_names(db_path):
    with sqlite3.connect(db_path) as conn:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')")}


def test_init_creates_schema(tmp_path):
    db_path = tmp_path / "anamnesis.db"

    Anamnesis(db_path)

    names = table_names(db_path)
    assert "memories" in names
    assert "memory_fts" in names
    assert "memory_inbox" in names
    assert "audit_log" in names


def test_add_memory_persists_record_and_fts(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")

    record = store.add_memory(
        "Primary user prefers local-only private memory.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp,cli",
        domain="privacy",
        source="test",
        importance=0.9,
    )

    assert isinstance(record, MemoryRecord)
    assert record.rid
    assert record.text == "Primary user prefers local-only private memory."
    with sqlite3.connect(store.db_path) as conn:
        saved = conn.execute("SELECT text,status,owner FROM memories WHERE rid=?", (record.rid,)).fetchone()
        fts = conn.execute("SELECT rid FROM memory_fts WHERE memory_fts MATCH 'local' ").fetchone()
    assert saved == (record.text, "active", "primary")
    assert fts == (record.rid,)
    assert store.audit_events(record.rid)[0]["event_type"] == "memory_added"


def test_recall_enforces_scope_before_ranking(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    visible = store.add_memory(
        "Primary user prefers local-only private memory.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp,cli",
        domain="privacy",
        source="test",
        importance=0.9,
    )
    store.add_memory(
        "Collaborator can ask weather questions but cannot control devices.",
        owner="hope",
        visibility="private",
        platform_scope="whatsapp",
        domain="permissions",
        source="test",
        importance=1.0,
    )

    results = store.recall(
        "local memory devices",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
    )

    assert [r.record.rid for r in results] == [visible.rid]
    assert isinstance(results[0], RecallResult)
    assert "keyword_match" in results[0].reasons
    assert "scope_match" in results[0].reasons
    assert results[0].score > 0

    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        recall_event = conn.execute(
            "SELECT * FROM audit_log WHERE event_type='recall_query' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert recall_event is not None
    assert recall_event["rid"] == "runtime"
    assert recall_event["reason"] == "store.recall"
    metadata = json.loads(recall_event["metadata_json"])
    assert metadata["query"] == "local memory devices"
    assert metadata["owner"] == "primary"
    assert metadata["platform"] == "whatsapp"
    assert metadata["allowed_visibility"] == ["private"]
    assert metadata["result_count"] == 1
    assert metadata["result_rids"] == [visible.rid]


def test_tombstone_excludes_memory_from_recall_and_audits(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    record = store.add_memory(
        "This memory should disappear from recall after tombstone.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        source="test",
    )

    store.tombstone(record.rid, reason="obsolete")

    recalled = store.recall("disappear tombstone", owner="primary", platform="whatsapp", allowed_visibility={"private"})
    assert recalled == []
    assert store.get_memory(record.rid).status == "tombstoned"
    events = [e["event_type"] for e in store.audit_events(record.rid)]
    assert events == ["memory_added", "memory_tombstoned"]


def test_propose_memory_creates_pending_inbox_item(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")

    item = store.propose_memory(
        "User prefers local-first memory.",
        source_snippet="Please keep my memory local.",
        proposed_kind="semantic",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
        source="test",
        confidence=0.82,
        why_save="Durable user preference",
        suggested_lifecycle="permanent",
    )

    assert isinstance(item, MemoryInboxItem)
    assert item.decision == "pending"
    assert item.proposed_text == "User prefers local-first memory."
    assert item.duplicate_rids == []
    assert store.inbox_items()[0].cid == item.cid


def test_accept_inbox_item_writes_memory_and_audits(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    item = store.propose_memory(
        "User prefers local-first memory.",
        source_snippet="Please keep my memory local.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
        source="test",
    )

    record = store.accept_inbox_item(item.cid)

    assert record.text == item.proposed_text
    assert store.inbox_items(decision="accepted")[0].accepted_rid == record.rid
    assert [e["event_type"] for e in store.audit_events(record.rid)] == ["memory_added", "inbox_accepted"]
    assert store.recall("local-first", owner="primary", platform="whatsapp", allowed_visibility={"private"})[0].record.rid == record.rid


def test_reject_inbox_item_keeps_memory_store_clean(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    item = store.propose_memory(
        "Temporary debug PID is 12345.",
        source_snippet="PID 12345",
        owner="primary",
        visibility="private",
        platform_scope="cli",
        source="test",
    )

    store.reject_inbox_item(item.cid, reason="temporary task state")

    assert store.inbox_items(decision="rejected")[0].review_reason == "temporary task state"
    assert store.recall("debug PID", owner="primary", platform="cli", allowed_visibility={"private"}) == []


def test_recall_suppresses_stale_operational_matches_for_ephemeral_queries(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    store.add_memory(
        "Background process PID 12345 matched watch pattern and should not be durable.",
        owner="primary",
        visibility="private",
        platform_scope="cli",
        domain="task-state",
        source="test",
        importance=1.0,
    )
    durable = store.add_memory(
        "Primary user prefers local-only private memory.",
        owner="primary",
        visibility="private",
        platform_scope="cli",
        domain="privacy",
        source="test",
        importance=0.9,
    )

    assert store.recall("PID", owner="primary", platform="cli", allowed_visibility={"private"}) == []
    recalled = store.recall("local-only private", owner="primary", platform="cli", allowed_visibility={"private"})
    assert [result.record.rid for result in recalled] == [durable.rid]


def test_recall_keeps_stable_infra_when_query_is_not_ephemeral(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    infra = store.add_memory(
        "Hindsight infrastructure runs on port 8888 and uses a private-network LLM endpoint.",
        owner="primary",
        visibility="private",
        platform_scope="cli",
        domain="infrastructure",
        source="test",
        importance=0.8,
        metadata={"intake_lifecycle": "stable_infrastructure"},
    )

    recalled = store.recall("Hindsight infrastructure", owner="primary", platform="cli", allowed_visibility={"private"})

    assert [result.record.rid for result in recalled] == [infra.rid]
    assert store.recall("temporary server PID port", owner="primary", platform="cli", allowed_visibility={"private"}) == []


def test_recall_top_k_expansion_is_monotonic_for_general_facts(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    exact = store.add_memory(
        "Project Atlas uses a private SQLite memory store.",
        owner="agent",
        visibility="private",
        platform_scope="cli",
        domain="architecture",
        importance=0.2,
    )
    for idx in range(40):
        store.add_memory(
            f"Project Atlas generic planning note number {idx}.",
            owner="agent",
            visibility="private",
            platform_scope="cli",
            domain="notes",
            importance=1.0,
        )

    top_one = store.recall("Atlas private SQLite memory", owner="agent", platform="cli", allowed_visibility={"private"}, limit=1)
    top_twenty = store.recall("Atlas private SQLite memory", owner="agent", platform="cli", allowed_visibility={"private"}, limit=20)

    assert top_one[0].record.rid == exact.rid
    assert exact.rid in [result.record.rid for result in top_twenty]


def test_recall_boosts_exact_numeric_identifier_matches(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    exact = store.add_memory(
        "Service Mercury listens on port 17847389.",
        owner="agent",
        visibility="private",
        platform_scope="cli",
        domain="infrastructure",
        importance=0.1,
    )
    for idx in range(15):
        store.add_memory(
            f"Service Mercury deployment note {idx} mentions ports generally.",
            owner="agent",
            visibility="private",
            platform_scope="cli",
            domain="ops",
            importance=1.0,
        )

    recalled = store.recall("Mercury 17847389", owner="agent", platform="cli", allowed_visibility={"private"}, limit=5)

    assert recalled[0].record.rid == exact.rid


def test_recall_suppresses_question_only_benchmark_rows(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    store.add_memory(
        "Question: Where does the memory dashboard live and what URL/port does it use?",
        owner="agent",
        visibility="private",
        platform_scope="cli",
        domain="benchmark",
        importance=1.0,
    )
    exact = store.add_memory(
        "Memory Dashboard lives at ~/.local/share/memory-dashboard and uses http://127.0.0.1:8765/.",
        owner="agent",
        visibility="private",
        platform_scope="cli",
        domain="infrastructure",
        importance=0.5,
    )

    recalled = store.recall(
        "Where does the memory dashboard live and what URL or port does it use?",
        owner="agent",
        platform="cli",
        allowed_visibility={"private"},
        limit=5,
    )

    assert [result.record.rid for result in recalled] == [exact.rid]


def test_generic_query_words_do_not_get_specific_term_boost(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    exact = store.add_memory(
        "Family member does swimming with coach Darien.",
        owner="agent",
        visibility="private",
        platform_scope="cli",
        domain="household",
        importance=0.1,
    )
    store.add_memory(
        "Agent Panchubs should avoid adult corporate styling.",
        owner="agent",
        visibility="private",
        platform_scope="cli",
        domain="creative",
        importance=1.0,
    )

    recalled = store.recall(
        "Who does swimming with kid and adult, and who is the coach?",
        owner="agent",
        platform="cli",
        allowed_visibility={"private"},
        limit=5,
    )

    assert recalled[0].record.rid == exact.rid
    assert "specific_term_match" not in recalled[0].reasons


def test_propose_memory_records_duplicate_hints(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    existing = store.add_memory(
        "User prefers local-first private memory.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
        source="test",
    )

    item = store.propose_memory(
        "User prefers local-first memory.",
        source_snippet="local memory please",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
        source="test",
    )

    assert item.duplicate_rids == [existing.rid]
    assert "possible_duplicate" in item.hints


def test_correct_memory_tombstones_old_creates_replacement_and_audits(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    old = store.add_memory(
        "Primary user prefers verbose status reports.",
        owner="primary",
        visibility="private",
        platform_scope="all",
        action_scope="read_only",
        domain="preference",
        source="test",
        importance=0.7,
        confidence=0.8,
        metadata={"source_platform": "whatsapp"},
    )

    replacement = store.correct_memory(
        old.rid,
        "Primary user prefers concise status reports with concrete next steps.",
        reason="user correction",
    )

    assert store.get_memory(old.rid).status == "tombstoned"
    assert replacement.rid != old.rid
    assert replacement.text == "Primary user prefers concise status reports with concrete next steps."
    assert replacement.owner == old.owner
    assert replacement.visibility == old.visibility
    assert replacement.platform_scope == old.platform_scope
    assert replacement.action_scope == old.action_scope
    assert replacement.domain == old.domain
    assert replacement.source == old.source
    assert replacement.importance == old.importance
    assert replacement.confidence == old.confidence
    assert replacement.metadata["source_platform"] == "whatsapp"
    assert replacement.metadata["corrects_rid"] == old.rid
    assert replacement.metadata["correction_reason"] == "user correction"

    recalled = store.recall(
        "concise verbose status reports next steps",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=10,
    )
    assert [result.record.rid for result in recalled] == [replacement.rid]

    old_events = store.audit_events(old.rid)
    new_events = [event for event in store.audit_events(replacement.rid) if event["event_type"] != "memory_recalled"]
    assert [event["event_type"] for event in old_events] == [
        "memory_added",
        "memory_tombstoned",
        "memory_corrected_from",
    ]
    assert old_events[-1]["metadata"]["replacement_rid"] == replacement.rid
    assert [event["event_type"] for event in new_events] == ["memory_added", "memory_corrected_to"]
    assert new_events[-1]["metadata"]["old_rid"] == old.rid


def test_correct_memory_refuses_non_active_source(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    old = store.add_memory("Primary user prefers concise replies.", owner="primary")
    store.tombstone(old.rid, reason="obsolete")

    try:
        store.correct_memory(old.rid, "Primary user prefers concise replies with next steps.")
    except ValueError as exc:
        assert "active" in str(exc)
    else:  # pragma: no cover - documents expected failure path
        raise AssertionError("correct_memory should reject tombstoned source memories")

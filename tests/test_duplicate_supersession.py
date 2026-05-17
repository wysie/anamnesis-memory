from anamnesis import Anamnesis


def test_supersede_duplicate_memories_keeps_best_and_suppresses_duplicates(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    canonical = store.add_memory(
        "Primary user prefers local-only WhatsApp memory summaries.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
        importance=0.9,
        confidence=0.9,
    )
    duplicate = store.add_memory(
        "Primary user prefers local-only WhatsApp memory summaries.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
        importance=0.4,
        confidence=0.4,
    )
    distinct = store.add_memory(
        "Primary user prefers local dashboards with polished mobile UX.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
    )

    superseded = store.supersede_duplicate_memories(owner="primary", domain="privacy")

    assert [item["superseded_rid"] for item in superseded] == [duplicate.rid]
    assert superseded[0]["canonical_rid"] == canonical.rid
    assert store.get_memory(canonical.rid).status == "active"
    assert store.get_memory(duplicate.rid).status == "superseded"
    assert store.get_memory(distinct.rid).status == "active"

    recalled = store.recall(
        "local-only WhatsApp memory summaries mobile dashboard",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=10,
    )
    recalled_rids = [result.record.rid for result in recalled]

    assert canonical.rid in recalled_rids
    assert distinct.rid in recalled_rids
    assert duplicate.rid not in recalled_rids


def test_supersede_duplicate_memories_respects_scope(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    primary = store.add_memory("Same text duplicate candidate.", owner="primary", domain="privacy")
    other = store.add_memory("Same text duplicate candidate.", owner="other", domain="privacy")

    assert store.supersede_duplicate_memories(owner="primary", domain="privacy") == []
    assert store.get_memory(primary.rid).status == "active"
    assert store.get_memory(other.rid).status == "active"

from anamnesis import Anamnesis


def test_recall_simulator_explains_included_and_excluded_memories(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    included = store.add_memory(
        "Primary user prefers local-only WhatsApp memory.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
    )
    wrong_owner = store.add_memory(
        "Other owner WhatsApp memory secret.",
        owner="other",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
    )
    wrong_platform = store.add_memory(
        "Primary user prefers CLI-only benchmark notes.",
        owner="primary",
        visibility="private",
        platform_scope="cli",
        domain="privacy",
    )
    tombstoned = store.add_memory(
        "Primary user old WhatsApp memory obsolete.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
    )
    store.tombstone(tombstoned.rid, reason="obsolete")
    pending = store.propose_memory(
        "Primary user maybe wants temporary WhatsApp memory.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
    )

    simulation = store.simulate_recall(
        "WhatsApp memory",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        domain="privacy",
        limit=5,
    )

    included_rids = [item["rid"] for item in simulation["included"]]
    excluded_by_id = {item.get("rid") or item.get("cid"): item for item in simulation["excluded"]}

    assert included_rids == [included.rid]
    assert "included_in_recall" in simulation["included"][0]["reasons"]
    assert "owner_mismatch" in excluded_by_id[wrong_owner.rid]["exclusion_reasons"]
    assert "platform_scope_mismatch" in excluded_by_id[wrong_platform.rid]["exclusion_reasons"]
    assert "status_tombstoned" in excluded_by_id[tombstoned.rid]["exclusion_reasons"]
    assert "inbox_pending_not_recallable" in excluded_by_id[pending.cid]["exclusion_reasons"]

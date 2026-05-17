from anamnesis import Anamnesis, Contradiction


def test_detect_contradictions_flags_opposite_permission_memories(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    allowed = store.add_memory(
        "Helper can control devices.",
        owner="hope",
        visibility="private",
        platform_scope="whatsapp",
        domain="permissions",
        source="test",
    )
    forbidden = store.add_memory(
        "Helper cannot control devices.",
        owner="hope",
        visibility="private",
        platform_scope="whatsapp",
        domain="permissions",
        source="test",
    )

    conflicts = store.detect_contradictions(owner="hope", domain="permissions")

    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert isinstance(conflict, Contradiction)
    assert {conflict.left_rid, conflict.right_rid} == {allowed.rid, forbidden.rid}
    assert conflict.status == "open"
    assert "polarity_conflict" in conflict.reasons


def test_detect_contradictions_is_idempotent(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    store.add_memory("User can share WhatsApp summaries.", owner="primary", domain="privacy", source="test")
    store.add_memory("User cannot share WhatsApp summaries.", owner="primary", domain="privacy", source="test")

    first = store.detect_contradictions(owner="primary", domain="privacy")
    second = store.detect_contradictions(owner="primary", domain="privacy")

    assert len(first) == 1
    assert [c.conflict_id for c in first] == [c.conflict_id for c in second]
    assert len(store.contradictions(status="open")) == 1


def test_resolve_contradiction_tombstones_loser(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    old = store.add_memory("User can share WhatsApp summaries.", owner="primary", domain="privacy", source="test")
    new = store.add_memory("User cannot share WhatsApp summaries.", owner="primary", domain="privacy", source="test")
    conflict = store.detect_contradictions(owner="primary", domain="privacy")[0]

    resolved = store.resolve_contradiction(conflict.conflict_id, winner_rid=new.rid, reason="newer user correction")

    assert resolved.status == "resolved"
    assert resolved.winner_rid == new.rid
    assert store.get_memory(old.rid).status == "tombstoned"
    assert store.get_memory(new.rid).status == "active"

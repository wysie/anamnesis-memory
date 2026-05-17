from anamnesis import Anamnesis


def test_expire_pending_inbox_items_marks_old_pending_only(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    old_item = store.propose_memory("Maybe old pending memory should expire.", owner="primary")
    fresh_item = store.propose_memory("Maybe fresh pending memory should stay.", owner="primary")
    rejected_item = store.propose_memory("Maybe rejected memory should stay rejected.", owner="primary")
    store.reject_inbox_item(rejected_item.cid, reason="test")

    with store._connect() as conn:  # noqa: SLF001 - test controls fixture timestamps.
        conn.execute(
            "UPDATE memory_inbox SET created_at=created_at - ? WHERE cid=?",
            (40 * 24 * 60 * 60, old_item.cid),
        )
        conn.execute(
            "UPDATE memory_inbox SET created_at=created_at - ? WHERE cid=?",
            (40 * 24 * 60 * 60, rejected_item.cid),
        )

    expired = store.expire_pending_inbox_items(max_age_days=30, reason="stale pending")

    assert [item.cid for item in expired] == [old_item.cid]
    assert store.get_inbox_item(old_item.cid).decision == "expired"
    assert store.get_inbox_item(old_item.cid).review_reason == "stale pending"
    assert store.get_inbox_item(fresh_item.cid).decision == "pending"
    assert store.get_inbox_item(rejected_item.cid).decision == "rejected"


def test_expired_inbox_item_is_explained_as_not_recallable(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    item = store.propose_memory(
        "Maybe Primary user wants WhatsApp memory expiry.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
    )
    with store._connect() as conn:  # noqa: SLF001 - test controls fixture timestamps.
        conn.execute(
            "UPDATE memory_inbox SET created_at=created_at - ? WHERE cid=?",
            (40 * 24 * 60 * 60, item.cid),
        )
    store.expire_pending_inbox_items(max_age_days=30)

    simulation = store.simulate_recall(
        "WhatsApp memory expiry",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        domain="privacy",
    )
    excluded_by_id = {entry.get("cid"): entry for entry in simulation["excluded"]}

    assert "inbox_expired_not_recallable" in excluded_by_id[item.cid]["exclusion_reasons"]

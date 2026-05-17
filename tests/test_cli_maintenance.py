import json

from anamnesis import Anamnesis
from anamnesis.cli import main


def test_cli_maintenance_preview_duplicates(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    keep = store.add_memory(
        "Primary user prefers local-only WhatsApp memory summaries.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
        importance=0.9,
    )
    duplicate = store.add_memory(
        "Primary user prefers local-only WhatsApp memory summaries.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
        importance=0.2,
    )

    assert (
        main(
            [
                "--db",
                str(db_path),
                "maintenance",
                "supersede-duplicates",
                "--owner",
                "primary",
                "--domain",
                "privacy",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["superseded"] == [
        {"canonical_rid": keep.rid, "superseded_rid": duplicate.rid, "overlap": 1.0}
    ]


def test_cli_maintenance_autopilot_expires_and_supersedes(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    keep = store.add_memory(
        "Primary user prefers local-only WhatsApp memory summaries.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
        importance=0.9,
    )
    duplicate = store.add_memory(
        "Primary user prefers local-only WhatsApp memory summaries.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
        importance=0.2,
    )
    old_item = store.propose_memory("Maybe stale pending memory should expire.", owner="primary", domain="privacy")
    with store._connect() as conn:  # noqa: SLF001 - test controls fixture timestamps.
        conn.execute(
            "UPDATE memory_inbox SET created_at=created_at - ? WHERE cid=?",
            (40 * 24 * 60 * 60, old_item.cid),
        )

    assert (
        main(
            [
                "--db",
                str(db_path),
                "maintenance",
                "autopilot",
                "--owner",
                "primary",
                "--domain",
                "privacy",
                "--max-inbox-age-days",
                "30",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["expired_inbox"][0]["cid"] == old_item.cid
    assert payload["superseded_duplicates"] == [
        {"canonical_rid": keep.rid, "superseded_rid": duplicate.rid, "overlap": 1.0}
    ]
    assert store.get_inbox_item(old_item.cid).decision == "expired"
    assert store.get_memory(duplicate.rid).status == "superseded"


def test_cli_maintenance_autopilot_bare_command_is_safe_across_scopes(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    primary_keep = store.add_memory(
        "Primary user prefers local-only WhatsApp memory summaries.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
        importance=0.9,
    )
    primary_dup = store.add_memory(
        "Primary user prefers local-only WhatsApp memory summaries.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
        importance=0.2,
    )
    other = store.add_memory(
        "Primary user prefers local-only WhatsApp memory summaries.",
        owner="other",
        platform_scope="whatsapp",
        domain="privacy",
        importance=0.2,
    )

    assert main(["--db", str(db_path), "maintenance", "autopilot", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["superseded_duplicates"] == [
        {"canonical_rid": primary_keep.rid, "superseded_rid": primary_dup.rid, "overlap": 1.0}
    ]
    assert store.get_memory(primary_dup.rid).status == "superseded"
    assert store.get_memory(other.rid).status == "active"


def test_cli_maintenance_report_lists_recent_autopilot_runs(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    store.propose_memory("Maybe stale pending memory should expire.", owner="primary")
    with store._connect() as conn:  # noqa: SLF001 - test controls fixture timestamps.
        conn.execute("UPDATE memory_inbox SET created_at=created_at - ?", (40 * 24 * 60 * 60,))

    assert main(["--db", str(db_path), "maintenance", "autopilot", "--json"]) == 0
    capsys.readouterr()
    assert main(["--db", str(db_path), "maintenance", "report", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["runs"][0]["event_type"] == "maintenance_autopilot"
    assert payload["runs"][0]["summary"] == {"expired_inbox": 1, "superseded_duplicates": 0}

from __future__ import annotations

import json

from anamnesis import Anamnesis
from anamnesis.cli import main


def test_cli_inbox_propose_list_accept_and_reject(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"

    assert (
        main(
            [
                "--db",
                str(db_path),
                "inbox",
                "propose",
                "Maybe the dashboard should live on port 8765.",
                "--owner",
                "primary",
                "--platform",
                "whatsapp",
                "--domain",
                "project",
                "--json",
            ]
        )
        == 0
    )
    proposed = json.loads(capsys.readouterr().out)
    assert proposed["decision"] == "pending"
    cid = proposed["cid"]

    assert main(["--db", str(db_path), "inbox", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [item["cid"] for item in listed["items"]] == [cid]

    assert main(["--db", str(db_path), "inbox", "accept", cid, "--json"]) == 0
    accepted = json.loads(capsys.readouterr().out)
    assert accepted["decision"] == "accepted"
    assert accepted["accepted_rid"]

    store = Anamnesis(db_path)
    recalled = store.recall(
        "dashboard port 8765",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
    )
    assert recalled[0].record.rid == accepted["accepted_rid"]

    assert (
        main(
            [
                "--db",
                str(db_path),
                "inbox",
                "propose",
                "Temporary debug port is 54321.",
                "--owner",
                "primary",
                "--platform",
                "cli",
                "--json",
            ]
        )
        == 0
    )
    rejected_candidate = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "--db",
                str(db_path),
                "inbox",
                "reject",
                rejected_candidate["cid"],
                "--reason",
                "temporary task state",
                "--json",
            ]
        )
        == 0
    )
    rejected = json.loads(capsys.readouterr().out)
    assert rejected["decision"] == "rejected"
    assert rejected["review_reason"] == "temporary task state"

    old_item = store.propose_memory("Maybe stale pending memory should expire.", owner="primary")
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
                "inbox",
                "expire",
                "--max-age-days",
                "30",
                "--reason",
                "stale pending",
                "--json",
            ]
        )
        == 0
    )
    expired = json.loads(capsys.readouterr().out)
    assert expired["expired"][0]["cid"] == old_item.cid
    assert expired["expired"][0]["decision"] == "expired"

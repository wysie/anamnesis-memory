from __future__ import annotations

import json

from anamnesis import Anamnesis
from anamnesis.cli import main


def test_cli_audit_shows_memory_events_and_correction_chain(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    old = store.add_memory("Primary user prefers verbose updates.", owner="primary", domain="preference")
    replacement = store.correct_memory(
        old.rid,
        "Primary user prefers concise updates with next steps.",
        reason="user correction",
    )

    assert main(["--db", str(db_path), "audit", old.rid, "--json"]) == 0
    old_payload = json.loads(capsys.readouterr().out)
    assert old_payload["rid"] == old.rid
    assert old_payload["memory"]["status"] == "tombstoned"
    assert old_payload["correction_chain"]["replacement_rid"] == replacement.rid
    assert [event["event_type"] for event in old_payload["events"]] == [
        "memory_added",
        "memory_tombstoned",
        "memory_corrected_from",
    ]

    assert main(["--db", str(db_path), "audit", replacement.rid, "--json"]) == 0
    replacement_payload = json.loads(capsys.readouterr().out)
    assert replacement_payload["rid"] == replacement.rid
    assert replacement_payload["memory"]["status"] == "active"
    assert replacement_payload["correction_chain"]["old_rid"] == old.rid
    assert [event["event_type"] for event in replacement_payload["events"]] == [
        "memory_added",
        "memory_corrected_to",
    ]

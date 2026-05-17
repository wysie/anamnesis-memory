from __future__ import annotations

import json

from anamnesis import Anamnesis
from anamnesis.cli import main


def test_cli_correct_tombstones_and_replaces_memory(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    old = store.add_memory(
        "Primary user prefers verbose updates.",
        owner="primary",
        platform_scope="all",
        domain="preference",
        source="test",
        metadata={"source_platform": "whatsapp"},
    )

    assert (
        main(
            [
                "--db",
                str(db_path),
                "correct",
                old.rid,
                "Primary user prefers concise updates with next steps.",
                "--reason",
                "user correction",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    replacement = Anamnesis(db_path).get_memory(payload["replacement"]["rid"])
    assert payload["old"]["rid"] == old.rid
    assert payload["old"]["status"] == "tombstoned"
    assert payload["replacement"]["text"] == "Primary user prefers concise updates with next steps."
    assert replacement.metadata["corrects_rid"] == old.rid
    assert replacement.metadata["correction_reason"] == "user correction"

    recalled = Anamnesis(db_path).recall(
        "verbose concise updates next steps",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=10,
    )
    assert [result.record.rid for result in recalled] == [replacement.rid]

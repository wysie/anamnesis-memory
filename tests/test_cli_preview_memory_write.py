from __future__ import annotations

import json

from anamnesis import Anamnesis
from anamnesis.cli import main


def test_cli_preview_memory_write_previews_hermes_memory_tool_policy_without_writes(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"

    assert (
        main(
            [
                "--db",
                str(db_path),
                "preview-memory-write",
                "How are U considering what's good to store or not without an llm",
                "--target",
                "memory",
                "--origin",
                "background_review",
                "--owner",
                "primary",
                "--platform",
                "whatsapp",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "preview_memory_write"
    assert payload["input"]["source"] == "hermes_memory_tool"
    assert payload["input"]["origin"] == "background_review"
    assert payload["would_write"]["action"] == "reject"
    assert payload["would_write"]["reasons"]
    with Anamnesis(db_path)._connect() as conn:  # noqa: SLF001 - assert dry-run mutation boundary.
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM memory_inbox").fetchone()[0] == 0


def test_cli_preview_memory_write_apply_uses_same_policy_as_hermes_memory_tool(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"

    assert (
        main(
            [
                "--db",
                str(db_path),
                "preview-memory-write",
                "Primary user prefers concise updates with next steps.",
                "--target",
                "preference",
                "--origin",
                "background_review",
                "--owner",
                "primary",
                "--platform",
                "whatsapp",
                "--apply",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["would_write"]["action"] == "accept"
    assert payload["applied"]["kind"] == "memory"
    record = Anamnesis(db_path).get_memory(payload["applied"]["rid"])
    assert record.text == "Primary user prefers concise updates with next steps."
    assert record.domain == "preference"
    assert record.source == "hermes_memory_tool"
    assert record.metadata["source_platform"] == "whatsapp"
    assert record.metadata["origin"] == "background_review"
    assert record.metadata["preview_memory_write_applied"] is True

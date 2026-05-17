import json

from anamnesis import Anamnesis
from anamnesis.cli import main


def test_cli_shadow_turn_reports_write_decision_and_recall_without_mutating(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    stored = store.add_memory(
        "Primary user prefers cross-platform Anamnesis memory by default.",
        owner="primary",
        platform_scope="all",
        domain="preference",
        metadata={"source_platform": "whatsapp"},
    )

    assert (
        main(
            [
                "--db",
                str(db_path),
                "shadow-turn",
                "Primary user prefers Anamnesis shadow-mode logging.",
                "--owner",
                "primary",
                "--platform",
                "telegram",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "shadow"
    assert payload["would_write"]["action"] == "accept"
    assert payload["would_write"]["platform_scope"] == "all"
    assert payload["would_write"]["source_platform"] == "telegram"
    assert payload["would_inject"]["included"][0]["rid"] == stored.rid
    with store._connect() as conn:  # noqa: SLF001 - assert shadow-mode has no side effects.
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM memory_inbox").fetchone()[0] == 0


def test_cli_shadow_turn_reports_rejected_low_value_fragment(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"

    assert (
        main(
            [
                "--db",
                str(db_path),
                "shadow-turn",
                "Ok go ahead la",
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

    assert payload["would_write"]["action"] == "reject"
    assert payload["would_write"]["reasons"] == ["low_value_chat_fragment"]
    assert payload["would_inject"]["included"] == []

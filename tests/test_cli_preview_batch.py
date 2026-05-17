import json

from anamnesis import Anamnesis
from anamnesis.cli import main


def test_cli_preview_batch_dry_runs_jsonl_without_mutating(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    transcript_path = tmp_path / "transcript.jsonl"
    store = Anamnesis(db_path)
    existing = store.add_memory(
        "Primary user prefers cross-platform memory by default.",
        owner="primary",
        platform_scope="all",
        domain="preference",
    )
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "Ok go ahead la"}),
                json.dumps({"user": "Primary user prefers batch preview before dashboard."}),
                json.dumps({"text": "Maybe dashboard review should happen weekly."}),
            ]
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--db",
                str(db_path),
                "preview-batch",
                str(transcript_path),
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

    assert payload["mode"] == "preview_batch"
    assert payload["summary"] == {"total": 3, "accept": 1, "inbox": 1, "reject": 1}
    assert payload["reason_counts"]["low_value_chat_fragment"] == 1
    assert payload["reason_counts"]["durable_signal"] == 1
    assert payload["reason_counts"]["ambiguous_or_sensitive"] == 1
    assert payload["turns"][0]["would_write"]["action"] == "reject"
    assert payload["turns"][1]["would_write"]["platform_scope"] == "all"
    assert payload["turns"][1]["would_inject"]["included"][0]["rid"] == existing.rid
    with store._connect() as conn:  # noqa: SLF001 - assert dry-run has no side effects.
        assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM memory_inbox").fetchone()[0] == 0


def test_cli_preview_batch_apply_writes_accepts_and_inboxes(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    transcript_path = tmp_path / "transcript.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps({"text": "Primary user prefers batch preview before dashboard."}),
                json.dumps({"text": "Maybe dashboard review should happen weekly."}),
                json.dumps({"text": "Thanks"}),
            ]
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--db",
                str(db_path),
                "preview-batch",
                str(transcript_path),
                "--owner",
                "primary",
                "--platform",
                "telegram",
                "--apply",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    store = Anamnesis(db_path)

    assert payload["summary"] == {"total": 3, "accept": 1, "inbox": 1, "reject": 1}
    memories = store.recall(
        "batch preview dashboard",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=10,
    )
    assert [result.record.text for result in memories] == [
        "Primary user prefers batch preview before dashboard."
    ]
    assert memories[0].record.metadata["source_platform"] == "telegram"
    assert memories[0].record.platform_scope == "all"
    assert [item.proposed_text for item in store.inbox_items(decision="pending")] == [
        "Maybe dashboard review should happen weekly."
    ]

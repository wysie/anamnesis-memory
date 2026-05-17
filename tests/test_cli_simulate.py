import json

from anamnesis import Anamnesis
from anamnesis.cli import main


def test_cli_simulate_outputs_included_and_excluded(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    store.add_memory(
        "Primary user prefers local-only WhatsApp memory.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
    )
    store.add_memory(
        "Other owner WhatsApp memory secret.",
        owner="other",
        platform_scope="whatsapp",
        domain="privacy",
    )

    assert (
        main(
            [
                "--db",
                str(db_path),
                "simulate",
                "WhatsApp memory",
                "--owner",
                "primary",
                "--platform",
                "whatsapp",
                "--domain",
                "privacy",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["included"][0]["text"] == "Primary user prefers local-only WhatsApp memory."
    assert any("owner_mismatch" in item["exclusion_reasons"] for item in payload["excluded"])

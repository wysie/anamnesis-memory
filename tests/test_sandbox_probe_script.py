from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import os


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "anamnesis_sandbox_probe.py"


def test_sandbox_probe_script_direct_mode_creates_profile_shim_and_filters_junk(tmp_path):
    profile_dir = tmp_path / "fresh-anamnesis"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profile-dir",
            str(profile_dir),
            "--owner",
            "default",
            "--platform",
            "cli",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["ok"] is True
    assert payload["profile_dir"] == str(profile_dir)
    assert payload["provider"] == "anamnesis"
    assert payload["direct_probe"]["prefetch_contains_durable_rule"] is True
    assert payload["direct_probe"]["stored_texts"] == [
        "Primary user wants Anamnesis sandbox trials to stay CLI-only until explicitly approved for gateway."
    ]
    assert payload["direct_probe"]["active_texts"] == payload["direct_probe"]["stored_texts"]
    assert payload["direct_probe"]["inbox_texts"] == []
    assert payload["direct_probe"]["recall_query_count"] > 0
    assert (profile_dir / "plugins" / "anamnesis" / "__init__.py").exists()
    assert (profile_dir / "plugins" / "anamnesis" / "plugin.yaml").exists()
    assert (profile_dir / "anamnesis" / "anamnesis.db").exists()


def test_sandbox_probe_script_plain_output_is_human_readable(tmp_path):
    profile_dir = tmp_path / "fresh-anamnesis"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profile-dir",
            str(profile_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Anamnesis sandbox probe: PASS" in completed.stdout
    assert "stored durable rows: 1" in completed.stdout
    assert "gateway untouched" in completed.stdout


def test_sandbox_probe_script_cli_mode_uses_profile_env_and_checks_output(tmp_path):
    profile_dir = tmp_path / "fresh-anamnesis"
    calls_path = tmp_path / "hermes-call.txt"
    fake_hermes = tmp_path / "hermes"
    fake_hermes.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import os, sys\n"
        f"Path({str(calls_path)!r}).write_text('ARGS=' + repr(sys.argv[1:]) + '\\nDB=' + os.environ.get('ANAMNESIS_DB_PATH', '') + '\\nOWNER=' + os.environ.get('ANAMNESIS_OWNER', ''))\n"
        "print('session_id: fake-session')\n"
        "print('No — based on the recalled approval context, sandbox trials should stay CLI-only and should not touch the gateway now.')\n",
        encoding="utf-8",
    )
    fake_hermes.chmod(fake_hermes.stat().st_mode | 0o111)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--profile-dir",
            str(profile_dir),
            "--profile-name",
            "fresh-anamnesis",
            "--hermes-bin",
            str(fake_hermes),
            "--run-cli",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SCRIPT.parents[1] / "src")},
    )

    payload = json.loads(completed.stdout)
    call_text = calls_path.read_text(encoding="utf-8")

    assert payload["ok"] is True
    assert payload["cli_probe"]["ok"] is True
    assert payload["cli_probe"]["returncode"] == 0
    assert "stay CLI-only" in payload["cli_probe"]["stdout"]
    assert "'-p', 'fresh-anamnesis'" in call_text
    assert f"DB={profile_dir / 'anamnesis' / 'anamnesis.db'}" in call_text
    assert "OWNER=default" in call_text

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_model_smoke_script_has_safe_defaults():
    script = Path("scripts/anamnesis_model_smoke.py")
    content = script.read_text()

    assert "ANAMNESIS_ALLOW_MODEL_DOWNLOADS" in content
    assert "tempfile.TemporaryDirectory" in content
    assert "potion-base-2M" in content
    assert "~/.hermes" not in content


def test_model_smoke_script_json_mode_with_keyword_embedder(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path("src").resolve())
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/anamnesis_model_smoke.py",
            "--embedder",
            "keyword",
            "--json",
            "--db-path",
            str(tmp_path / "smoke.db"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["embedder"] == "keyword"
    assert payload["embedded"] == 2
    assert payload["top_reason"] in payload["reasons"]
    assert "semantic_match" in payload["reasons"]

from __future__ import annotations

import tomllib
from pathlib import Path


def test_model2vec_extra_is_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    extras = pyproject["project"]["optional-dependencies"]
    assert "model2vec" in extras
    assert any(dep.startswith("model2vec") for dep in extras["model2vec"])


def test_dev_extra_includes_model2vec_extra_for_ci_smoke():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    dev = pyproject["project"]["optional-dependencies"]["dev"]
    assert "model2vec>=0.8" in dev

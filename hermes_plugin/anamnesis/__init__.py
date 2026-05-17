from __future__ import annotations

import sys
from pathlib import Path

# Development plugin wrapper: make the repository package importable when this
# directory is symlinked into a Hermes profile's plugins/ directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from anamnesis.hermes_provider import AnamnesisMemoryProvider, register  # noqa: E402

__all__ = ["AnamnesisMemoryProvider", "register"]

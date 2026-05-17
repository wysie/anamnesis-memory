#!/usr/bin/env python3
"""Repeatable Anamnesis sandbox provider probe.

This script is intentionally self-contained and profile-scoped. It installs a
minimal memory-provider shim into a Hermes profile directory, pins the sandbox
Anamnesis DB path in that profile's .env, then runs a direct provider lifecycle
probe that does not touch the live gateway or other memory providers.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from anamnesis import Anamnesis  # noqa: E402
from anamnesis.hermes_provider import AnamnesisMemoryProvider  # noqa: E402

DURABLE_RULE = (
    "Primary user wants Anamnesis sandbox trials to stay CLI-only until explicitly "
    "approved for gateway."
)
JUNK_INPUTS = (
    "ok.",
    "yes go ahead",
    "[System note: The following is recalled memory context, NOT new user input.]",
)


def main() -> int:
    args = _parse_args()
    profile_dir = args.profile_dir.expanduser().resolve()
    db_path = profile_dir / "anamnesis" / "anamnesis.db"

    _install_profile_shim(profile_dir)
    _pin_profile_env(profile_dir, owner=args.owner, db_path=db_path)
    if args.reset_db and db_path.parent.exists():
        shutil.rmtree(db_path.parent)

    direct = _run_direct_probe(
        profile_dir=profile_dir,
        db_path=db_path,
        owner=args.owner,
        platform=args.platform,
    )
    cli_probe = None
    if args.run_cli:
        cli_probe = _run_cli_probe(
            hermes_bin=args.hermes_bin,
            profile_name=args.profile_name,
            db_path=db_path,
            owner=args.owner,
            timeout=args.cli_timeout,
        )
    payload: dict[str, Any] = {
        "ok": direct["ok"] and (cli_probe is None or cli_probe["ok"]),
        "provider": "anamnesis",
        "profile_dir": str(profile_dir),
        "profile_name": args.profile_name,
        "db_path": str(db_path),
        "gateway_touched": False,
        "direct_probe": direct,
    }
    if cli_probe is not None:
        payload["cli_probe"] = cli_probe

    if args.report_path:
        args.report_path.expanduser().write_text(
            _render_human(payload), encoding="utf-8"
        )

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_render_human(payload), end="")
    return 0 if payload["ok"] else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path.home() / ".hermes" / "profiles" / "fresh-anamnesis",
        help="Hermes profile directory to use for the sandbox probe.",
    )
    parser.add_argument("--owner", default="default")
    parser.add_argument("--platform", default="cli")
    parser.add_argument(
        "--profile-name",
        default="fresh-anamnesis",
        help="Hermes profile name to pass to `hermes -p` when --run-cli is set.",
    )
    parser.add_argument(
        "--hermes-bin",
        type=Path,
        default=Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes",
        help="Hermes executable for optional --run-cli probe.",
    )
    parser.add_argument(
        "--run-cli",
        action="store_true",
        help="Also run a real Hermes CLI one-shot against the sandbox profile.",
    )
    parser.add_argument(
        "--cli-timeout",
        type=int,
        default=300,
        help="Timeout in seconds for the optional Hermes CLI probe.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Optional path to write the human-readable probe report.",
    )
    parser.add_argument(
        "--no-reset-db",
        dest="reset_db",
        action="store_false",
        help="Do not clear the sandbox Anamnesis DB before probing.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.set_defaults(reset_db=True)
    return parser.parse_args()


def _install_profile_shim(profile_dir: Path) -> None:
    plugin_dir = profile_dir / "plugins" / "anamnesis"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "__init__.py").write_text(
        "\"\"\"Profile-local Anamnesis sandbox memory-provider shim.\"\"\"\n"
        "from __future__ import annotations\n\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        f"_REPO_SRC = Path({str(SRC_DIR)!r})\n"
        "if str(_REPO_SRC) not in sys.path:\n"
        "    sys.path.insert(0, str(_REPO_SRC))\n\n"
        "from anamnesis.hermes_provider import AnamnesisMemoryProvider  # noqa: E402\n\n\n"
        "def register(ctx):\n"
        "    ctx.register_memory_provider(AnamnesisMemoryProvider())\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: anamnesis\n"
        "description: Profile-local sandbox shim for the Anamnesis memory provider.\n"
        "version: 0.0.0-sandbox\n"
        "kind: exclusive\n",
        encoding="utf-8",
    )


def _pin_profile_env(profile_dir: Path, *, owner: str, db_path: Path) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    env_path = profile_dir / ".env"
    text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = text.splitlines()
    wanted = {
        "ANAMNESIS_OWNER": owner,
        "ANAMNESIS_DB_PATH": str(db_path),
    }
    existing_keys = {
        line.split("=", 1)[0]
        for line in lines
        if "=" in line and not line.lstrip().startswith("#")
    }
    if not any(line.strip() == "# Anamnesis sandbox memory provider" for line in lines):
        lines.extend(["", "# Anamnesis sandbox memory provider"])
    for key, value in wanted.items():
        if key in existing_keys:
            lines = [
                f"{key}={value}" if line.startswith(f"{key}=") else line
                for line in lines
            ]
        else:
            lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_direct_probe(
    *, profile_dir: Path, db_path: Path, owner: str, platform: str
) -> dict[str, Any]:
    os.environ["ANAMNESIS_OWNER"] = owner
    os.environ["ANAMNESIS_DB_PATH"] = str(db_path)

    provider = AnamnesisMemoryProvider()
    provider.initialize("sandbox-probe", hermes_home=str(profile_dir), platform=platform)

    for text in JUNK_INPUTS:
        provider.sync_turn(text, "assistant junk should not be stored")
    provider.sync_turn(DURABLE_RULE, "assistant junk should not be stored")
    provider.queue_prefetch("Anamnesis sandbox gateway approval", session_id="sandbox-probe")
    block = provider.prefetch("Anamnesis sandbox gateway approval", session_id="sandbox-probe")

    store = Anamnesis(db_path)
    rows = store.recall(
        "ok yes go ahead system note Anamnesis sandbox gateway approval",
        owner=owner,
        platform=platform,
        allowed_visibility={"private"},
        limit=10,
    )
    recalled_texts = [row.record.text for row in rows]
    with store._connect() as conn:  # noqa: SLF001 - sandbox probe verifies DB side effects.
        active_texts = [
            str(row["text"])
            for row in conn.execute("SELECT text FROM memories WHERE status='active' ORDER BY created_at")
        ]
        inbox_texts = [
            str(row["proposed_text"])
            for row in conn.execute("SELECT proposed_text FROM memory_inbox ORDER BY created_at")
        ]
        recall_query_count = int(
            conn.execute("SELECT COUNT(*) FROM audit_log WHERE event_type='recall_query'").fetchone()[0]
        )
    inspected_text = "\n".join(recalled_texts + active_texts + inbox_texts + [block])
    junk_leaked = any(junk in inspected_text for junk in JUNK_INPUTS)
    prefetch_contains = DURABLE_RULE in block
    only_durable_active = active_texts == [DURABLE_RULE]
    only_durable_recalled = recalled_texts == [DURABLE_RULE]
    return {
        "ok": bool(
            provider.is_available()
            and prefetch_contains
            and only_durable_active
            and only_durable_recalled
            and not junk_leaked
            and recall_query_count > 0
        ),
        "provider_available": provider.is_available(),
        "prefetch_contains_durable_rule": prefetch_contains,
        "prefetch_block": block,
        "stored_texts": recalled_texts,
        "active_texts": active_texts,
        "inbox_texts": inbox_texts,
        "recall_query_count": recall_query_count,
        "junk_leaked": junk_leaked,
    }


def _run_cli_probe(
    *, hermes_bin: Path, profile_name: str, db_path: Path, owner: str, timeout: int
) -> dict[str, Any]:
    env = os.environ.copy()
    env["ANAMNESIS_OWNER"] = owner
    env["ANAMNESIS_DB_PATH"] = str(db_path)
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    query = (
        "Use anamnesis_recall for this exact query: Anamnesis sandbox gateway approval. "
        "Then answer in one sentence: should sandbox trials touch gateway now?"
    )
    command = [
        str(hermes_bin.expanduser()),
        "-p",
        profile_name,
        "chat",
        "-Q",
        "-q",
        query,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "command": command,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "failure": "timeout",
        }
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    low_stdout = stdout.lower()
    expected = (
        "stay cli-only" in low_stdout
        and "not touch" in low_stdout
        and "gateway" in low_stdout
    )
    return {
        "ok": completed.returncode == 0 and expected,
        "command": command,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "expected_answer_seen": expected,
    }


def _render_human(payload: dict[str, Any]) -> str:
    status = "PASS" if payload["ok"] else "FAIL"
    direct = payload["direct_probe"]
    lines = [
        f"Anamnesis sandbox probe: {status}",
        f"profile: {payload['profile_dir']}",
        f"db: {payload['db_path']}",
        f"provider available: {direct['provider_available']}",
        f"stored durable rows: {len(direct['stored_texts'])}",
        f"junk leaked: {direct['junk_leaked']}",
        "gateway untouched: true",
    ]
    cli_probe = payload.get("cli_probe")
    if cli_probe:
        lines.extend(
            [
                f"cli probe: {'PASS' if cli_probe['ok'] else 'FAIL'}",
                f"cli returncode: {cli_probe['returncode']}",
                "--- cli stdout ---",
                cli_probe.get("stdout") or "<empty>",
            ]
        )
        if cli_probe.get("stderr"):
            lines.extend(["--- cli stderr ---", cli_probe["stderr"]])
    lines.extend(["--- prefetch block ---", direct["prefetch_block"] or "<empty>"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

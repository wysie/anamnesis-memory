#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from anamnesis import Anamnesis
from anamnesis.embedding_models import load_embedder_from_env
from anamnesis.embeddings import KeywordEmbedder


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Anamnesis embedding recall in an isolated DB.")
    parser.add_argument("--embedder", default="potion-base-2M", help="official model name, local path, or 'keyword'")
    parser.add_argument("--db-path", default="", help="optional sandbox DB path; default uses a temp dir")
    parser.add_argument("--allow-download", action="store_true", help="allow first-use official model download")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    if args.allow_download:
        os.environ["ANAMNESIS_ALLOW_MODEL_DOWNLOADS"] = "1"

    with tempfile.TemporaryDirectory(prefix="anamnesis-model-smoke-") as tmp:
        db_path = Path(args.db_path) if args.db_path else Path(tmp) / "smoke.db"
        store = Anamnesis(db_path)
        if args.embedder == "keyword":
            embedder = KeywordEmbedder(
                dimensions=("car", "vehicle", "wash", "clean", "hope", "household"),
                synonyms={"vehicle": "car", "clean": "wash"},
            )
            embedder_label = "keyword"
        else:
            os.environ["ANAMNESIS_EMBEDDER"] = args.embedder
            embedder = load_embedder_from_env()
            if embedder is None:
                raise SystemExit("ANAMNESIS_EMBEDDER resolved to no embedder")
            embedder_label = args.embedder

        target = store.add_memory(
            "Collaborator handles car washing at the house.",
            owner="primary",
            platform_scope="whatsapp",
            domain="household",
        )
        store.add_memory(
            "The study room was busy in the afternoon.",
            owner="primary",
            platform_scope="whatsapp",
            domain="household",
        )
        report = store.embed_missing(embedder)
        results = store.recall(
            "Who cleans the vehicle?",
            owner="primary",
            platform="whatsapp",
            allowed_visibility={"private"},
            limit=3,
            embedder=embedder,
        )
        top = results[0] if results else None
        payload: dict[str, Any] = {
            "ok": bool(top and top.record.rid == target.rid and "semantic_match" in top.reasons),
            "db_path": str(db_path),
            "embedder": embedder_label,
            "model_id": getattr(embedder, "model_id", embedder_label),
            "dimension": getattr(embedder, "dimension", None),
            "embedded": report["embedded"],
            "skipped": report["skipped"],
            "top_text": top.record.text if top else "",
            "top_score": top.score if top else 0.0,
            "top_reason": top.reasons[0] if top and top.reasons else "",
            "reasons": top.reasons if top else [],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Anamnesis model smoke: {'PASS' if payload['ok'] else 'FAIL'}")
            print(f"embedder: {payload['embedder']}")
            print(f"model_id: {payload['model_id']}")
            print(f"dimension: {payload['dimension']}")
            print(f"embedded: {payload['embedded']} skipped: {payload['skipped']}")
            print(f"top: {payload['top_text']}")
            print(f"reasons: {', '.join(payload['reasons'])}")
        return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

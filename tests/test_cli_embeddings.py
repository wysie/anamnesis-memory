from __future__ import annotations

import json
import tomllib
from pathlib import Path

from anamnesis import Anamnesis
from anamnesis.cli import main
from anamnesis.embedding_models import OFFICIAL_EMBEDDING_MODELS


def test_package_declares_anamnesis_console_script():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["anamnesis"] == "anamnesis.cli:main"


def test_embeddings_switch_persists_active_model_and_reports_missing_backfill(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    store.add_memory("Primary user prefers local-first memory.", owner="primary", platform_scope="cli")
    store.add_memory("Anamnesis uses governed recall.", owner="primary", platform_scope="cli")

    exit_code = main(["--db", str(db_path), "embeddings", "switch", "potion-base-32M"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "active_model=potion-base-32M" in output
    assert Anamnesis(db_path).active_embedding_model() == "potion-base-32M"

    exit_code = main(["--db", str(db_path), "embeddings", "status", "--json"])
    status = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert status["active_model"] == "potion-base-32M"
    assert status["dimension"] == 512
    assert status["total_active"] == 2
    assert status["embedded"] == 0
    assert status["missing"] == 2
    assert status["backfill_required"] is True
    assert status["fts_fallback"] is True


def test_embeddings_backfill_uses_active_model_coverage(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    store.add_memory("Primary user prefers local-first memory.", owner="primary", platform_scope="cli")
    store.set_active_embedding_model("potion-base-2M")

    exit_code = main([
        "--db",
        str(db_path),
        "embeddings",
        "backfill",
        "--model",
        "potion-base-2M",
        "--test-keyword-embedder",
    ])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["model"] == "potion-base-2M"
    assert report["before"]["missing"] == 1
    assert report["embedded"] == 1
    assert report["after"]["missing"] == 0


def test_recall_config_persists_defaults_and_recall_explain_uses_them(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    index_db = tmp_path / "vectors.db"
    store = Anamnesis(db_path)
    target = store.add_memory(
        "Helper handles car washing and pool maintenance.",
        owner="primary",
        platform_scope="whatsapp",
        domain="household",
    )
    store.set_active_embedding_model("potion-base-2M")
    assert main([
        "--db",
        str(db_path),
        "embeddings",
        "backfill",
        "--model",
        "potion-base-2M",
        "--test-keyword-embedder",
    ]) == 0
    capsys.readouterr()
    assert main([
        "--db",
        str(db_path),
        "embeddings",
        "index-rebuild",
        "--model",
        "potion-base-2M",
        "--index-db",
        str(index_db),
    ]) == 0
    capsys.readouterr()

    assert main([
        "--db",
        str(db_path),
        "recall-config",
        "set",
        "--model",
        "potion-base-2M",
        "--ann-backend",
        "sqlite-vec",
        "--index-db",
        str(index_db),
        "--recall-policy",
        "recall_first",
        "--ann-candidate-limit",
        "10",
        "--ann-min-keyword-candidates",
        "3",
        "--vector-candidate-limit",
        "1000",
        "--json",
    ]) == 0
    config = json.loads(capsys.readouterr().out)
    assert config["model"] == "potion-base-2M"
    assert config["recall_policy"] == "recall_first"
    assert config["ann_candidate_limit"] == 10
    assert Anamnesis(db_path).recall_config()["ann_candidate_limit"] == "10"

    assert main([
        "--db",
        str(db_path),
        "recall",
        "Who handles car washing?",
        "--owner",
        "primary",
        "--platform",
        "whatsapp",
        "--test-keyword-embedder",
        "--explain",
        "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["config"]["recall_policy"] == "recall_first"
    assert payload["config"]["ann_candidate_limit"] == 10
    assert payload["explain"]["ann_searched"] is True
    assert payload["results"][0]["rid"] == target.rid
    assert "ann_match" in payload["results"][0]["reasons"]


def test_synthesis_config_cli_persists_own_local_llm_endpoint(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"

    assert main([
        "--db",
        str(db_path),
        "synthesis-config",
        "set",
        "--base-url",
        "http://127.0.0.1:8060/v1",
        "--model",
        "local-private-model",
        "--api-key-env",
        "ANAMNESIS_LLM_API_KEY",
        "--temperature",
        "0",
        "--max-tokens",
        "512",
        "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["base_url"] == "http://127.0.0.1:8060/v1"
    assert payload["model"] == "local-private-model"
    assert payload["api_key_env"] == "ANAMNESIS_LLM_API_KEY"
    assert Anamnesis(db_path).synthesis_config()["model"] == "local-private-model"


def test_embeddings_sqlite_vec_index_status_and_rebuild(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    index_db = tmp_path / "vectors.db"
    store = Anamnesis(db_path)
    record = store.add_memory("Primary user prefers local-first memory.", owner="primary", platform_scope="cli")
    store.set_active_embedding_model("potion-base-2M")
    assert main([
        "--db",
        str(db_path),
        "embeddings",
        "backfill",
        "--model",
        "potion-base-2M",
        "--test-keyword-embedder",
    ]) == 0
    capsys.readouterr()

    assert main([
        "--db",
        str(db_path),
        "embeddings",
        "index-status",
        "--model",
        "potion-base-2M",
        "--index-db",
        str(index_db),
        "--json",
    ]) == 0
    before = json.loads(capsys.readouterr().out)
    assert before["backend"] == "sqlite-vec"
    assert before["embedded"] == 1
    assert before["indexed"] == 0
    assert before["rebuild_required"] is True

    assert main([
        "--db",
        str(db_path),
        "embeddings",
        "index-rebuild",
        "--model",
        "potion-base-2M",
        "--index-db",
        str(index_db),
    ]) == 0
    rebuilt = json.loads(capsys.readouterr().out)
    assert rebuilt["indexed"] == 1
    assert rebuilt["skipped"] == 0

    assert main([
        "--db",
        str(db_path),
        "embeddings",
        "index-status",
        "--model",
        "potion-base-2M",
        "--index-db",
        str(index_db),
        "--json",
    ]) == 0
    after = json.loads(capsys.readouterr().out)
    assert after["indexed"] == 1
    assert after["missing"] == 0
    assert after["stale"] == 0
    assert after["rebuild_required"] is False

    Anamnesis(db_path).tombstone(record.rid, reason="metadata stale test")
    assert main([
        "--db",
        str(db_path),
        "embeddings",
        "index-status",
        "--model",
        "potion-base-2M",
        "--index-db",
        str(index_db),
        "--json",
    ]) == 0
    stale = json.loads(capsys.readouterr().out)
    assert stale["indexed"] == 1
    assert stale["stale"] == 1
    assert stale["rebuild_required"] is True


def test_embeddings_switch_accepts_retrieval_32m_as_active_model(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    store.add_memory("Retrieval fine tune should be selectable.", owner="primary", platform_scope="cli")

    exit_code = main(["--db", str(db_path), "embeddings", "switch", "potion-retrieval-32M"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "active_model=potion-retrieval-32M" in output
    assert Anamnesis(db_path).active_embedding_model() == "potion-retrieval-32M"

    exit_code = main(["--db", str(db_path), "embeddings", "status", "--json"])
    status = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert status["active_model"] == "potion-retrieval-32M"
    assert status["dimension"] == 512


def test_embeddings_benchmark_compares_models_without_enabling_cascade(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"

    exit_code = main([
        "--db",
        str(db_path),
        "embeddings",
        "benchmark",
        "--models",
        "potion-base-2M,potion-base-8M,potion-base-32M",
        "--test-keyword-embedder",
        "--synthetic-count",
        "25",
        "--include-adversarial",
        "--ann-candidate-limit",
        "10",
        "--ann-backend",
        "sqlite-vec",
        "--recall-policy",
        "recall_first",
        "--ann-min-keyword-candidates",
        "7",
        "--json",
    ])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["cascade"] is False
    assert report["synthetic_count"] == 25
    assert report["include_adversarial"] is True
    assert report["vector_candidate_limit"] == 1000
    assert report["ann_candidate_limit"] == 10
    assert report["ann_backend"] == "sqlite-vec"
    assert report["recall_policy"] == "recall_first"
    assert report["ann_min_keyword_candidates"] == 7
    assert report["models"] == ["potion-base-2M", "potion-base-8M", "potion-base-32M"]
    assert len(report["results"]) == 3
    assert {result["dimension"] for result in report["results"]} == {64, 256, 512}
    assert all(result["embedded"] >= 29 for result in report["results"])
    assert all(result["db_size_bytes"] > 0 for result in report["results"])
    assert all(result["backfill_memories_per_second"] >= 0 for result in report["results"])
    assert all(result["backfill_seconds"] >= 0 for result in report["results"])
    assert all(result["recall_p50_ms"] >= 0 for result in report["results"])
    assert all(result["ann_rebuild_seconds"] >= 0 for result in report["results"])
    assert all(result["ann_indexed"] >= result["embedded"] for result in report["results"])
    assert all(result["score"] >= 0 for result in report["results"])


def test_embeddings_benchmark_default_models_include_retrieval_finetune(tmp_path, capsys):
    db_path = tmp_path / "anamnesis.db"

    exit_code = main([
        "--db",
        str(db_path),
        "embeddings",
        "benchmark",
        "--test-keyword-embedder",
        "--synthetic-count",
        "2",
        "--json",
    ])
    report = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert report["models"] == [
        "potion-base-2M",
        "potion-base-8M",
        "potion-base-32M",
        "potion-retrieval-32M",
    ]


def test_official_model_specs_have_stable_model_ids():
    model_ids = {spec.model_id for spec in OFFICIAL_EMBEDDING_MODELS.values()}

    assert len(model_ids) == len(OFFICIAL_EMBEDDING_MODELS)
    assert all(":" in model_id for model_id in model_ids)

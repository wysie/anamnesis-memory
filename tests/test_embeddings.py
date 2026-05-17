from __future__ import annotations

import sqlite3

import pytest

from anamnesis import Anamnesis
from anamnesis.embeddings import KeywordEmbedder
from anamnesis.vector_index import (
    ExactVectorIndex,
    SQLiteVecVectorIndex,
    VectorIndexRow,
    sqlite_vec_available,
)


def test_recall_falls_back_to_fts_without_embedder(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    record = store.add_memory(
        "Primary user prefers local-first governed memory systems.",
        owner="primary",
        platform_scope="cli",
        domain="preference",
    )

    recalled = store.recall(
        "local-first governed memory",
        owner="primary",
        platform="cli",
        allowed_visibility={"private"},
        limit=5,
    )

    assert recalled[0].record.rid == record.rid
    assert "keyword_match" in recalled[0].reasons
    assert "semantic_match" not in recalled[0].reasons


def test_embed_missing_and_vector_recall_handles_paraphrase(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(
        dimensions=("car", "vehicle", "wash", "clean", "hope", "household"),
        synonyms={"vehicle": "car", "clean": "wash"},
    )
    record = store.add_memory(
        "Collaborator handles car washing at the house.",
        owner="primary",
        platform_scope="whatsapp",
        domain="household",
    )

    report = store.embed_missing(embedder)
    recalled = store.recall(
        "Who cleans the vehicle?",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=5,
        embedder=embedder,
    )

    assert report == {"embedded": 1, "skipped": 0}
    assert recalled[0].record.rid == record.rid
    assert "semantic_match" in recalled[0].reasons


def test_semantic_recall_expands_low_overlap_intent_paraphrases(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(
        dimensions=("child", "asleep", "quiet", "audio", "automation", "voice"),
    )
    target = store.add_memory(
        "When the child is asleep, household automation replies should be quiet and avoid unnecessary audio.",
        owner="primary",
        platform_scope="whatsapp",
        domain="semantic_hard",
    )
    store.add_memory(
        "Voice replies are useful when the user sends a voice message.",
        owner="primary",
        platform_scope="whatsapp",
        domain="semantic_hard",
    )
    store.add_memory(
        "Automation workflows can turn on lights during morning routines.",
        owner="primary",
        platform_scope="whatsapp",
        domain="semantic_hard",
    )
    store.embed_missing(embedder)

    recalled = store.recall(
        "if the little one is sleeping how should automation respond",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        domain="semantic_hard",
        limit=3,
        embedder=embedder,
        recall_policy="semantic_only",
    )

    assert recalled[0].record.rid == target.rid
    assert "semantic_intent_expansion" in recalled[0].reasons


def test_identifier_config_recall_allows_host_server_queries(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    record = store.add_memory(
        "The recall worker runs on 10.30.0.12 and calls the model server at 10.30.0.13.",
        owner="bench",
        platform_scope="cli",
        domain="identifier",
    )

    recalled = store.recall(
        "which host does recall worker call for model server",
        owner="bench",
        platform="cli",
        allowed_visibility={"private"},
        domain="identifier",
        limit=5,
    )

    assert recalled
    assert recalled[0].record.rid == record.rid
    assert "10.30.0.12" in recalled[0].record.text
    assert "10.30.0.13" in recalled[0].record.text


def test_identifier_config_recall_allows_current_version_queries(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    record = store.add_memory(
        "The public billing API version is 2026-02-15.",
        owner="bench",
        platform_scope="cli",
        domain="identifier",
    )

    recalled = store.recall(
        "current public billing API version",
        owner="bench",
        platform="cli",
        allowed_visibility={"private"},
        domain="identifier",
        limit=5,
    )

    assert recalled
    assert recalled[0].record.rid == record.rid


def test_project_recall_allows_test_command_queries(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    record = store.add_memory(
        "The server package full test command is pytest tests/server -q.",
        owner="bench",
        platform_scope="cli",
        domain="project",
    )

    recalled = store.recall(
        "full test command for server package",
        owner="bench",
        platform="cli",
        allowed_visibility={"private"},
        domain="project",
        limit=5,
    )

    assert recalled
    assert recalled[0].record.rid == record.rid


def test_vector_recall_cannot_bypass_scope_or_tombstones(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(
        dimensions=("token", "orchid", "vault", "secret"),
        synonyms={"secret": "token"},
    )
    private = store.add_memory(
        "Primary user's private token label is sample-vault.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
    )
    helper = store.add_memory(
        "Collaborator can ask questions but cannot control devices.",
        owner="hope",
        platform_scope="whatsapp",
        domain="permissions",
    )
    obsolete = store.add_memory(
        "Collaborator secret device token is old-sample-vault.",
        owner="hope",
        platform_scope="whatsapp",
        domain="privacy",
    )
    store.tombstone(obsolete.rid, reason="obsolete")
    store.embed_missing(embedder)

    recalled = store.recall(
        "secret orchid vault token",
        owner="hope",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=10,
        embedder=embedder,
    )
    recalled_ids = {result.record.rid for result in recalled}

    assert private.rid not in recalled_ids
    assert obsolete.rid not in recalled_ids
    assert helper.rid not in recalled_ids


def test_embedding_schema_is_idempotent_for_existing_database(tmp_path):
    db_path = tmp_path / "anamnesis.db"
    Anamnesis(db_path).add_memory("Dashboard runs on port 8765.")

    store = Anamnesis(db_path)
    report = store.embed_missing(KeywordEmbedder(dimensions=("dashboard", "8765")))

    assert report == {"embedded": 1, "skipped": 0}


def test_legacy_single_model_embedding_table_migrates_without_memory_migration(tmp_path):
    db_path = tmp_path / "anamnesis.db"
    store = Anamnesis(db_path)
    record = store.add_memory("Primary user prefers local-first memory.", owner="primary", platform_scope="cli")
    legacy = KeywordEmbedder(dimensions=("memory",), name="potion-2m")
    store.embed_missing(legacy)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            ALTER TABLE memory_embeddings RENAME TO memory_embeddings_new;
            CREATE TABLE memory_embeddings (
                rid TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                vector_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            INSERT INTO memory_embeddings
            SELECT rid, model_id, dimension, vector_json, created_at, updated_at
            FROM memory_embeddings_new;
            DROP TABLE memory_embeddings_new;
            """
        )

    migrated = Anamnesis(db_path)
    upgraded = KeywordEmbedder(dimensions=("memory", "local"), name="potion-32m")

    assert migrated.get_memory(record.rid).text == "Primary user prefers local-first memory."
    assert migrated.embedding_status(legacy)["embedded"] == 1
    assert migrated.embed_missing(upgraded) == {"embedded": 1, "skipped": 0}
    assert migrated.embedding_status(legacy)["embedded"] == 1
    assert migrated.embedding_status(upgraded)["embedded"] == 1


def test_different_dimension_embeddings_coexist_per_model(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    record = store.add_memory(
        "Collaborator handles automobile maintenance.",
        owner="primary",
        platform_scope="whatsapp",
        domain="household",
    )
    small = KeywordEmbedder(
        dimensions=("automobile",), synonyms={"car": "automobile"}, name="potion-2m"
    )
    large = KeywordEmbedder(
        dimensions=("automobile", "maintenance", "hope"),
        synonyms={"car": "automobile"},
        name="potion-32m",
    )

    assert store.embed_missing(small) == {"embedded": 1, "skipped": 0}
    assert store.embed_missing(large) == {"embedded": 1, "skipped": 0}

    recalled_with_small = store.recall(
        "car",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        embedder=small,
    )
    recalled_with_large = store.recall(
        "car",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        embedder=large,
    )

    assert recalled_with_small[0].record.rid == record.rid
    assert recalled_with_large[0].record.rid == record.rid


def test_embedding_status_is_per_active_model_and_backfillable(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    store.add_memory("Primary user prefers local-first memory.", owner="primary", platform_scope="cli")
    store.add_memory("Anamnesis uses governed recall.", owner="primary", platform_scope="cli")
    small = KeywordEmbedder(dimensions=("memory",), name="potion-2m")
    large = KeywordEmbedder(dimensions=("memory", "anamnesis"), name="potion-32m")

    store.embed_missing(small)

    assert store.embedding_status(small) == {
        "model_id": small.model_id,
        "dimension": 1,
        "total_active": 2,
        "embedded": 2,
        "missing": 0,
    }
    assert store.embedding_status(large) == {
        "model_id": large.model_id,
        "dimension": 2,
        "total_active": 2,
        "embedded": 0,
        "missing": 2,
    }

    assert store.embed_missing(large) == {"embedded": 2, "skipped": 0}
    assert store.embedding_status(large)["missing"] == 0


def test_vector_recall_can_prune_to_keyword_candidates(tmp_path, monkeypatch):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(dimensions=("needle", "haystack", "noise"))
    target = store.add_memory(
        "Needle memory should be found quickly.",
        owner="primary",
        platform_scope="cli",
        domain="benchmark",
    )
    for idx in range(20):
        store.add_memory(
            f"Haystack distractor {idx} with noise only.",
            owner="primary",
            platform_scope="cli",
            domain="benchmark",
        )
    store.embed_missing(embedder)
    calls = {"count": 0}

    def counting_cosine(left, right):
        calls["count"] += 1
        return sum(a * b for a, b in zip(left, right, strict=True))

    monkeypatch.setattr("anamnesis.core.cosine_similarity", counting_cosine)

    recalled = store.recall(
        "needle",
        owner="primary",
        platform="cli",
        allowed_visibility={"private"},
        embedder=embedder,
        vector_candidate_limit=5,
    )

    assert recalled[0].record.rid == target.rid
    assert calls["count"] <= 5


def test_ann_index_recovers_semantic_matches_when_keyword_pruning_has_no_candidates(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(
        dimensions=("car", "wash", "hope"),
        synonyms={"vehicle": "car", "cleans": "wash", "cleaning": "wash"},
    )
    target = store.add_memory(
        "Collaborator handles car washing at the house.",
        owner="primary",
        platform_scope="whatsapp",
        domain="household",
    )
    store.embed_missing(embedder)
    index = ExactVectorIndex(model_id=embedder.model_id, dimension=embedder.dimension)
    store.rebuild_vector_index(embedder, index)

    recalled = store.recall(
        "Who cleans the vehicle?",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=5,
        embedder=embedder,
        vector_candidate_limit=1000,
        vector_index=index,
        ann_candidate_limit=10,
    )

    assert recalled[0].record.rid == target.rid
    assert "ann_match" in recalled[0].reasons
    assert "semantic_match" in recalled[0].reasons


def test_latency_first_policy_skips_ann_when_keyword_candidates_are_sufficient(tmp_path, monkeypatch):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(dimensions=("needle", "other"))
    target = store.add_memory(
        "Needle memory should use the cheap keyword-pruned path.",
        owner="primary",
        platform_scope="cli",
        domain="benchmark",
    )
    store.embed_missing(embedder)
    index = ExactVectorIndex(model_id=embedder.model_id, dimension=embedder.dimension)
    store.rebuild_vector_index(embedder, index)
    calls = {"search": 0}

    def counting_search(query_vector, *, top_k, **kwargs):
        calls["search"] += 1
        return []

    monkeypatch.setattr(index, "search", counting_search)

    recalled = store.recall(
        "needle",
        owner="primary",
        platform="cli",
        allowed_visibility={"private"},
        limit=5,
        embedder=embedder,
        vector_candidate_limit=1000,
        vector_index=index,
        ann_candidate_limit=10,
        recall_policy="latency_first",
        ann_min_keyword_candidates=1,
    )

    assert recalled[0].record.rid == target.rid
    assert calls["search"] == 0


def test_recall_first_policy_always_merges_ann_candidates(tmp_path, monkeypatch):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(
        dimensions=("needle", "automobile"), synonyms={"vehicle": "automobile"}
    )
    keyword = store.add_memory(
        "Needle keyword memory.",
        owner="primary",
        platform_scope="cli",
        domain="benchmark",
    )
    semantic = store.add_memory(
        "Automobile semantic memory.",
        owner="primary",
        platform_scope="cli",
        domain="benchmark",
    )
    store.embed_missing(embedder)
    index = ExactVectorIndex(model_id=embedder.model_id, dimension=embedder.dimension)
    index.build([(semantic.rid, embedder.embed("vehicle"))])
    calls = {"search": 0}
    original_search = index.search

    def counting_search(query_vector, *, top_k, **kwargs):
        calls["search"] += 1
        return original_search(query_vector, top_k=top_k, **kwargs)

    monkeypatch.setattr(index, "search", counting_search)

    recalled = store.recall(
        "needle vehicle",
        owner="primary",
        platform="cli",
        allowed_visibility={"private"},
        limit=5,
        embedder=embedder,
        vector_candidate_limit=1000,
        vector_index=index,
        ann_candidate_limit=10,
        recall_policy="recall_first",
        ann_min_keyword_candidates=1,
    )
    recalled_ids = {result.record.rid for result in recalled}

    assert keyword.rid in recalled_ids
    assert semantic.rid in recalled_ids
    assert calls["search"] == 1
    assert any("ann_match" in result.reasons for result in recalled if result.record.rid == semantic.rid)


def test_semantic_only_policy_uses_ann_without_returning_fts_only_results(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(
        dimensions=("needle", "automobile"), synonyms={"vehicle": "automobile"}
    )
    keyword = store.add_memory(
        "Needle keyword-only memory.",
        owner="primary",
        platform_scope="cli",
        domain="benchmark",
    )
    semantic = store.add_memory(
        "Automobile semantic memory.",
        owner="primary",
        platform_scope="cli",
        domain="benchmark",
    )
    store.embed_missing(embedder)
    index = ExactVectorIndex(model_id=embedder.model_id, dimension=embedder.dimension)
    index.build([(semantic.rid, embedder.embed("vehicle"))])

    recalled = store.recall(
        "needle vehicle",
        owner="primary",
        platform="cli",
        allowed_visibility={"private"},
        limit=5,
        embedder=embedder,
        vector_candidate_limit=1000,
        vector_index=index,
        ann_candidate_limit=10,
        recall_policy="semantic_only",
    )
    recalled_ids = {result.record.rid for result in recalled}

    assert semantic.rid in recalled_ids
    assert keyword.rid not in recalled_ids


def test_ann_candidates_cannot_bypass_scope_or_tombstones(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(
        dimensions=("token", "orchid", "vault"),
        synonyms={"secret": "token"},
    )
    private = store.add_memory(
        "Primary user's private token label is sample-vault.",
        owner="primary",
        platform_scope="whatsapp",
        domain="privacy",
    )
    obsolete = store.add_memory(
        "Collaborator secret token label is sample-vault.",
        owner="hope",
        platform_scope="whatsapp",
        domain="privacy",
    )
    store.tombstone(obsolete.rid, reason="obsolete")
    store.embed_missing(embedder)
    index = ExactVectorIndex(model_id=embedder.model_id, dimension=embedder.dimension)
    store.rebuild_vector_index(embedder, index)

    recalled = store.recall(
        "secret orchid vault token",
        owner="hope",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=10,
        embedder=embedder,
        vector_candidate_limit=1000,
        vector_index=index,
        ann_candidate_limit=10,
    )
    recalled_ids = {result.record.rid for result in recalled}

    assert private.rid not in recalled_ids
    assert obsolete.rid not in recalled_ids
    assert recalled_ids == set()


@pytest.mark.skipif(not sqlite_vec_available(), reason="sqlite-vec is not installed")
def test_sqlite_vec_vector_index_searches_persisted_vectors(tmp_path):
    index = SQLiteVecVectorIndex(
        db_path=tmp_path / "vectors.db",
        model_id="test-model",
        dimension=3,
    )
    index.build([
        ("alpha", [1.0, 0.0, 0.0]),
        ("beta", [0.0, 1.0, 0.0]),
        ("bad-dimension", [1.0, 0.0]),
    ])

    recalled = index.search([1.0, 0.0, 0.0], top_k=2)

    assert recalled[0][0] == "alpha"
    assert recalled[0][1] > recalled[1][1]
    assert {rid for rid, _score in recalled} == {"alpha", "beta"}


@pytest.mark.skipif(not sqlite_vec_available(), reason="sqlite-vec is not installed")
def test_sqlite_vec_vector_index_prefilters_metadata(tmp_path):
    index = SQLiteVecVectorIndex(
        db_path=tmp_path / "vectors.db",
        model_id="test-model",
        dimension=3,
    )
    index.build([
        VectorIndexRow(
            rid="target",
            vector=[1.0, 0.0, 0.0],
            owner="primary",
            visibility="private",
            platform_scope="whatsapp,telegram",
            status="active",
            domain="household",
        ),
        VectorIndexRow(
            rid="wrong-owner",
            vector=[1.0, 0.0, 0.0],
            owner="hope",
            visibility="private",
            platform_scope="whatsapp",
            status="active",
            domain="household",
        ),
        VectorIndexRow(
            rid="wrong-platform",
            vector=[1.0, 0.0, 0.0],
            owner="primary",
            visibility="private",
            platform_scope="telegram",
            status="active",
            domain="household",
        ),
        VectorIndexRow(
            rid="tombstone",
            vector=[1.0, 0.0, 0.0],
            owner="primary",
            visibility="private",
            platform_scope="whatsapp",
            status="tombstoned",
            domain="household",
        ),
        VectorIndexRow(
            rid="wrong-domain",
            vector=[1.0, 0.0, 0.0],
            owner="primary",
            visibility="private",
            platform_scope="whatsapp",
            status="active",
            domain="work",
        ),
    ])

    recalled = index.search(
        [1.0, 0.0, 0.0],
        top_k=10,
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        domain="household",
        status="active",
    )

    assert recalled == [("target", 1.0)]


@pytest.mark.skipif(not sqlite_vec_available(), reason="sqlite-vec is not installed")
def test_rebuild_vector_index_stores_governance_metadata_for_prefiltering(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(dimensions=("car", "wash"), synonyms={"vehicle": "car"})
    target = store.add_memory(
        "Collaborator handles car washing.",
        owner="primary",
        platform_scope="whatsapp",
        domain="household",
    )
    wrong_owner = store.add_memory(
        "Collaborator handles vehicle washing for another owner.",
        owner="hope",
        platform_scope="whatsapp",
        domain="household",
    )
    store.tombstone(wrong_owner.rid, reason="obsolete")
    store.embed_missing(embedder)
    index = SQLiteVecVectorIndex(
        db_path=tmp_path / "vectors.db",
        model_id=embedder.model_id,
        dimension=embedder.dimension,
    )
    store.rebuild_vector_index(embedder, index)

    recalled = index.search(
        embedder.embed("vehicle washing"),
        top_k=10,
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        domain="household",
        status="active",
    )

    assert [rid for rid, _score in recalled] == [target.rid]


@pytest.mark.skipif(not sqlite_vec_available(), reason="sqlite-vec is not installed")
def test_recall_can_use_sqlite_vec_backend_for_ann_candidates(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    embedder = KeywordEmbedder(
        dimensions=("car", "wash", "hope"),
        synonyms={"vehicle": "car", "cleans": "wash"},
    )
    target = store.add_memory(
        "Collaborator handles car washing at the house.",
        owner="primary",
        platform_scope="whatsapp",
        domain="household",
    )
    store.embed_missing(embedder)
    index = SQLiteVecVectorIndex(
        db_path=tmp_path / "vectors.db",
        model_id=embedder.model_id,
        dimension=embedder.dimension,
    )
    store.rebuild_vector_index(embedder, index)

    recalled = store.recall(
        "Who cleans the vehicle?",
        owner="primary",
        platform="whatsapp",
        allowed_visibility={"private"},
        limit=5,
        embedder=embedder,
        vector_candidate_limit=1000,
        vector_index=index,
        ann_candidate_limit=10,
    )

    assert recalled[0].record.rid == target.rid
    assert "ann_match" in recalled[0].reasons

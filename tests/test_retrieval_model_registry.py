from __future__ import annotations

from anamnesis.embedding_models import get_model_spec


def test_potion_retrieval_32m_is_registered_as_first_party_release_asset():
    spec = get_model_spec("potion-retrieval-32M")

    assert spec.name == "potion-retrieval-32M"
    assert spec.dimension == 512
    assert spec.architecture == "model2vec"
    assert spec.asset_name == "potion-retrieval-32M.tar.gz"
    assert spec.release_tag == "v0.2.0"
    assert "anamnesis-memory/anamnesis-models" in spec.url
    assert "huggingface.co" not in spec.url
    assert len(spec.sha256) == 64


def test_potion_retrieval_32m_model_id_is_distinct_from_base_32m():
    retrieval = get_model_spec("potion-retrieval-32M")
    base = get_model_spec("potion-base-32M")

    assert retrieval.model_id != base.model_id
    assert retrieval.dimension == base.dimension

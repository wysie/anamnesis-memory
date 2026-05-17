from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from anamnesis.embedding_models import (
    EmbeddingModelSpec,
    ModelDownloadNotAllowed,
    ModelVerificationError,
    default_model_cache_dir,
    ensure_model_cached,
    get_model_spec,
    load_embedder_from_env,
)


def _make_model_tar(path: Path, files: dict[str, str]) -> str:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_known_official_model_infers_dimension_and_release_metadata():
    spec = get_model_spec("potion-base-32M")

    assert spec.name == "potion-base-32M"
    assert spec.dimension == 512
    assert spec.release_tag == "v0.1.0"
    assert spec.asset_name == "potion-base-32M.tar.gz"
    assert spec.url == "https://github.com/anamnesis-memory/anamnesis-models/releases/download/v0.1.0/potion-base-32M.tar.gz"
    assert spec.sha256 == "ea2b3e5d0717f6215b15c5cba01481bbb3e1d167f6d684bd51d41a0658dbaba6"


def test_default_embedder_model_is_potion_base_2m():
    from anamnesis.embedding_models import DEFAULT_EMBEDDER_MODEL, recall_mode_from_env

    spec = get_model_spec(DEFAULT_EMBEDDER_MODEL)
    assert DEFAULT_EMBEDDER_MODEL == "potion-base-2M"
    assert spec.dimension == 64
    assert spec.url == "https://github.com/anamnesis-memory/anamnesis-models/releases/download/v0.1.0/potion-base-2M.tar.gz"
    assert spec.sha256 == "56f3fc0104e93f694b6e02b90995c134fa2209d8f28270f2caa39b1de3c4603e"
    assert recall_mode_from_env() == "embedder"


def test_recall_mode_env_supports_explicit_fts(monkeypatch):
    from anamnesis.embedding_models import recall_mode_from_env

    monkeypatch.setenv("ANAMNESIS_RECALL_MODE", "fts")
    assert recall_mode_from_env() == "fts"


def test_official_registry_uses_project_owned_release_urls():
    for spec in [get_model_spec("potion-base-8M"), get_model_spec("potion-base-32M")]:
        assert "anamnesis-memory/anamnesis-models" in spec.url
        assert spec.url.startswith("https://github.com/anamnesis-memory/anamnesis-models/releases/download/")


def test_default_cache_dir_uses_platform_cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("ANAMNESIS_MODEL_CACHE_DIR", str(tmp_path / "models"))

    assert default_model_cache_dir() == tmp_path / "models"


def test_ensure_model_cached_refuses_network_when_download_disabled(tmp_path):
    spec = EmbeddingModelSpec(
        name="demo-model",
        dimension=3,
        release_tag="v0.0.1",
        asset_name="demo-model.tar.gz",
        sha256="0" * 64,
        url="https://example.invalid/demo-model.tar.gz",
    )

    with pytest.raises(ModelDownloadNotAllowed):
        ensure_model_cached(spec, cache_dir=tmp_path, allow_download=False)


def test_ensure_model_cached_downloads_verifies_and_extracts_release_asset(tmp_path):
    archive = tmp_path / "demo-model.tar.gz"
    digest = _make_model_tar(
        archive,
        {
            "config.json": '{"model_type":"model2vec","hidden_dim":3}',
            "tokenizer.json": "{}",
            "model.safetensors": "weights",
        },
    )
    spec = EmbeddingModelSpec(
        name="demo-model",
        dimension=3,
        release_tag="v0.0.1",
        asset_name="demo-model.tar.gz",
        sha256=digest,
        url=archive.as_uri(),
    )

    model_dir = ensure_model_cached(spec, cache_dir=tmp_path / "cache", allow_download=True)

    assert model_dir == tmp_path / "cache" / "demo-model-v0.0.1"
    assert (model_dir / "config.json").read_text() == '{"model_type":"model2vec","hidden_dim":3}'
    assert (model_dir / ".anamnesis-model.json").exists()


def test_ensure_model_cached_uses_existing_verified_cache_without_download(tmp_path):
    spec = EmbeddingModelSpec(
        name="demo-model",
        dimension=3,
        release_tag="v0.0.1",
        asset_name="demo-model.tar.gz",
        sha256="0" * 64,
        url="https://example.invalid/demo-model.tar.gz",
    )
    model_dir = tmp_path / "cache" / "demo-model-v0.0.1"
    model_dir.mkdir(parents=True)
    (model_dir / ".anamnesis-model.json").write_text("{}")

    assert ensure_model_cached(spec, cache_dir=tmp_path / "cache", allow_download=False) == model_dir


def test_ensure_model_cached_rejects_sha_mismatch_without_partial_extract(tmp_path):
    archive = tmp_path / "demo-model.tar.gz"
    _make_model_tar(archive, {"config.json": "{}"})
    spec = EmbeddingModelSpec(
        name="demo-model",
        dimension=3,
        release_tag="v0.0.1",
        asset_name="demo-model.tar.gz",
        sha256="f" * 64,
        url=archive.as_uri(),
    )

    with pytest.raises(ModelVerificationError):
        ensure_model_cached(spec, cache_dir=tmp_path / "cache", allow_download=True)

    assert not (tmp_path / "cache" / "demo-model-v0.0.1").exists()


def test_load_embedder_from_env_requires_explicit_download_permission(monkeypatch, tmp_path):
    monkeypatch.delenv("ANAMNESIS_EMBEDDER", raising=False)
    monkeypatch.setenv("ANAMNESIS_RECALL_MODE", "embedder")
    monkeypatch.setenv("ANAMNESIS_MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ANAMNESIS_ALLOW_MODEL_DOWNLOADS", raising=False)

    with pytest.raises(ModelDownloadNotAllowed):
        load_embedder_from_env()


def test_load_embedder_from_env_returns_none_for_fts_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("ANAMNESIS_EMBEDDER", raising=False)
    monkeypatch.setenv("ANAMNESIS_RECALL_MODE", "fts")
    monkeypatch.setenv("ANAMNESIS_MODEL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("ANAMNESIS_ALLOW_MODEL_DOWNLOADS", raising=False)

    assert load_embedder_from_env() is None

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ModelDownloadNotAllowed(RuntimeError):
    """Raised when a model is missing and downloads are disabled."""


class ModelVerificationError(RuntimeError):
    """Raised when a downloaded model archive fails verification."""


class ModelLoadError(RuntimeError):
    """Raised when an installed model cannot be loaded as an embedder."""


@dataclass(frozen=True)
class EmbeddingModelSpec:
    name: str
    dimension: int
    release_tag: str
    asset_name: str
    sha256: str
    url: str
    architecture: str = "model2vec"
    description: str = ""

    @property
    def model_id(self) -> str:
        return f"{self.name}:{self.release_tag}:{self.sha256[:12]}"

    @property
    def cache_key(self) -> str:
        return f"{self.name}-{self.release_tag}"


_MODELS_BASE = "https://github.com/anamnesis-memory/anamnesis-models/releases/download"
DEFAULT_RECALL_MODE = "embedder"
DEFAULT_EMBEDDER_MODEL = "potion-base-2M"

OFFICIAL_EMBEDDING_MODELS: dict[str, EmbeddingModelSpec] = {
    "potion-base-2M": EmbeddingModelSpec(
        name="potion-base-2M",
        dimension=64,
        release_tag="v0.1.0",
        asset_name="potion-base-2M.tar.gz",
        sha256="56f3fc0104e93f694b6e02b90995c134fa2209d8f28270f2caa39b1de3c4603e",
        url=f"{_MODELS_BASE}/v0.1.0/potion-base-2M.tar.gz",
        description="Static model2vec embedder, ~7 MB cached, lightweight default semantic recall.",
    ),
    "potion-base-8M": EmbeddingModelSpec(
        name="potion-base-8M",
        dimension=256,
        release_tag="v0.1.0",
        asset_name="potion-base-8M.tar.gz",
        sha256="74360e28f7e9a7beecb785132772b4f9ae5be5b1ec45d4470c804c1f3f1cf6a9",
        url=f"{_MODELS_BASE}/v0.1.0/potion-base-8M.tar.gz",
        description="Static model2vec embedder, ~28 MB cached, ~92% MiniLM quality.",
    ),
    "potion-base-32M": EmbeddingModelSpec(
        name="potion-base-32M",
        dimension=512,
        release_tag="v0.1.0",
        asset_name="potion-base-32M.tar.gz",
        sha256="ea2b3e5d0717f6215b15c5cba01481bbb3e1d167f6d684bd51d41a0658dbaba6",
        url=f"{_MODELS_BASE}/v0.1.0/potion-base-32M.tar.gz",
        description="Static model2vec embedder, ~121 MB cached, ~95% MiniLM quality.",
    ),
    "potion-retrieval-32M": EmbeddingModelSpec(
        name="potion-retrieval-32M",
        dimension=512,
        release_tag="v0.2.0",
        asset_name="potion-retrieval-32M.tar.gz",
        sha256="0424b427046d7f95a03b8a7495aec235fa224c149310ec677fd414437f5d1b06",
        url=f"{_MODELS_BASE}/v0.2.0/potion-retrieval-32M.tar.gz",
        description="Static model2vec retrieval-finetuned embedder, ~121 MB cached, optimized for query-memory recall.",
    ),
}


def get_model_spec(name: str) -> EmbeddingModelSpec:
    try:
        return OFFICIAL_EMBEDDING_MODELS[name]
    except KeyError as exc:
        known = ", ".join(sorted(OFFICIAL_EMBEDDING_MODELS))
        raise KeyError(f"unknown Anamnesis embedding model {name!r}; known: {known}") from exc


def default_model_cache_dir() -> Path:
    override = os.environ.get("ANAMNESIS_MODEL_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Caches" / "anamnesis" / "models"
    if system == "Windows":
        root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(root) / "anamnesis" / "Cache" / "models"
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "anamnesis" / "models"


def downloads_allowed() -> bool:
    value = os.environ.get("ANAMNESIS_ALLOW_MODEL_DOWNLOADS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def recall_mode_from_env() -> str:
    value = os.environ.get("ANAMNESIS_RECALL_MODE", DEFAULT_RECALL_MODE).strip().lower()
    if value not in {"fts", "embedder"}:
        raise ValueError("ANAMNESIS_RECALL_MODE must be 'fts' or 'embedder'")
    return value


def ensure_model_cached(
    spec: EmbeddingModelSpec,
    *,
    cache_dir: str | Path | None = None,
    allow_download: bool = False,
) -> Path:
    cache_root = Path(cache_dir).expanduser() if cache_dir is not None else default_model_cache_dir()
    model_dir = cache_root / spec.cache_key
    manifest_path = model_dir / ".anamnesis-model.json"
    if manifest_path.exists():
        return model_dir
    if not allow_download:
        raise ModelDownloadNotAllowed(
            f"{spec.name} is not cached at {model_dir}. Set "
            "ANAMNESIS_ALLOW_MODEL_DOWNLOADS=1 to download official models."
        )

    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="anamnesis-model-") as tmp:
        tmp_dir = Path(tmp)
        archive_path = tmp_dir / spec.asset_name
        _download(spec.url, archive_path)
        actual = _sha256(archive_path)
        if actual != spec.sha256:
            raise ModelVerificationError(
                f"sha256 mismatch for {spec.name}: expected {spec.sha256}, got {actual}"
            )
        extract_dir = tmp_dir / "extract"
        extract_dir.mkdir()
        _safe_extract_tar_gz(archive_path, extract_dir)
        _write_manifest(extract_dir, spec, actual)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.move(str(extract_dir), str(model_dir))
    return model_dir


def load_embedder_from_env() -> Any | None:
    """Load the configured embedder.

    Returns None for unset/none so callers can keep FTS-only recall. Official
    model names use Anamnesis release assets and local cache. Local paths and
    Hugging Face/BYO loaders can be added without changing the cache contract.
    """
    selected = os.environ.get("ANAMNESIS_EMBEDDER", "").strip()
    if recall_mode_from_env() == "fts":
        return None
    if not selected:
        selected = DEFAULT_EMBEDDER_MODEL
    return load_embedder_by_name(selected)


def load_embedder_by_name(selected: str) -> Any | None:
    selected = selected.strip()
    if not selected or selected.lower() == "none":
        return None
    if selected in OFFICIAL_EMBEDDING_MODELS:
        spec = get_model_spec(selected)
        model_dir = ensure_model_cached(spec, allow_download=downloads_allowed())
        return _load_model2vec_embedder(spec, model_dir)
    path = Path(selected).expanduser()
    if path.exists():
        spec = _spec_from_local_model_dir(path)
        return _load_model2vec_embedder(spec, path)
    raise KeyError(f"unknown Anamnesis embedder {selected!r}")


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_tar_gz(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tar:
        dest = destination.resolve()
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(dest) + os.sep) and target != dest:
                raise ModelVerificationError(f"unsafe tar path in model archive: {member.name}")
        tar.extractall(destination)  # noqa: S202 - paths verified above


def _write_manifest(model_dir: Path, spec: EmbeddingModelSpec, actual_sha256: str) -> None:
    manifest = asdict(spec) | {"verified_sha256": actual_sha256}
    (model_dir / ".anamnesis-model.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _spec_from_local_model_dir(path: Path) -> EmbeddingModelSpec:
    config_path = path / "config.json"
    if not config_path.exists():
        raise ModelLoadError(f"local embedder path lacks config.json: {path}")
    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"invalid config.json in {path}: {exc}") from exc
    dimension = int(config.get("hidden_dim") or config.get("dim") or config.get("embedding_dim") or 0)
    if dimension <= 0:
        raise ModelLoadError(f"cannot infer embedding dimension from {config_path}")
    return EmbeddingModelSpec(
        name=path.name,
        dimension=dimension,
        release_tag="local",
        asset_name="local",
        sha256="local",
        url=str(path),
        description="Local model path",
    )


def _load_model2vec_embedder(spec: EmbeddingModelSpec, model_dir: Path) -> Any:
    try:
        from model2vec import StaticModel  # type: ignore
    except ImportError as exc:
        raise ModelLoadError(
            "model2vec is required for Anamnesis official embedding models. "
            "Install the optional model2vec dependency or use a custom Embedder."
        ) from exc
    model = StaticModel.from_pretrained(str(model_dir))
    return Model2VecEmbedder(spec=spec, model=model)


@dataclass(frozen=True)
class Model2VecEmbedder:
    spec: EmbeddingModelSpec
    model: Any

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    @property
    def dimension(self) -> int:
        return self.spec.dimension

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(text)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        values = [float(value) for value in vector]
        if len(values) != self.dimension:
            raise ModelLoadError(
                f"embedder {self.spec.name} returned dim {len(values)}, expected {self.dimension}"
            )
        return values

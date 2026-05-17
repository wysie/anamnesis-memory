"""Anamnesis: local-first governed memory for agents."""

from .benchmarks import RecallBenchmarkCase, RecallBenchmarkCaseResult, RecallBenchmarkReport, run_recall_benchmark
from .core import Anamnesis, Contradiction, MemoryInboxItem, MemoryRecord, RecallResult
from .synthesis import LocalLLMConfig, OpenAICompatibleLLMClient, SynthesisResult, synthesize_from_recall
from .synthesis_benchmark import (
    SynthesisBenchmarkCase,
    SynthesisBenchmarkCaseResult,
    SynthesisBenchmarkReport,
    run_synthesis_benchmark,
)
from .embedding_models import (
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_RECALL_MODE,
    EmbeddingModelSpec,
    ensure_model_cached,
    get_model_spec,
    load_embedder_by_name,
    load_embedder_from_env,
    recall_mode_from_env,
)
from .embeddings import Embedder, KeywordEmbedder
from .vector_index import ExactVectorIndex, SQLiteVecVectorIndex, VectorIndex, sqlite_vec_available

__all__ = [
    "Anamnesis",
    "Contradiction",
    "DEFAULT_EMBEDDER_MODEL",
    "DEFAULT_RECALL_MODE",
    "Embedder",
    "EmbeddingModelSpec",
    "ExactVectorIndex",
    "KeywordEmbedder",
    "LocalLLMConfig",
    "MemoryInboxItem",
    "MemoryRecord",
    "RecallBenchmarkCase",
    "RecallBenchmarkCaseResult",
    "RecallBenchmarkReport",
    "RecallResult",
    "SQLiteVecVectorIndex",
    "SynthesisBenchmarkCase",
    "SynthesisBenchmarkCaseResult",
    "SynthesisBenchmarkReport",
    "SynthesisResult",
    "VectorIndex",
    "ensure_model_cached",
    "get_model_spec",
    "load_embedder_by_name",
    "load_embedder_from_env",
    "OpenAICompatibleLLMClient",
    "recall_mode_from_env",
    "run_recall_benchmark",
    "run_synthesis_benchmark",
    "sqlite_vec_available",
    "synthesize_from_recall",
]

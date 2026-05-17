from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol


class Embedder(Protocol):
    """Minimal local embedder interface for optional semantic recall."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class KeywordEmbedder:
    """Deterministic test/local embedder based on keyword dimensions.

    This is intentionally simple and dependency-free. Production embedders can
    implement the same protocol while tests can assert vector behaviour without
    network/model downloads.
    """

    dimensions: tuple[str, ...]
    synonyms: dict[str, str] = field(default_factory=dict)
    name: str = "keyword"

    @property
    def model_id(self) -> str:
        return f"{self.name}:{','.join(self.dimensions)}"

    @property
    def dimension(self) -> int:
        return len(self.dimensions)

    def embed(self, text: str) -> list[float]:
        tokens = [_normalize_token(token, self.synonyms) for token in _tokens(text)]
        counts = {token: tokens.count(token) for token in set(tokens)}
        vector = [float(counts.get(dim.lower(), 0)) for dim in self.dimensions]
        return normalize(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))


def normalize(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_@.+-]+", text.lower())


def _normalize_token(token: str, synonyms: dict[str, str]) -> str:
    return synonyms.get(token, token)

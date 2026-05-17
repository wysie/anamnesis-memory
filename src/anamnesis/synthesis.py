from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from .core import RecallResult

Transport = Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]]


@dataclass(frozen=True)
class LocalLLMConfig:
    """Anamnesis-owned local/private OpenAI-compatible synthesis config."""

    base_url: str
    model: str
    api_key_env: str | None = None
    max_tokens: int = 512
    temperature: float = 0.0
    timeout: int = 60
    max_context_chars: int = 8000
    max_memory_chars: int = 1200


@dataclass(frozen=True)
class PackedMemorySources:
    memory_block: str
    memory_ids: list[str]
    truncated_memory_ids: list[str]
    char_count: int


@dataclass(frozen=True)
class SynthesisResult:
    query: str
    answer: str
    memory_ids: list[str]
    cited_memory_ids: list[str]
    model: str
    uncited_memory_ids: list[str]
    citation_missing: bool
    insufficient_evidence: bool
    retry_count: int
    packed_char_count: int
    truncated_memory_ids: list[str]


def pack_memory_sources(
    recall_results: list[RecallResult],
    *,
    max_context_chars: int = 8000,
    max_memory_chars: int = 1200,
) -> PackedMemorySources:
    chunks: list[str] = []
    memory_ids: list[str] = []
    truncated: list[str] = []
    budget = max(0, max_context_chars)
    for result in sorted(recall_results, key=lambda item: item.score, reverse=True):
        text = result.record.text.strip()
        was_truncated = len(text) > max_memory_chars
        if was_truncated:
            text = text[: max(0, max_memory_chars - 15)].rstrip() + " ...[truncated]"
        chunk = (
            f"[{result.record.rid}] domain={result.record.domain or '(none)'} "
            f"score={result.score}\n{text}"
        )
        projected_len = len("\n\n".join(chunks + [chunk]))
        if projected_len > budget:
            continue
        chunks.append(chunk)
        memory_ids.append(result.record.rid)
        if was_truncated:
            truncated.append(result.record.rid)
    block = "\n\n".join(chunks)
    return PackedMemorySources(
        memory_block=block,
        memory_ids=memory_ids,
        truncated_memory_ids=truncated,
        char_count=len(block),
    )


def build_memory_synthesis_messages(
    query: str,
    recall_results: list[RecallResult],
    *,
    max_context_chars: int = 8000,
    max_memory_chars: int = 1200,
) -> list[dict[str, str]]:
    packed = pack_memory_sources(
        recall_results,
        max_context_chars=max_context_chars,
        max_memory_chars=max_memory_chars,
    )
    return [
        {
            "role": "system",
            "content": (
                "You synthesize answers from Anamnesis memory recall results. "
                "This is read-only. Do not write, update, delete, propose memory mutations, "
                "or infer from outside the supplied memories. If the memories are insufficient, say so. "
                "Cite memory IDs inline using [memory_id] for every factual claim."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\n"
                "Recalled memories:\n"
                f"{packed.memory_block if packed.memory_block else '(none)'}\n\n"
                "Answer from the recalled memories only, with memory-ID citations."
            ),
        },
    ]


class OpenAICompatibleLLMClient:
    def __init__(self, config: LocalLLMConfig, transport: Transport | None = None) -> None:
        self.config = config
        self._transport = transport or _default_transport

    def complete(self, messages: list[dict[str, str]]) -> str:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key_env:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Local LLM API key env var {self.config.api_key_env} is not set"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        response = self._transport(
            _chat_completions_url(self.config.base_url),
            payload,
            headers,
            self.config.timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Local LLM response did not contain choices[0].message.content: {response!r}"
            ) from exc
        return str(content).strip()


def synthesize_from_recall(
    query: str,
    recall_results: list[RecallResult],
    config: LocalLLMConfig,
    *,
    transport: Transport | None = None,
) -> SynthesisResult:
    packed = pack_memory_sources(
        recall_results,
        max_context_chars=config.max_context_chars,
        max_memory_chars=config.max_memory_chars,
    )
    if not packed.memory_ids:
        return _no_evidence_result(query, config, packed, "I could not find relevant recalled memories to answer this.")
    if _clearly_insufficient_evidence(query, recall_results):
        return _no_evidence_result(
            query,
            config,
            packed,
            "I have insufficient recalled memory evidence to answer this.",
        )

    messages = build_memory_synthesis_messages(
        query,
        recall_results,
        max_context_chars=config.max_context_chars,
        max_memory_chars=config.max_memory_chars,
    )
    client = OpenAICompatibleLLMClient(config, transport=transport)
    answer = client.complete(messages)
    retry_count = 0
    cited = _cited_memory_ids(answer, packed.memory_ids)
    if _answer_needs_citations(answer) and not cited:
        retry_count = 1
        retry_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Previous answer was missing citations. Rewrite the answer with inline memory-ID "
                    "citations like [memory_id]. If evidence is insufficient, say that explicitly."
                ),
            }
        ]
        answer = client.complete(retry_messages)
        cited = _cited_memory_ids(answer, packed.memory_ids)

    insufficient = _is_insufficient_answer(answer)
    citation_missing = _answer_needs_citations(answer) and not cited and not insufficient
    uncited = [rid for rid in packed.memory_ids if rid not in cited]
    return SynthesisResult(
        query=query,
        answer=answer,
        memory_ids=packed.memory_ids,
        cited_memory_ids=cited,
        model=config.model,
        uncited_memory_ids=uncited,
        citation_missing=citation_missing,
        insufficient_evidence=insufficient,
        retry_count=retry_count,
        packed_char_count=packed.char_count,
        truncated_memory_ids=packed.truncated_memory_ids,
    )


def _no_evidence_result(
    query: str, config: LocalLLMConfig, packed: PackedMemorySources, answer: str
) -> SynthesisResult:
    return SynthesisResult(
        query=query,
        answer=answer,
        memory_ids=packed.memory_ids,
        cited_memory_ids=[],
        model=config.model,
        uncited_memory_ids=packed.memory_ids,
        citation_missing=False,
        insufficient_evidence=True,
        retry_count=0,
        packed_char_count=packed.char_count,
        truncated_memory_ids=packed.truncated_memory_ids,
    )


def _answer_needs_citations(answer: str) -> bool:
    return bool(answer.strip()) and not _is_insufficient_answer(answer)


def _is_insufficient_answer(answer: str) -> bool:
    lowered = answer.lower()
    return "insufficient" in lowered or "could not find" in lowered or "not enough" in lowered


def _cited_memory_ids(answer: str, memory_ids: list[str]) -> list[str]:
    return [rid for rid in memory_ids if re.search(rf"\[{re.escape(rid)}\]", answer)]


def _clearly_insufficient_evidence(query: str, recall_results: list[RecallResult]) -> bool:
    query_terms = _meaningful_terms(query)
    if not query_terms:
        return False
    text_terms = _meaningful_terms("\n".join(result.record.text for result in recall_results))
    overlap = query_terms & text_terms
    if overlap:
        return False
    sensitive_unknown_markers = {
        "account",
        "bank",
        "card",
        "credit",
        "password",
        "token",
        "secret",
        "ssn",
        "passport",
    }
    return bool(query_terms & sensitive_unknown_markers)


def _meaningful_terms(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "are",
        "does",
        "for",
        "from",
        "have",
        "how",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "primary",
        "user",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9_]{3,}", text.lower())
        if token not in stop
    }


def _chat_completions_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return f"{root}/chat/completions"


def _default_transport(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured local/private endpoint
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local LLM HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Local LLM request failed: {exc}") from exc

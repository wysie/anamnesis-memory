from __future__ import annotations

import json

import pytest

from anamnesis import Anamnesis
from anamnesis.core import MemoryRecord, RecallResult
from anamnesis.synthesis import (
    LocalLLMConfig,
    OpenAICompatibleLLMClient,
    build_memory_synthesis_messages,
    pack_memory_sources,
    synthesize_from_recall,
)


def _record(rid: str, text: str) -> MemoryRecord:
    return MemoryRecord(
        rid=rid,
        text=text,
        kind="semantic",
        owner="primary",
        visibility="private",
        platform_scope="cli",
        action_scope="read_only",
        domain="preference",
        source="test",
        importance=0.7,
        confidence=1.0,
        status="active",
        created_at=1.0,
        updated_at=1.0,
        last_access=None,
        ttl_days=None,
        metadata={},
    )


def test_synthesis_config_is_anamnesis_owned_and_uses_api_key_env(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")

    config = store.set_synthesis_config(
        base_url="http://127.0.0.1:8060/v1",
        model="local-private-model",
        api_key_env="ANAMNESIS_LLM_API_KEY",
        temperature=0,
        max_tokens=512,
    )

    assert config == {
        "base_url": "http://127.0.0.1:8060/v1",
        "model": "local-private-model",
        "api_key_env": "ANAMNESIS_LLM_API_KEY",
        "temperature": "0",
        "max_tokens": "512",
    }
    assert Anamnesis(tmp_path / "anamnesis.db").synthesis_config()["model"] == "local-private-model"


def test_synthesis_prompt_is_read_only_and_cites_memory_ids():
    hits = [
        RecallResult(_record("mem_a", "Primary user prefers local-only memory synthesis."), 3.0, ["keyword_match"]),
        RecallResult(_record("mem_b", "Answers should cite memory IDs."), 2.0, ["semantic_match"]),
    ]

    messages = build_memory_synthesis_messages("How should synthesis work?", hits)
    serialized = json.dumps(messages)

    assert "read-only" in messages[0]["content"]
    assert "Do not write" in messages[0]["content"]
    assert "[mem_a]" in serialized
    assert "[mem_b]" in serialized
    assert "cite memory IDs" in serialized


def test_openai_compatible_client_uses_own_endpoint_model_and_api_key(monkeypatch):
    calls = []

    def fake_transport(url, payload, headers, timeout):
        calls.append((url, payload, headers, timeout))
        return {"choices": [{"message": {"content": "Use local synthesis [mem_a]."}}]}

    monkeypatch.setenv("ANAMNESIS_LLM_API_KEY", "secret-key")
    client = OpenAICompatibleLLMClient(
        LocalLLMConfig(
            base_url="http://127.0.0.1:8060/v1",
            model="qwen-local",
            api_key_env="ANAMNESIS_LLM_API_KEY",
            max_tokens=128,
            temperature=0.0,
        ),
        transport=fake_transport,
    )

    answer = client.complete([{"role": "user", "content": "hi"}])

    assert answer == "Use local synthesis [mem_a]."
    url, payload, headers, timeout = calls[0]
    assert url == "http://127.0.0.1:8060/v1/chat/completions"
    assert payload["model"] == "qwen-local"
    assert payload["max_tokens"] == 128
    assert headers["Authorization"] == "Bearer secret-key"
    assert timeout == 60


def test_synthesize_from_recall_is_read_only_and_returns_citations():
    calls = []
    hits = [RecallResult(_record("mem_a", "Primary user prefers local-only memory synthesis."), 3.0, [])]

    def fake_transport(url, payload, headers, timeout):
        calls.append(payload)
        return {"choices": [{"message": {"content": "Use a private local endpoint [mem_a]."}}]}

    result = synthesize_from_recall(
        "How should synthesis run?",
        hits,
        LocalLLMConfig(base_url="http://localhost:8060/v1", model="local-model"),
        transport=fake_transport,
    )

    assert result.answer == "Use a private local endpoint [mem_a]."
    assert result.cited_memory_ids == ["mem_a"]
    assert result.memory_ids == ["mem_a"]
    assert "Do not write" in calls[0]["messages"][0]["content"]


def test_openai_compatible_client_requires_api_key_env_when_configured(monkeypatch):
    monkeypatch.delenv("MISSING_ANAMNESIS_KEY", raising=False)
    client = OpenAICompatibleLLMClient(
        LocalLLMConfig(
            base_url="http://127.0.0.1:8060/v1",
            model="qwen-local",
            api_key_env="MISSING_ANAMNESIS_KEY",
        ),
        transport=lambda *_args: {},
    )

    with pytest.raises(RuntimeError, match="MISSING_ANAMNESIS_KEY"):
        client.complete([{"role": "user", "content": "hi"}])


def test_pack_memory_sources_enforces_budget_and_preserves_highest_scores():
    hits = [
        RecallResult(_record("low", "low score " + "x" * 200), 1.0, []),
        RecallResult(_record("high", "high score " + "y" * 200), 9.0, []),
    ]

    packed = pack_memory_sources(hits, max_context_chars=120, max_memory_chars=40)

    assert packed.memory_ids == ["high"]
    assert packed.truncated_memory_ids == ["high"]
    assert "[high]" in packed.memory_block
    assert "[low]" not in packed.memory_block
    assert len(packed.memory_block) <= 120


def test_synthesize_retries_once_when_answer_has_no_citations():
    calls = []
    hits = [RecallResult(_record("mem_a", "Primary user prefers local-only memory synthesis."), 3.0, [])]

    def fake_transport(url, payload, headers, timeout):
        calls.append(payload)
        if len(calls) == 1:
            return {"choices": [{"message": {"content": "Use local-only synthesis."}}]}
        return {"choices": [{"message": {"content": "Use local-only synthesis [mem_a]."}}]}

    result = synthesize_from_recall(
        "How should synthesis run?",
        hits,
        LocalLLMConfig(base_url="http://localhost:8060/v1", model="local-model"),
        transport=fake_transport,
    )

    assert result.answer == "Use local-only synthesis [mem_a]."
    assert result.citation_missing is False
    assert result.retry_count == 1
    assert result.cited_memory_ids == ["mem_a"]
    assert "Previous answer was missing citations" in calls[1]["messages"][-1]["content"]


def test_synthesize_marks_missing_citations_after_retry_exhausted():
    hits = [RecallResult(_record("mem_a", "Primary user prefers local-only memory synthesis."), 3.0, [])]

    result = synthesize_from_recall(
        "How should synthesis run?",
        hits,
        LocalLLMConfig(base_url="http://localhost:8060/v1", model="local-model"),
        transport=lambda *_args: {"choices": [{"message": {"content": "Use local-only synthesis."}}]},
    )

    assert result.citation_missing is True
    assert result.retry_count == 1
    assert result.cited_memory_ids == []
    assert result.uncited_memory_ids == ["mem_a"]


def test_synthesize_refuses_when_recalled_memories_are_insufficient_evidence():
    hits = [RecallResult(_record("mem_a", "Primary user prefers local-only memory synthesis."), 3.0, [])]

    result = synthesize_from_recall(
        "What is Primary user's bank account number?",
        hits,
        LocalLLMConfig(base_url="http://localhost:8060/v1", model="local-model"),
        transport=lambda *_args: pytest.fail("LLM should not be called when evidence is clearly insufficient"),
    )

    assert result.insufficient_evidence is True
    assert "insufficient recalled memory evidence" in result.answer.lower()
    assert result.citation_missing is False

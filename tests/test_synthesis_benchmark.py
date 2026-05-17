from __future__ import annotations

from anamnesis.core import MemoryRecord, RecallResult
from anamnesis.synthesis import LocalLLMConfig
from anamnesis.synthesis_benchmark import SynthesisBenchmarkCase, run_synthesis_benchmark


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


def test_synthesis_benchmark_scores_answer_citations_and_refusal():
    calls = []

    def transport(url, payload, headers, timeout):
        calls.append(payload)
        query = payload["messages"][1]["content"]
        if "What does Helper handle" in query:
            return {"choices": [{"message": {"content": "Helper handles pool maintenance [mem_pool]."}}]}
        raise AssertionError("insufficient-evidence case should not call transport")

    cases = [
        SynthesisBenchmarkCase(
            name="answered_with_citation",
            query="What does Helper handle?",
            recall_results=[RecallResult(_record("mem_pool", "Helper handles pool maintenance."), 3.0, [])],
            required_terms=("pool maintenance",),
            expected_citations=("mem_pool",),
        ),
        SynthesisBenchmarkCase(
            name="insufficient_refusal",
            query="What is Primary user's bank account number?",
            recall_results=[RecallResult(_record("mem_pref", "Primary user prefers local-only memory synthesis."), 3.0, [])],
            expect_insufficient=True,
        ),
    ]

    report = run_synthesis_benchmark(
        cases,
        LocalLLMConfig(base_url="http://localhost:8060/v1", model="local-model"),
        transport=transport,
    )

    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert len(calls) == 1
    assert report.results[0].passed is True
    assert report.results[1].insufficient_evidence is True

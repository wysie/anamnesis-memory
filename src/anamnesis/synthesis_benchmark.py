from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import RecallResult
from .synthesis import LocalLLMConfig, SynthesisResult, Transport, synthesize_from_recall


@dataclass(frozen=True)
class SynthesisBenchmarkCase:
    name: str
    query: str
    recall_results: list[RecallResult]
    required_terms: tuple[str, ...] = ()
    expected_citations: tuple[str, ...] = ()
    expect_insufficient: bool = False


@dataclass(frozen=True)
class SynthesisBenchmarkCaseResult:
    name: str
    passed: bool
    failures: list[str]
    answer: str
    cited_memory_ids: list[str]
    citation_missing: bool
    insufficient_evidence: bool
    retry_count: int


@dataclass(frozen=True)
class SynthesisBenchmarkReport:
    total: int
    passed: int
    failed: int
    results: list[SynthesisBenchmarkCaseResult]


def run_synthesis_benchmark(
    cases: list[SynthesisBenchmarkCase],
    config: LocalLLMConfig,
    *,
    transport: Transport | None = None,
) -> SynthesisBenchmarkReport:
    results = [_run_case(case, config, transport=transport) for case in cases]
    passed = sum(1 for result in results if result.passed)
    return SynthesisBenchmarkReport(
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        results=results,
    )


def _run_case(
    case: SynthesisBenchmarkCase,
    config: LocalLLMConfig,
    *,
    transport: Transport | None,
) -> SynthesisBenchmarkCaseResult:
    result = synthesize_from_recall(case.query, case.recall_results, config, transport=transport)
    failures = _score_case(case, result)
    return SynthesisBenchmarkCaseResult(
        name=case.name,
        passed=not failures,
        failures=failures,
        answer=result.answer,
        cited_memory_ids=result.cited_memory_ids,
        citation_missing=result.citation_missing,
        insufficient_evidence=result.insufficient_evidence,
        retry_count=result.retry_count,
    )


def _score_case(case: SynthesisBenchmarkCase, result: SynthesisResult) -> list[str]:
    failures: list[str] = []
    answer_lower = result.answer.lower()
    for term in case.required_terms:
        if term.lower() not in answer_lower:
            failures.append(f"missing_required_term:{term}")
    for citation in case.expected_citations:
        if citation not in result.cited_memory_ids:
            failures.append(f"missing_expected_citation:{citation}")
    if case.expect_insufficient and not result.insufficient_evidence:
        failures.append("expected_insufficient_evidence")
    if not case.expect_insufficient and result.insufficient_evidence:
        failures.append("unexpected_insufficient_evidence")
    if result.citation_missing:
        failures.append("citation_missing")
    return failures


def report_to_dict(report: SynthesisBenchmarkReport) -> dict[str, Any]:
    return {
        "total": report.total,
        "passed": report.passed,
        "failed": report.failed,
        "results": [
            {
                "name": result.name,
                "passed": result.passed,
                "failures": result.failures,
                "answer": result.answer,
                "cited_memory_ids": result.cited_memory_ids,
                "citation_missing": result.citation_missing,
                "insufficient_evidence": result.insufficient_evidence,
                "retry_count": result.retry_count,
            }
            for result in report.results
        ],
    }

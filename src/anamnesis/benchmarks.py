from __future__ import annotations

from dataclasses import dataclass

from .core import Anamnesis


@dataclass(frozen=True)
class RecallBenchmarkCase:
    name: str
    query: str
    owner: str
    platform: str
    allowed_visibility: set[str]
    expected_rids: set[str]
    forbidden_rids: set[str] | None = None
    domain: str | None = None
    limit: int = 10


@dataclass(frozen=True)
class RecallBenchmarkCaseResult:
    name: str
    passed: bool
    recalled_rids: list[str]
    missing_expected: list[str]
    forbidden_recalled: list[str]
    failures: list[str]


@dataclass(frozen=True)
class RecallBenchmarkReport:
    total: int
    passed: int
    failed: int
    case_results: list[RecallBenchmarkCaseResult]


def run_recall_benchmark(store: Anamnesis, cases: list[RecallBenchmarkCase]) -> RecallBenchmarkReport:
    results: list[RecallBenchmarkCaseResult] = []
    for case in cases:
        recalled = store.recall(
            case.query,
            owner=case.owner,
            platform=case.platform,
            allowed_visibility=case.allowed_visibility,
            domain=case.domain,
            limit=case.limit,
        )
        recalled_rids = [result.record.rid for result in recalled]
        recalled_set = set(recalled_rids)
        missing_expected = sorted(case.expected_rids - recalled_set)
        forbidden = case.forbidden_rids or set()
        forbidden_recalled = sorted(forbidden & recalled_set)
        failures = []
        if missing_expected:
            failures.append("missing_expected")
        if forbidden_recalled:
            failures.append("forbidden_recalled")
        results.append(
            RecallBenchmarkCaseResult(
                name=case.name,
                passed=not failures,
                recalled_rids=recalled_rids,
                missing_expected=missing_expected,
                forbidden_recalled=forbidden_recalled,
                failures=failures,
            )
        )
    passed = sum(1 for result in results if result.passed)
    return RecallBenchmarkReport(total=len(results), passed=passed, failed=len(results) - passed, case_results=results)

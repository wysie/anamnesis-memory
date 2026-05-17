from anamnesis import Anamnesis, RecallBenchmarkCase, run_recall_benchmark


def test_recall_benchmark_reports_passed_cases_and_leak_failures(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    wanted = store.add_memory(
        "Primary user prefers local-only private memory.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
        source="test",
    )
    blocked = store.add_memory(
        "Helper cannot control smart home devices.",
        owner="hope",
        visibility="private",
        platform_scope="whatsapp",
        domain="permissions",
        source="test",
    )

    report = run_recall_benchmark(
        store,
        [
            RecallBenchmarkCase(
                name="primary privacy recall",
                query="local private memory",
                owner="primary",
                platform="whatsapp",
                allowed_visibility={"private"},
                expected_rids={wanted.rid},
                forbidden_rids={blocked.rid},
            )
        ],
    )

    assert report.total == 1
    assert report.passed == 1
    assert report.failed == 0
    assert report.case_results[0].passed is True
    assert report.case_results[0].recalled_rids == [wanted.rid]


def test_recall_benchmark_fails_when_expected_memory_is_missing(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    wanted = store.add_memory(
        "Primary user prefers local-only private memory.",
        owner="primary",
        visibility="private",
        platform_scope="whatsapp",
        domain="privacy",
        source="test",
    )

    report = run_recall_benchmark(
        store,
        [
            RecallBenchmarkCase(
                name="wrong owner",
                query="local private memory",
                owner="hope",
                platform="whatsapp",
                allowed_visibility={"private"},
                expected_rids={wanted.rid},
            )
        ],
    )

    assert report.passed == 0
    assert report.failed == 1
    assert "missing_expected" in report.case_results[0].failures

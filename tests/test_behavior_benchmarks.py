from anamnesis import Anamnesis
from anamnesis.behavior_benchmarks import build_core_behavior_suite, seed_core_behavior_fixture


def test_core_behavior_suite_covers_privacy_scope_tombstones_and_rejections(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    fixture = seed_core_behavior_fixture(store)

    report = build_core_behavior_suite(store, fixture).run()

    assert report.total >= 6
    assert report.failed == 0
    case_names = {case.name for case in report.case_results}
    assert "owner_private_memory_visible_to_owner" in case_names
    assert "delegate_cannot_recall_owner_private_memory" in case_names
    assert "collaborator_cannot_recall_owner_private_memory" in case_names
    assert "tombstoned_memory_stays_hidden" in case_names
    assert "rejected_task_state_stays_out_of_recall" in case_names
    assert "resolved_contradiction_keeps_winner_only" in case_names


def test_core_behavior_fixture_flags_duplicate_candidates_without_accepting_them(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    fixture = seed_core_behavior_fixture(store)

    duplicate_item = store.get_inbox_item(fixture["duplicate_candidate_cid"])

    assert duplicate_item.decision == "pending"
    assert duplicate_item.duplicate_rids == [fixture["owner_privacy_rid"]]
    assert "possible_duplicate" in duplicate_item.hints


def test_core_behavior_suite_fails_if_private_memory_is_visible_to_wrong_owner(tmp_path):
    store = Anamnesis(tmp_path / "anamnesis.db")
    fixture = seed_core_behavior_fixture(store)
    leaked = store.add_memory(
        "Owner prefers local-only private memory.",
        owner="delegate",
        visibility="private",
        platform_scope="chat",
        domain="privacy",
        source="test-leak",
    )
    fixture["forbidden_leak_rids"].append(leaked.rid)

    report = build_core_behavior_suite(store, fixture).run()

    assert report.failed >= 1
    leaked_cases = [case for case in report.case_results if "forbidden_recalled" in case.failures]
    assert leaked_cases

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .benchmarks import RecallBenchmarkCase, RecallBenchmarkReport, run_recall_benchmark
from .core import Anamnesis

Fixture = dict[str, Any]


@dataclass(frozen=True)
class CoreBehaviorSuite:
    store: Anamnesis
    cases: list[RecallBenchmarkCase]

    def run(self) -> RecallBenchmarkReport:
        return run_recall_benchmark(self.store, self.cases)


def seed_core_behavior_fixture(store: Anamnesis) -> Fixture:
    owner_private = store.add_memory(
        "Owner prefers local-only private memory.",
        owner="owner",
        visibility="private",
        platform_scope="chat,cli",
        domain="privacy",
        source="fixture",
        importance=0.95,
    )
    delegate_permission = store.add_memory(
        "Delegate can ask questions but cannot control devices.",
        owner="delegate",
        visibility="private",
        platform_scope="chat",
        domain="permissions",
        source="fixture",
        importance=0.95,
    )
    collaborator_access = store.add_memory(
        "Collaborator has broad access but cannot access owner chat history unless owner approves.",
        owner="collaborator",
        visibility="private",
        platform_scope="chat",
        domain="privacy",
        source="fixture",
        importance=0.95,
    )
    stale_task = store.add_memory(
        "Temporary task PID 12345 should not be durable.",
        owner="owner",
        visibility="private",
        platform_scope="cli",
        domain="task-state",
        source="fixture",
    )
    store.tombstone(stale_task.rid, reason="temporary task state")

    rejected = store.propose_memory(
        "Temporary debug port is 54321.",
        source_snippet="debug port 54321",
        owner="owner",
        visibility="private",
        platform_scope="cli",
        domain="task-state",
        source="fixture",
    )
    store.reject_inbox_item(rejected.cid, reason="temporary task state")

    duplicate = store.propose_memory(
        "Owner prefers local-first private memory.",
        source_snippet="local private memory preference",
        owner="owner",
        visibility="private",
        platform_scope="chat",
        domain="privacy",
        source="fixture",
    )

    old_permission = store.add_memory(
        "Owner can share chat summaries.",
        owner="owner",
        visibility="private",
        platform_scope="chat",
        domain="privacy",
        source="fixture",
    )
    new_permission = store.add_memory(
        "Owner cannot share chat summaries.",
        owner="owner",
        visibility="private",
        platform_scope="chat",
        domain="privacy",
        source="fixture",
        importance=0.9,
    )
    conflict = store.detect_contradictions(owner="owner", domain="privacy")[0]
    store.resolve_contradiction(conflict.conflict_id, winner_rid=new_permission.rid, reason="newer correction")

    return {
        "owner_privacy_rid": owner_private.rid,
        "delegate_permission_rid": delegate_permission.rid,
        "collaborator_access_rid": collaborator_access.rid,
        "tombstoned_task_rid": stale_task.rid,
        "rejected_task_cid": rejected.cid,
        "duplicate_candidate_cid": duplicate.cid,
        "resolved_loser_rid": old_permission.rid,
        "resolved_winner_rid": new_permission.rid,
        "forbidden_leak_rids": [],
    }


def build_core_behavior_suite(store: Anamnesis, fixture: Fixture) -> CoreBehaviorSuite:
    leak_rids = set(fixture.get("forbidden_leak_rids", []))
    cases = [
        RecallBenchmarkCase(
            name="owner_private_memory_visible_to_owner",
            query="local private memory",
            owner="owner",
            platform="chat",
            allowed_visibility={"private"},
            expected_rids={fixture["owner_privacy_rid"]},
            forbidden_rids={fixture["delegate_permission_rid"], fixture["collaborator_access_rid"]} | leak_rids,
            domain="privacy",
        ),
        RecallBenchmarkCase(
            name="delegate_cannot_recall_owner_private_memory",
            query="local private memory",
            owner="delegate",
            platform="chat",
            allowed_visibility={"private"},
            expected_rids=set(),
            forbidden_rids={fixture["owner_privacy_rid"]} | leak_rids,
            domain="privacy",
        ),
        RecallBenchmarkCase(
            name="collaborator_cannot_recall_owner_private_memory",
            query="local private memory",
            owner="collaborator",
            platform="chat",
            allowed_visibility={"private"},
            expected_rids=set(),
            forbidden_rids={fixture["owner_privacy_rid"]} | leak_rids,
            domain="privacy",
        ),
        RecallBenchmarkCase(
            name="tombstoned_memory_stays_hidden",
            query="temporary task PID",
            owner="owner",
            platform="cli",
            allowed_visibility={"private"},
            expected_rids=set(),
            forbidden_rids={fixture["tombstoned_task_rid"]},
            domain="task-state",
        ),
        RecallBenchmarkCase(
            name="rejected_task_state_stays_out_of_recall",
            query="temporary debug port",
            owner="owner",
            platform="cli",
            allowed_visibility={"private"},
            expected_rids=set(),
            forbidden_rids=set(),
            domain="task-state",
        ),
        RecallBenchmarkCase(
            name="resolved_contradiction_keeps_winner_only",
            query="chat summaries",
            owner="owner",
            platform="chat",
            allowed_visibility={"private"},
            expected_rids={fixture["resolved_winner_rid"]},
            forbidden_rids={fixture["resolved_loser_rid"]},
            domain="privacy",
        ),
    ]
    return CoreBehaviorSuite(store=store, cases=cases)

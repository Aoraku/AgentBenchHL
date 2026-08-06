from __future__ import annotations

from pathlib import Path

import pytest

from agentbench_hl.adapters.antwar2.policy_probe import DecisionSample
from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.application.metrics_service import MetricsService
from agentbench_hl.reporting.curves import build_curves


def build_service(root: Path, *, probe_schema: str = "legacy") -> MetricsService:
    return MetricsService(
        event_store=JsonlEventStore(root / "events.jsonl"),
        human_ratings={"rank20": 1200.0},
        expected_case_ids=("rank20:P0:1", "rank20:P1:1"),
        epsilon=0.01,
        probe_schema=probe_schema,
    )


def test_metrics_service_requires_complete_fixed_case_panel(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    service.record_decision_sample(
        "v001",
        DecisionSample(
            "state-1",
            ("HOLD", "BUILD:8,11"),
            "HOLD",
            "BUILD:8,11",
            "opening",
            "opening",
        ),
    )
    service.record_match(
        candidate_id="v001",
        case_id="rank20:P0:1",
        opponent_id="rank20",
        role="P0",
        status="complete",
        points=1.0,
        score_margin=8.0,
    )
    service.record_match(
        candidate_id="v001",
        case_id="rank20:P1:1",
        opponent_id="rank20",
        role="P1",
        status="incomplete",
        points=None,
        score_margin=None,
    )

    row = service.finalize_iteration("v001", champion_id="v000")

    assert row.research_iteration == 1
    assert row.behavioral_ig is not None and row.behavioral_ig > 0
    assert row.elo is None
    assert row.win_rate is None
    assert row.score_margin is None


def test_complete_panel_produces_version_local_performance_metrics(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    for role, points, margin in (("P0", 1.0, 8.0), ("P1", 0.0, -4.0)):
        service.record_match(
            candidate_id="v002",
            case_id=f"rank20:{role}:1",
            opponent_id="rank20",
            role=role,
            status="complete",
            points=points,
            score_margin=margin,
        )

    row = service.finalize_iteration("v002", champion_id="v002")

    assert row.win_rate == pytest.approx(0.5)
    assert row.score_margin == pytest.approx(2.0)
    assert row.elo is not None
    assert row.elo.p0 > row.elo.p1


def test_smoke_and_formal_metrics_use_distinct_ledger_keys(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    shared = {
        "candidate_id": "v000",
        "case_id": "rank20:P0:1",
        "opponent_id": "rank20",
        "role": "P0",
    }
    service.record_match(
        **shared,
        scope="smoke",
        status="complete",
        points=1.0,
        score_margin=12.0,
    )

    service.record_match(
        **shared,
        scope="evaluation",
        status="incomplete",
        points=None,
        score_margin=None,
    )

    events = JsonlEventStore(tmp_path / "events.jsonl").read_all()
    assert len(events) == 2


def test_metrics_service_uses_separate_own_rollout_occupancy_traces(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    service.record_occupancy_trace(
        candidate_id="v001",
        case_id="rank20:P0:1",
        parent_state_ids=("state-a", "state-a"),
        candidate_state_ids=("state-a", "state-b"),
    )
    for role, points, margin in (("P0", 1.0, 8.0), ("P1", 0.0, -4.0)):
        service.record_match(
            candidate_id="v001",
            case_id=f"rank20:{role}:1",
            opponent_id="rank20",
            role=role,
            status="complete",
            points=points,
            score_margin=margin,
        )

    row = service.finalize_iteration("v001", champion_id="v001")

    assert row.occupancy_shift == pytest.approx(0.5)


def test_iteration_ig_ignores_samples_outside_the_current_case_panel(
    tmp_path: Path,
) -> None:
    service = build_service(tmp_path)
    service.record_decision_sample(
        "v001",
        DecisionSample(
            "v001-rank01-p0-s1:r0000:p0:a000",
            ("HOLD", "BUILD"),
            "HOLD",
            "BUILD",
            "rank01-parent",
            "rank01-candidate",
        ),
    )
    service.record_decision_sample(
        "v001",
        DecisionSample(
            "v001-rank20-p0-s1:r0000:p0:a000",
            ("HOLD", "BUILD"),
            "HOLD",
            "HOLD",
            "rank20",
            "rank20",
        ),
    )
    for role in ("P0", "P1"):
        service.record_match(
            candidate_id="v001",
            case_id=f"rank20:{role}:1",
            opponent_id="rank20",
            role=role,
            status="complete",
            points=1.0,
            score_margin=5.0,
        )

    row = service.finalize_iteration("v001", champion_id="v001")

    assert row.behavioral_ig == 0.0


def test_corrected_probe_schema_can_supersede_legacy_samples_append_only(
    tmp_path: Path,
) -> None:
    legacy = build_service(tmp_path)
    corrected = build_service(tmp_path, probe_schema="antwar2-round-v2")
    shared_state = "v001-rank20-p0-s1:r0001:p0:a000"
    legacy.record_decision_sample(
        "v001",
        DecisionSample(
            shared_state,
            ("HOLD", "BUILD"),
            "HOLD",
            "BUILD",
            "legacy-parent",
            "legacy-candidate",
        ),
    )
    corrected.record_decision_sample(
        "v001",
        DecisionSample(
            shared_state,
            ("HOLD", "BUILD"),
            "HOLD",
            "HOLD",
            "correct-parent",
            "correct-candidate",
        ),
    )
    legacy.record_occupancy_trace(
        candidate_id="v001",
        case_id="rank20:P0:1",
        parent_state_ids=("legacy-a",),
        candidate_state_ids=("legacy-b",),
    )
    corrected.record_occupancy_trace(
        candidate_id="v001",
        case_id="rank20:P0:1",
        parent_state_ids=("correct-a",),
        candidate_state_ids=("correct-a",),
    )

    row = corrected.finalize_iteration("v001", champion_id="v001")

    assert row.behavioral_ig == 0.0
    assert row.occupancy_shift == 0.0
    events = JsonlEventStore(tmp_path / "events.jsonl").read_all()
    assert len([event for event in events if event.event_type == "DecisionSampleRecorded"]) == 2
    assert len([event for event in events if event.event_type == "OccupancyTraceRecorded"]) == 2


def test_metrics_service_records_decision_samples_as_one_batch(tmp_path: Path) -> None:
    service = build_service(tmp_path, probe_schema="antwar2-round-v2")
    samples = tuple(
        DecisionSample(
            f"v001-rank20-p0-s1:r{index:04d}:p0:a000",
            ("HOLD", "BUILD"),
            "HOLD",
            "BUILD",
            f"parent-{index}",
            f"candidate-{index}",
        )
        for index in range(3)
    )

    assert service.record_decision_samples("v001", samples) == 3
    assert service.record_decision_samples("v001", samples) == 0
    events = JsonlEventStore(tmp_path / "events.jsonl").read_all()
    assert len(events) == 3


def test_curve_rows_use_integer_iteration_and_four_primary_panels(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    for role, points, margin in (("P0", 1.0, 8.0), ("P1", 0.0, -4.0)):
        service.record_match(
            candidate_id="v001",
            case_id=f"rank20:{role}:1",
            opponent_id="rank20",
            role=role,
            status="complete",
            points=points,
            score_margin=margin,
        )
    row = service.finalize_iteration("v001", champion_id="v001")

    artifacts = build_curves((row,), tmp_path / "curves")

    assert artifacts.primary_png.is_file()
    assert artifacts.score_margin_png.is_file()
    assert artifacts.csv.is_file()
    header = artifacts.csv.read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("research_iteration,")
    assert artifacts.panel_names == (
        "Behavioral IG",
        "Occupancy shift",
        "Fixed-pool Elo",
        "Win rate",
    )

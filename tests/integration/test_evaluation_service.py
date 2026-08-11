from __future__ import annotations

import threading
from pathlib import Path

from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.application.evaluation_service import (
    EvaluationMatrix,
    EvaluationObservation,
    EvaluationService,
    HumanCalibrationMatch,
    evaluate_candidate,
    fit_human_calibration,
)
from agentbench_hl.ports.arena import MatchResult


def observation(opponent: str, role: str, result: str) -> EvaluationObservation:
    return EvaluationObservation(
        case_id=f"{opponent}:{role}:1",
        opponent_id=opponent,
        role=role,
        seed=1,
        status="complete",
        result=result,
    )


def test_locked_regression_loss_preserves_old_champion() -> None:
    result = evaluate_candidate(
        observations=(
            observation("rank18", "P0", "win"),
            observation("rank18", "P1", "win"),
            observation("rank20", "P0", "win"),
            observation("rank20", "P1", "loss"),
        ),
        target_id="rank18",
        locked_regression_ids=("rank20",),
        roles=("P0", "P1"),
        seeds=(1,),
    )

    assert result.target_solved is True
    assert result.regressions_passed is False
    assert result.promotable is False
    assert result.frontier_eligible is True


def test_incomplete_case_is_retryable_and_never_promotable() -> None:
    result = evaluate_candidate(
        observations=(
            observation("rank20", "P0", "win"),
            EvaluationObservation(
                "rank20:P1:1",
                "rank20",
                "P1",
                1,
                "incomplete",
                None,
            ),
        ),
        target_id="rank20",
        locked_regression_ids=(),
        roles=("P0", "P1"),
        seeds=(1,),
    )

    assert result.promotable is False
    assert result.frontier_eligible is False
    assert result.retry_case_ids == ("rank20:P1:1",)


def test_human_calibration_is_hash_cached_and_order_independent() -> None:
    first_order = (
        HumanCalibrationMatch("rank20", "rank19", 1.0),
        HumanCalibrationMatch("rank19", "rank18", 0.0),
    )
    second_order = tuple(reversed(first_order))

    first = fit_human_calibration(first_order)
    second = fit_human_calibration(second_order)

    assert first.ratings == second.ratings
    assert first.matrix_hash == second.matrix_hash


class EvidenceArena:
    def __init__(self, replay: Path) -> None:
        self.replay = replay
        self.calls = 0

    def run_case(self, case, candidate_root):
        self.calls += 1
        won = case.role == "P0"
        return MatchResult(
            case=case,
            status="complete",
            result="win" if won else "loss",
            points=1.0 if won else 0.0,
            score_margin=7.0 if won else -3.0,
            rounds=42,
            payload={"terminal_base_hp": (7.0, 0.0) if won else (0.0, 3.0)},
            replay_path=self.replay,
            trace_path=self.replay.with_suffix(".trace.jsonl"),
            events_path=self.replay.with_suffix(".events.jsonl"),
            process_returncodes=(0, 0, 0),
        )


def test_complete_evaluation_cache_preserves_replay_and_metric_evidence(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text("[]", encoding="utf-8")
    store = JsonlEventStore(tmp_path / "events.jsonl")
    arena = EvidenceArena(replay)
    service = EvaluationService(
        arena=arena,
        event_store=store,
        backend_hash="backend",
        candidate_root=lambda _version: tmp_path,
        candidate_hash=lambda _version: "candidate",
        opponent_hashes={"rank20": "human"},
    )
    matrix = EvaluationMatrix(("rank20",), ("P0", "P1"), (1,))

    first = service.evaluate_version(
        "v001",
        matrix,
        target_id="rank20",
        locked_regression_ids=(),
    )
    cached = service.evaluate_version(
        "v001",
        matrix,
        target_id="rank20",
        locked_regression_ids=(),
    )

    assert arena.calls == 2
    assert first.observations == cached.observations
    assert first.observations[0].match_id == "v001-rank20-p0-s1"
    assert first.observations[0].points == 1.0
    assert first.observations[0].score_margin == 7.0
    assert first.observations[0].replay_path == replay.resolve()
    assert first.observations[1].points == 0.0


def test_uncached_evaluation_cases_run_concurrently_but_keep_matrix_order(
    tmp_path: Path,
) -> None:
    class ConcurrentArena(EvidenceArena):
        def __init__(self, replay: Path) -> None:
            super().__init__(replay)
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()
            self.overlap = threading.Event()

        def run_case(self, case, candidate_root):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if self.active >= 2:
                    self.overlap.set()
            assert self.overlap.wait(timeout=2.0)
            try:
                return super().run_case(case, candidate_root)
            finally:
                with self.lock:
                    self.active -= 1

    replay = tmp_path / "replay.json"
    replay.write_text("[]", encoding="utf-8")
    arena = ConcurrentArena(replay)
    service = EvaluationService(
        arena=arena,
        event_store=JsonlEventStore(tmp_path / "events.jsonl"),
        backend_hash="backend",
        candidate_root=lambda _version: tmp_path,
        candidate_hash=lambda _version: "candidate",
        opponent_hashes={"rank20": "human"},
        max_workers=4,
    )

    result = service.evaluate_version(
        "v001",
        EvaluationMatrix(("rank20",), ("P0", "P1"), (1, 2)),
        target_id="rank20",
        locked_regression_ids=(),
    )

    assert arena.max_active >= 2
    assert tuple(item.case_id for item in result.observations) == (
        "rank20:P0:1",
        "rank20:P0:2",
        "rank20:P1:1",
        "rank20:P1:2",
    )
    recorded = JsonlEventStore(tmp_path / "events.jsonl").read_all()
    assert tuple(event.payload["case_id"] for event in recorded) == tuple(
        item.case_id for item in result.observations
    )


def test_incomplete_evaluation_event_retains_curriculum_identity(
    tmp_path: Path,
) -> None:
    class IncompleteArena:
        def run_case(self, case, candidate_root):
            return MatchResult(
                case=case,
                status="incomplete",
                result=None,
                points=None,
                score_margin=None,
                rounds=None,
                replay_path=tmp_path / "missing.json",
                trace_path=None,
                events_path=None,
                process_returncodes=(124,),
                error="fixture timeout",
            )

    store = JsonlEventStore(tmp_path / "events.jsonl")
    service = EvaluationService(
        arena=IncompleteArena(),
        event_store=store,
        backend_hash="backend",
        candidate_root=lambda _version: tmp_path,
        candidate_hash=lambda _version: "candidate",
        opponent_hashes={"rank20": "human"},
    )

    service.evaluate_version(
        "v003",
        EvaluationMatrix(("rank20",), ("P1",), (7,)),
        target_id="rank20",
        locked_regression_ids=(),
    )

    payload = store.read_all()[-1].payload
    assert payload["version_id"] == "v003"
    assert payload["opponent_id"] == "rank20"
    assert payload["role"] == "P1"
    assert payload["seed"] == 7
    assert payload["status"] == "incomplete"

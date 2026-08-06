from __future__ import annotations

from pathlib import Path

from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.application.research_service import ResearchService
from agentbench_hl.domain.experience import EvidenceWindow, ExperienceRecord


def experience(
    experience_id: str,
    *,
    verdict: str,
    outcome: str = "loss",
    iteration: int = 1,
    role: str = "P0",
    supersedes: tuple[str, ...] = (),
) -> ExperienceRecord:
    return ExperienceRecord(
        experience_id=experience_id,
        scientific_iteration=iteration,
        target_opponent="rank20",
        role=role,
        verdict=verdict,
        condition=f"{experience_id} 的公开触发条件",
        mechanism=f"{experience_id} 的可证伪机制",
        proposed_change=f"{experience_id} 的候选改动",
        expected_observation=f"{experience_id} 的预期公开现象",
        parent_id="v000",
        candidate_id=f"v{iteration:03d}",
        selection="frontier",
        match_ids=(f"match-{iteration}",),
        evidence_windows=(
            EvidenceWindow(
                f"match-{iteration}",
                f"match-{iteration}:r0001:p0",
                f"match-{iteration}:r0028:p0",
            ),
        ),
        measured_outcome={"result": outcome, "score_margin": -8.0},
        supersedes=supersedes,
    )


def build_service(root: Path) -> ResearchService:
    return ResearchService(
        event_store=JsonlEventStore(root / "events.jsonl"),
        artifact_root=root / "research",
    )


def test_bad_experience_survives_skill_materialization(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    service.record(experience("exp-good", verdict="supported", outcome="win"))
    service.record(experience("exp-bad", verdict="refuted", iteration=2))

    artifacts = service.materialize()

    assert "exp-good" in artifacts.playbook.read_text(encoding="utf-8")
    assert "exp-bad" in artifacts.failed_hypotheses.read_text(encoding="utf-8")
    context = service.context(target="rank20", role="P0", max_records=8)
    assert {record.experience_id for record in context.records} == {
        "exp-good",
        "exp-bad",
    }


def test_supersession_never_deletes_source_record(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    service.record(experience("exp-1", verdict="mixed"))
    service.record(
        experience(
            "exp-2",
            verdict="supported",
            outcome="win",
            iteration=2,
            supersedes=("exp-1",),
        )
    )

    assert [record.experience_id for record in service.read_all()] == [
        "exp-1",
        "exp-2",
    ]
    resumed = build_service(tmp_path)
    assert resumed.read_all() == service.read_all()


def test_materialization_writes_roles_opponents_and_iteration_reports(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    service.record(experience("exp-p0", verdict="inconclusive"))
    service.record(experience("exp-p1", verdict="not_activated", iteration=2, role="P1"))

    artifacts = service.materialize()

    assert "exp-p0" in artifacts.role_p0.read_text(encoding="utf-8")
    assert "exp-p1" in artifacts.role_p1.read_text(encoding="utf-8")
    assert "rank20" in artifacts.opponent_notes.read_text(encoding="utf-8")
    assert len(artifacts.iteration_reports) == 2

from __future__ import annotations

import pytest

from agentbench_hl.domain.experience import EvidenceWindow, ExperienceRecord


def experience(
    experience_id: str,
    *,
    verdict: str = "supported",
    mechanism: str = "集中资源形成可见基地压力",
    evidence_windows: tuple[EvidenceWindow, ...] | None = None,
    supersedes: tuple[str, ...] = (),
) -> ExperienceRecord:
    windows = (
        (EvidenceWindow("match-1", "match-1:r0001:p0", "match-1:r0028:p0"),)
        if evidence_windows is None
        else evidence_windows
    )
    return ExperienceRecord(
        experience_id=experience_id,
        scientific_iteration=1,
        target_opponent="rank20",
        role="P0",
        verdict=verdict,
        condition="公开状态显示敌方右侧基地通路防御较薄",
        mechanism=mechanism,
        proposed_change="在合法位置优先建基础塔并保留闪电金币",
        expected_observation="首次武器后敌方基地 HP 下降",
        parent_id="v000",
        candidate_id="v001",
        selection="frontier",
        match_ids=("match-1",),
        evidence_windows=windows,
        measured_outcome={"result": "win", "score_margin": 8.0},
        supersedes=supersedes,
    )


def test_experience_rejects_missing_replay_evidence() -> None:
    with pytest.raises(ValueError, match="evidence"):
        experience("exp-1", evidence_windows=())


def test_experience_rejects_secret() -> None:
    with pytest.raises(ValueError, match="credential"):
        experience("exp-1", mechanism="token sk-abcdefghijk must never persist")


def test_integration_failure_preserves_hypothesis_without_falsifying_it() -> None:
    record = experience("exp-1", verdict="integration_failure")

    assert record.verdict == "integration_failure"
    assert record.is_strategy_failure is False

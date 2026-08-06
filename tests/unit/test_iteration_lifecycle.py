from __future__ import annotations

from pathlib import Path

from agentbench_hl.application.evaluation_service import EvaluationResult
from agentbench_hl.application.iteration_service import (
    IterationProposal,
    build_iteration_prompt,
    choose_iteration_parent,
    select_iteration_candidate,
)
from agentbench_hl.domain.lineage import CandidateVersion, LineageState


def version(version_id: str, parent_id: str | None) -> CandidateVersion:
    return CandidateVersion(
        version_id=version_id,
        parent_id=parent_id,
        workspace_id=f"w-{version_id}",
        content_hash=version_id * 16,
        object_path=Path(f"/{version_id}"),
        source_hashes={"ai.py": version_id * 16},
        reason="fixture",
    )


def test_failed_complete_candidate_remains_frontier_and_becomes_next_parent() -> None:
    state = LineageState.empty().add_version(version("v000", None)).promote("v000")
    state = state.add_version(version("v001", "v000"))
    evaluation = EvaluationResult(
        observations=(),
        target_solved=False,
        regressions_passed=True,
        promotable=False,
        frontier_eligible=True,
        retry_case_ids=(),
    )
    proposal = IterationProposal(
        condition="敌方首塔形成后资源仍持续空闲",
        mechanism="我方开局投资太晚，未形成同步压力",
        intervention="提前状态机的首轮建塔分支",
        expected_observation="首塔提前且基地首次受伤轮次推迟",
        continuation_rationale="本轮机制只完成第一步，需要继续验证后续升级",
    )

    selected = select_iteration_candidate(state, "v001", evaluation, proposal)

    assert selected.champion_id == "v000"
    assert selected.frontier_id == "v001"
    assert selected.exploration_debt == 1
    assert choose_iteration_parent(selected) == "v001"


def test_iteration_prompt_references_large_evidence_without_embedding_it(
    tmp_path: Path,
) -> None:
    gamepack = tmp_path / "gamepack"
    replay = tmp_path / "replays"
    research = tmp_path / "research"
    for root in (gamepack, replay, research):
        root.mkdir()
    (gamepack / "rules.md").write_text("R" * 1_000_000, encoding="utf-8")
    (replay / "narrative.md").write_text("P" * 1_000_000, encoding="utf-8")
    (research / "PLAYBOOK.md").write_text("E" * 1_000_000, encoding="utf-8")

    prompt = build_iteration_prompt(
        iteration=1,
        parent_id="v000",
        target_id="rank20",
        gamepack_root=gamepack,
        replay_root=replay,
        research_root=research,
        evidence_entries=(f"loss | P1 | seed=1 | {replay / 'v000-rank20-p1-s1/narrative.md'}",),
    )

    assert len(prompt) < 4000
    assert str(gamepack.resolve()) in prompt
    assert str(replay.resolve()) in prompt
    assert str(research.resolve()) in prompt
    assert "网格" in prompt
    assert "必须优先从败局" in prompt
    assert "v000-rank20-p1-s1/narrative.md" in prompt
    assert "后端会独立执行" in prompt
    assert "完成导入与公开 SDK 合法性检查" not in prompt
    assert "R" * 100 not in prompt

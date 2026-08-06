from __future__ import annotations

from pathlib import Path

from agentbench_hl.adapters.antwar2.runtime import Opponent
from agentbench_hl.application.curriculum_service import (
    CurriculumMatch,
    CurriculumService,
)


def opponent(rank: int, *, runnable: bool = True) -> Opponent:
    return Opponent(
        opponent_id=f"rank{rank:02d}",
        rank=rank,
        username=f"human-{rank}",
        score=1000,
        archive=Path(f"rank{rank:02d}.zip"),
        archive_sha256=f"{rank:064x}",
        package_root=Path(f"rank{rank:02d}"),
        runnable=runnable,
        entry_command=("python", "main.py") if runnable else None,
        exclusion_diagnostic=None if runnable else "missing entry",
    )


def win(opponent_id: str, role: str) -> CurriculumMatch:
    return CurriculumMatch(opponent_id, role, 1, "complete", "win")


def test_curriculum_selects_weakest_runnable_unsolved_rank() -> None:
    service = CurriculumService(
        opponents=(opponent(20), opponent(19, runnable=False), opponent(18)),
        roles=("P0", "P1"),
        seeds=(1,),
        matches=(win("rank20", "P0"), win("rank20", "P1")),
    )

    assert service.default_target().opponent_id == "rank18"
    assert service.status().locked_regression_ids == ("rank20",)


def test_target_is_not_solved_when_one_role_is_incomplete() -> None:
    service = CurriculumService(
        opponents=(opponent(20),),
        roles=("P0", "P1"),
        seeds=(1,),
        matches=(
            win("rank20", "P0"),
            CurriculumMatch("rank20", "P1", 1, "incomplete", None),
        ),
    )

    assert service.status().by_opponent["rank20"].state == "incomplete"


def test_opponent_is_solved_at_half_win_rate() -> None:
    service = CurriculumService(
        opponents=(opponent(20), opponent(18)),
        roles=("P0", "P1"),
        seeds=(1, 2),
        matches=(
            win("rank20", "P0"),
            win("rank20", "P1"),
            CurriculumMatch("rank20", "P0", 2, "complete", "loss"),
            CurriculumMatch("rank20", "P1", 2, "complete", "loss"),
        ),
    )

    assert service.status().by_opponent["rank20"].state == "solved"
    assert service.default_target().opponent_id == "rank18"

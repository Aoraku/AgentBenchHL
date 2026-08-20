"""顺序征服课程（实验 5）的判据测试。

这些测试锁住的是"什么时候才算打赢了、可以换下一个对手"——判据一旦漂移，
不同 run 的进度就不可比，实验 5 的结论也就无效了。
"""

from __future__ import annotations

from agentbench_hl.application.conquest import (
    AdvanceRule,
    RoundResult,
    evaluate,
    round_results,
)
from agentbench_hl.application.opponent_policy import LadderEntry, build_policy


def ladder(*ranks: int) -> tuple[LadderEntry, ...]:
    return tuple(
        LadderEntry(opponent_id=f"rank{rank:02d}", rank=rank, score=2000.0 - rank * 30)
        for rank in ranks
    )


# ----------------------------------------------------------------- 目标序列


def test_ladder_up_starts_at_the_configured_rank_and_walks_toward_the_top() -> None:
    policy = build_policy("ladder_up", ladder(*range(1, 21)), start_rank=10)

    sequence = policy.target_sequence()

    assert sequence[0] == "rank10"  # 从第 10 名开始
    assert sequence[:4] == ("rank10", "rank09", "rank08", "rank07")
    assert sequence[-1] == "rank01"  # 一路打到榜首
    assert len(sequence) == 10  # 不含 11..20（弱于起点的不打）


def test_ladder_down_starts_at_the_top_and_walks_toward_the_tail() -> None:
    policy = build_policy("ladder_down", ladder(*range(1, 6)), start_rank=1)

    assert policy.target_sequence() == ("rank01", "rank02", "rank03", "rank04", "rank05")


def test_ladder_up_falls_back_when_the_start_rank_is_beyond_the_ladder() -> None:
    # 榜单只有 3 人却要求从第 10 名起：应从最弱的可用对手起，而不是空序列。
    policy = build_policy("ladder_up", ladder(1, 2, 3), start_rank=10)

    assert policy.target_sequence() == ("rank03", "rank02", "rank01")


def test_ladder_skips_rank_gaps_left_by_the_runnability_audit() -> None:
    # 审计淘汰了 rank07/rank09 ⇒ 榜单里没有它们，序列必须自然跳过。
    policy = build_policy("ladder_up", ladder(1, 2, 3, 4, 5, 6, 8, 10), start_rank=10)

    assert policy.target_sequence() == (
        "rank10", "rank08", "rank06", "rank05", "rank04", "rank03", "rank02", "rank01",
    )


def test_non_sequential_policies_expose_no_target_sequence() -> None:
    for name in ("self_decide", "fixed_top", "fixed_rank", "random", "k_diverse"):
        policy = build_policy(name, ladder(1, 2, 3), target_rank=2)
        assert policy.target_sequence() == (), name


# --------------------------------------------------------------- 推进判据


def test_target_advances_only_after_the_current_target_is_beaten() -> None:
    sequence = ("rank10", "rank09", "rank08")
    results = (
        RoundResult(iteration=1, opponent_id="rank10", played=2, points=2.0),  # 2 胜 ⇒ 达标
        RoundResult(iteration=2, opponent_id="rank09", played=2, points=0.0),  # 全负
    )

    state = evaluate(sequence, results)

    assert state.cleared == 1
    assert state.cleared_ids == ("rank10",)
    assert state.target_id == "rank09"
    assert state.finished is False


def test_beating_other_opponents_does_not_move_the_cursor() -> None:
    # 关键反例：偶然赢了个弱手不该跳级（旧实现会 +1）。
    sequence = ("rank10", "rank09")
    results = (
        RoundResult(iteration=1, opponent_id="rank20", played=4, points=4.0),
        RoundResult(iteration=2, opponent_id="rank18", played=4, points=4.0),
    )

    state = evaluate(sequence, results)

    assert state.cleared == 0
    assert state.target_id == "rank10"


def test_single_lucky_win_is_not_enough_when_min_matches_is_two() -> None:
    results = (RoundResult(iteration=1, opponent_id="rank10", played=1, points=1.0),)

    state = evaluate(("rank10", "rank09"), results)

    assert state.cleared == 0  # 只打了 1 局，样本不足
    assert state.target_id == "rank10"


def test_streak_requirement_demands_consecutive_qualifying_rounds() -> None:
    sequence = ("rank10", "rank09")
    rule = AdvanceRule(min_matches=2, win_rate=0.6, streak=2)
    mixed = (
        RoundResult(iteration=1, opponent_id="rank10", played=2, points=2.0),  # 达标
        RoundResult(iteration=2, opponent_id="rank10", played=2, points=0.0),  # 断了
        RoundResult(iteration=3, opponent_id="rank10", played=2, points=2.0),  # 重新开始
    )

    assert evaluate(sequence, mixed, rule=rule).cleared == 0

    consecutive = mixed + (
        RoundResult(iteration=4, opponent_id="rank10", played=2, points=2.0),
    )
    state = evaluate(sequence, consecutive, rule=rule)
    assert state.cleared == 1
    assert state.target_id == "rank09"


def test_draws_count_as_half_and_can_satisfy_a_low_threshold() -> None:
    results = (RoundResult(iteration=1, opponent_id="rank10", played=2, points=1.0),)

    assert evaluate(("rank10",), results, rule=AdvanceRule(win_rate=0.5)).cleared == 1
    assert evaluate(("rank10",), results, rule=AdvanceRule(win_rate=0.6)).cleared == 0


def test_finished_state_reports_no_target() -> None:
    results = (RoundResult(iteration=1, opponent_id="rank01", played=2, points=2.0),)

    state = evaluate(("rank01",), results)

    assert state.finished is True
    assert state.target_id is None
    assert state.cleared == 1


def test_policy_selects_the_conquest_target_for_the_current_cleared_count() -> None:
    policy = build_policy("ladder_up", ladder(*range(1, 11)), start_rank=10)

    assert policy.select(iteration=1, k=2, cleared=0) == ("rank10",)
    assert policy.select(iteration=5, k=2, cleared=3) == ("rank07",)
    # 全部征服后停在最后一个目标（继续巩固），不越界。
    assert policy.select(iteration=99, k=2, cleared=50) == ("rank01",)


# ------------------------------------------------------- 逐局记录的聚合


def test_round_results_group_by_iteration_and_ignore_infrastructure_failures() -> None:
    rows = [
        {"iteration": 1, "opponent_id": "rank10", "status": "complete", "points": 1.0},
        {"iteration": 1, "opponent_id": "rank10", "status": "complete", "points": 0.0},
        {"iteration": 1, "opponent_id": "rank10", "status": "infra_error", "points": None},
        {"iteration": 2, "opponent_id": "rank10", "status": "complete", "points": 1.0},
    ]

    results = round_results(rows)

    assert [(item.iteration, item.played, item.points) for item in results] == [
        (1, 2, 1.0),
        (2, 1, 1.0),
    ]
    assert results[0].score_rate == 0.5

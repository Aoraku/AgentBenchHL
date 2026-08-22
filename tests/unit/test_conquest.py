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
from agentbench_hl.application.opponent_policy import (
    LadderEntry,
    OpponentHistory,
    build_policy,
    canonical_policy_name,
)


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
    # 新的四个主设置（random/self/progress/fix）与历史别名都不是"单目标顺序征服"，
    # 所以它们没有 target_sequence；只有 ladder_up / ladder_down 有。
    for name in ("self", "self_decide", "fix", "fixed_top", "fixed_rank", "random", "k_diverse"):
        policy = build_policy(name, ladder(1, 2, 3), target_rank=2)
        assert not hasattr(policy, "target_sequence"), name


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


def test_policy_selects_the_conquest_target_from_the_beaten_history() -> None:
    """``ladder_up`` 的游标现在从**战绩**推导，不再由调用方传 ``cleared``。

    为什么改：k=1 × b 个对手之后，"已征服几个"不是一个能从外部传进来的标量
    （b 个槽位各自有进度）。让策略自己读战绩，进度对所有策略都是同一套口径，
    断点续跑也不会因为调用方忘了传 cleared 而把游标重置。
    """

    policy = build_policy("ladder_up", ladder(*range(1, 11)), start_rank=10)

    assert policy.select(iteration=1, batch=1) == ("rank10",)

    # 打赢 rank10 / 09 / 08（各 2 局全胜）后，目标推进到 rank07。
    beaten = OpponentHistory(
        {
            "rank10": {"played": 2.0, "points": 2.0},
            "rank09": {"played": 2.0, "points": 2.0},
            "rank08": {"played": 2.0, "points": 2.0},
        }
    )
    assert policy.select(iteration=5, batch=1, history=beaten) == ("rank07",)

    # 全部征服后停在最后一个目标（继续巩固），不越界。
    everything = OpponentHistory(
        {f"rank{rank:02d}": {"played": 2.0, "points": 2.0} for rank in range(1, 11)}
    )
    assert policy.select(iteration=99, batch=1, history=everything) == ("rank01",)


# ------------------------------------------- 四种对手选择方式（k=1 × b 个对手）


def test_fix_takes_the_top_b_of_the_leaderboard() -> None:
    policy = build_policy("fix", ladder(*range(1, 11)))

    assert policy.select(iteration=1, batch=4) == ("rank01", "rank02", "rank03", "rank04")
    # b 可消融：b=1 时退化成"只打榜首"。
    assert policy.select(iteration=1, batch=1) == ("rank01",)
    # 老名字 fixed_top 必须等价（已完成的 run 要能复现）。
    assert build_policy("fixed_top", ladder(*range(1, 11))).select(
        iteration=1, batch=4
    ) == ("rank01", "rank02", "rank03", "rank04")


def test_random_picks_b_distinct_opponents_and_changes_each_iteration() -> None:
    policy = build_policy("random", ladder(*range(1, 21)), seed=7)

    first = policy.select(iteration=1, batch=4)
    assert len(first) == 4
    assert len(set(first)) == 4, "同一轮内不允许重复（那会浪费一个对局槽位）"

    # 同一轮可复现（seed + iteration 决定），跨轮换人（抗过拟合）。
    assert policy.select(iteration=1, batch=4) == first
    assert policy.select(iteration=2, batch=4) != first


def test_random_does_not_crash_when_the_ladder_is_shorter_than_b() -> None:
    policy = build_policy("random", ladder(1, 2), seed=1)

    assert set(policy.select(iteration=1, batch=8)) == {"rank01", "rank02"}


def test_progress_window_starts_at_the_configured_rank() -> None:
    policy = build_policy("progress", ladder(*range(1, 31)), start_rank=20)

    # 窗口从第 20 名往榜首方向铺开 b 个。
    assert policy.select(iteration=1, batch=4) == ("rank20", "rank19", "rank18", "rank17")


def test_progress_advances_a_slot_and_skips_ranks_already_beaten() -> None:
    """核心用例：19 已晋级到 18 之后，20 晋级必须跳过 19 直接到 16。

    不跳的话 b 个槽位会互相踩（20 去 19、19 去 18…），实际只在少数几个对手
    上打转，而"稳定上升"这件事从曲线上完全看不出来。
    """

    policy = build_policy(
        "progress", ladder(*range(1, 31)), start_rank=20, advance_win_rate=0.75
    )
    # rank20 与 rank19 都已稳定打赢（得分率 1.0 > 0.75）。
    history = OpponentHistory(
        {
            "rank20": {"played": 4.0, "points": 4.0},
            "rank19": {"played": 4.0, "points": 4.0},
        }
    )

    window = policy.select(iteration=3, batch=4, history=history)

    assert "rank20" not in window and "rank19" not in window, "打赢的不再重复打"
    assert len(set(window)) == 4, "槽位之间不许撞人"
    # 原本 [20,19,18,17] → 18/17 留下，20 与 19 各自往前找到还没打过的 16/15。
    assert set(window) == {"rank18", "rank17", "rank16", "rank15"}


def test_progress_keeps_an_unbeaten_opponent_in_place() -> None:
    policy = build_policy(
        "progress", ladder(*range(1, 31)), start_rank=20, advance_win_rate=0.75
    )
    # 打了 4 局只拿 1 分（0.25 ≤ 0.75）：不算稳定击败，留在原位继续研究。
    history = OpponentHistory({"rank20": {"played": 4.0, "points": 1.0}})

    assert "rank20" in policy.select(iteration=5, batch=4, history=history)


def test_progress_with_b_equals_one_is_the_old_one_by_one_ladder() -> None:
    policy = build_policy(
        "progress", ladder(*range(1, 21)), start_rank=20, advance_win_rate=0.75
    )

    assert policy.select(iteration=1, batch=1) == ("rank20",)
    history = OpponentHistory({"rank20": {"played": 2.0, "points": 2.0}})
    assert policy.select(iteration=2, batch=1, history=history) == ("rank19",)


def test_self_leaves_the_choice_to_the_agent() -> None:
    policy = build_policy("self", ladder(*range(1, 11)))

    assert policy.select(iteration=1, batch=4) == (), "空 = 框架不干预"
    instruction = policy.instruction(iteration=1, batch=4)
    assert "selected_rivals" in instruction
    assert "4" in instruction, "要告诉 agent 挑几个"


def test_self_instruction_surfaces_the_beaten_and_stuck_opponents() -> None:
    """``self`` 要真的能"自己决定"，就得看得到战绩摘要。

    只给一份排行榜的话，agent 无从判断"哪些已经稳定赢了、可以不再打"，
    于是它只能按名次盲选——那和 fix 就没区别了，消融变量不干净。
    """

    policy = build_policy("self", ladder(*range(1, 11)))
    history = OpponentHistory(
        {
            "rank09": {"played": 4.0, "points": 4.0},  # 稳定赢
            "rank01": {"played": 4.0, "points": 0.0},  # 一直输
        }
    )

    instruction = policy.instruction(iteration=3, batch=4, history=history)

    assert "rank09" in instruction
    assert "rank01" in instruction


def test_legacy_policy_names_are_canonicalised() -> None:
    assert canonical_policy_name("fixed_top") == "fix"
    assert canonical_policy_name("self_decide") == "self"
    # 新名字与未知别名原样返回（未知名字由 build_policy 报错，不在这里吞掉）。
    assert canonical_policy_name("progress") == "progress"


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

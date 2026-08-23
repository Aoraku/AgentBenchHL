"""迭代中的 Elo 反解：口径与失效模式。

这一条曲线回答的问题是 **"这一版插进冻结人类池会排在哪"**，
用的是"该候选这一轮真打过的那几局 + 池内锚点"，一分钱不多花。

它与慢通道（每 N 轮让中间版本打完全池）**必须同尺度**，否则两条线画在
同一根轴上没有意义。所以两边都走 ``domain.pool_elo.estimate_pool_elo``。

被修掉的失效模式：钳位造成的假平坦曲线
--------------------------------------
原实现是逐对手 ``anchor + 400·log10(p/(1-p))`` 再取平均，并把 p 钳到
``[0.02, 0.98]`` 以避免 ``log(0)``。后果是**全败时估计变成常数**：

    fix 组固定打榜单前 4 名（2278 / 2118 / 2051 / 1983，均值 2107.5），
    胜率恒 0 → p 恒被钳成 0.02 → 400·log10(0.02/0.98) = −676
    → 反解恒等于 2107.5 − 676 = 1431.4

实测 14 轮全是 ``1431.37``，一位小数都不差。看图会得出"这一组完全没在学"，
而同期平均分差从 −36.12 收窄到 −28.12（最好的一局 −32 → −18，改善 44%）——
agent 确实在稳步变强，只是还没跨过"能赢前 4 名"那道门槛。

正则 MLE 没有这个毛病：全败时 θ 由先验（2 场对平均锚点的虚拟平局）拉住而
不是被硬钳，且估计值取决于**对手锚点的具体分布**，所以换了对手它就会动。
"""

from __future__ import annotations

from agentbench_hl.domain.pool_elo import estimate_pool_elo

# antwar2 冻结池的真实锚点（前 4 名与中段），用来复现线上口径。
TOP4 = {
    "rank01": 2278.1,
    "anon_a": 2118.0,
    "anon_b": 2050.6,
    "anon_c": 1983.1,
}
MIDDLE = {"mid_20": 1685.4, "mid_30": 1495.4, "mid_40": 1426.7, "mid_50": 1378.3}


def _sweep(opponents: dict[str, float], points: float) -> float | None:
    """对 opponents 每人两局（双座次），每局得分 points。"""

    rows = [
        {"opponent_id": name, "points": points}
        for name in opponents
        for _ in range(2)
    ]
    estimate = estimate_pool_elo(rows, opponents)
    return None if estimate is None else estimate.elo


def test_all_losses_do_not_collapse_to_a_constant() -> None:
    """全败时，估计必须仍然**依赖对手强度**，而不是一个固定的钳位值。

    这是那条假平坦曲线的直接反面：打前 4 名全败与打中段全败，
    水平结论显然不同（输给 2278 的人比输给 1378 的人强），
    估计值必须能区分这两种情况。
    """

    lost_to_top = _sweep(TOP4, 0.0)
    lost_to_middle = _sweep(MIDDLE, 0.0)

    assert lost_to_top is not None and lost_to_middle is not None
    assert lost_to_top > lost_to_middle, (
        "输给强者的估计必须高于输给弱者；两者相等说明估计被钳位或与对手无关"
    )


def test_all_losses_stay_below_every_opponent() -> None:
    """全败 = 没有证据表明能赢任何人，估计不该高于最弱的那个对手。"""

    elo = _sweep(TOP4, 0.0)

    assert elo is not None
    assert elo < min(TOP4.values())


def test_all_wins_stay_above_every_opponent_and_remain_finite() -> None:
    """全胜也必须有限（不能 +inf），且高于最强的对手。"""

    elo = _sweep(TOP4, 1.0)

    assert elo is not None
    assert min(TOP4.values()) < elo
    assert elo < 10_000, "正则先验必须把全胜时的 θ 拉回有限范围"


def test_more_losses_push_the_estimate_down() -> None:
    """同一批对手上输得越多，估计越低（更多证据 = 更强的下压）。

    钳位实现做不到这一点：p 一旦触到下界，再多的败绩都不会改变结果。
    """

    two_losses = estimate_pool_elo(
        [{"opponent_id": "rank01", "points": 0.0} for _ in range(2)], TOP4
    )
    twenty_losses = estimate_pool_elo(
        [{"opponent_id": "rank01", "points": 0.0} for _ in range(20)], TOP4
    )

    assert two_losses is not None and twenty_losses is not None
    assert twenty_losses.elo < two_losses.elo
    # 可信度线索也要跟着走，否则无法判断一个数字该不该信。
    assert twenty_losses.matches == 20
    assert twenty_losses.anchored_matches == 20


def test_beating_the_weak_and_losing_to_the_strong_lands_in_between() -> None:
    """打赢弱者、输给强者时，估计应落在两者之间。

    这是"逐对手反解再平均"与"总胜率配平均锚点"的分水岭：后者会把
    (赢 1378 / 输 2278) 的 0.5 胜率报成锚点均值 1828，明显偏高。
    """

    rows = [
        {"opponent_id": "mid_50", "points": 1.0},  # 1378.3 赢
        {"opponent_id": "mid_50", "points": 1.0},
        {"opponent_id": "rank01", "points": 0.0},  # 2278.1 输
        {"opponent_id": "rank01", "points": 0.0},
    ]
    anchors = {"mid_50": MIDDLE["mid_50"], "rank01": TOP4["rank01"]}

    estimate = estimate_pool_elo(rows, anchors)

    assert estimate is not None
    assert MIDDLE["mid_50"] < estimate.elo < TOP4["rank01"]


def test_opponents_without_anchors_are_skipped_not_averaged() -> None:
    """没有池内 Elo 的对手必须**跳过**，不能当成平均水平混进去。

    宁可样本少，也不要用编出来的锚点污染整条曲线。
    """

    rows = [
        {"opponent_id": "rank01", "points": 0.0},
        {"opponent_id": "unknown_player", "points": 1.0},
    ]

    estimate = estimate_pool_elo(rows, {"rank01": TOP4["rank01"], "unknown_player": None})

    assert estimate is not None
    assert estimate.matches == 2
    assert estimate.anchored_matches == 1, "只有 1 局有锚点可用"


def test_no_usable_sample_reports_none() -> None:
    """一局锚点都没有时返回 None —— 绝不编一个数。"""

    assert estimate_pool_elo([], TOP4) is None
    assert estimate_pool_elo([{"opponent_id": "ghost", "points": 1.0}], {"ghost": None}) is None

"""分差面板的适用性判据。

为什么需要这道闸门
------------------
**分差不是所有游戏都有意义的。** 实测 8 个游戏的 ``score_margin`` 取值个数：

===========  ==================  ====================================
游戏         取值个数 / 局数     值域
===========  ==================  ====================================
antwar       54 / 344            −50 … +50      （基地 HP 差，连续）
antwar2      55 / 440            −40 … +22
snakego      13 /  16            −207 … −127
rollman       9 /  16            0 … 1724
miracle       4 /  16            −30000 … +28000（准连续，样本少）
generals      2 /  16            {−1, +1}       ← **分差就是胜负**
lostspace     2 /  16            {−3, 0}
aquawar       2 /  15            {−2, 0}
===========  ==================  ====================================

generals 的分差字面上等于胜负（赢 +1 / 输 −1），画它等于把胜率图再画一遍。

硬画的代价不是"多一张没用的图"，而是**误导**：一条在 {−3, 0} 之间跳的折线
看起来就像"分差一直没改善"，而事实是这个游戏没有"分差"这个连续量。
所以要么如实留白并说明，要么不画。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_learning_curves import MARGIN_DISTINCT_MIN, Run  # noqa: E402


def _run(game: str, margins: set[float]) -> Run:
    return Run(
        game=game,
        points=[],
        pool_size=0,
        pool_top_elo=None,
        pool_median_elo=None,
        evaluated=0,
        run_id=f"run-{game}",
        raw_margin_values=frozenset(margins),
    )


def test_continuous_margin_games_are_plotted() -> None:
    """antwar / antwar2 / snakego / rollman 的分差是真正的连续量。"""

    assert _run("antwar", set(range(-50, 51))).margin_is_informative
    assert _run("antwar2", set(range(-40, 23))).margin_is_informative
    assert _run("snakego", {-207.0, -191.0, -187.0, -185.0, -181.0, -179.0}).margin_is_informative
    assert _run("rollman", {0.0, 658.0, 855.0, 952.0, 1273.0, 1607.0}).margin_is_informative


def test_generals_margin_is_just_the_verdict() -> None:
    """generals 的分差取值只有 {−1, +1} —— 它**就是**胜负。

    这是最该被拦住的一个：画出来的折线在 −1 和 +1 之间跳，
    和胜率图一模一样，但看图的人会以为那是一个独立的指标。
    """

    assert not _run("generals", {-1.0, 1.0}).margin_is_informative


def test_tiny_discrete_margins_are_not_informative() -> None:
    """aquawar {−2, 0} 与 lostspace {−3, 0} 撑不起一条曲线。"""

    assert not _run("aquawar", {-2.0, 0.0}).margin_is_informative
    assert not _run("lostspace", {-3.0, 0.0}).margin_is_informative


def test_threshold_is_just_above_the_verdict_cardinality() -> None:
    """闸门必须严格高于"胜/平/负"三档。

    胜负本身最多 3 种取值，所以分差要带来新信息就必须多于 3 种。
    阈值定在 3 会把"胜/平/负分差"（例如 {−1, 0, +1}）当成有效连续量放过去。
    """

    assert MARGIN_DISTINCT_MIN == 4
    assert not _run("x", {-1.0, 0.0, 1.0}).margin_is_informative
    assert _run("x", {-1.0, 0.0, 0.5, 1.0}).margin_is_informative


def test_miracle_with_few_samples_still_counts() -> None:
    """miracle 只有 4 种取值，但值域 ±3 万 —— 它是准连续量，样本少而已。

    用"取值个数"而不是"值域宽度"作判据是有意为之：值域宽度需要按游戏
    定标（rollman 的 1724 与 aquawar 的 2 不可比），而取值个数是无量纲的。
    4 种取值刚好过线，随着轮数增加会更明显。
    """

    assert _run("miracle", {-30000.0, -29996.0, 0.0, 28000.0}).margin_is_informative


def test_no_margin_data_is_not_treated_as_uninformative() -> None:
    """还没有任何对局时（空集）不该显示"此游戏不适用"。

    那是"数据还没来"，与"这个游戏没有分差这个量"是两件不同的事，
    混在一起会让新起的 run 一开始就被误判。绘图侧因此额外检查
    ``raw_margin_values`` 非空才显示"不适用"说明。
    """

    empty = _run("antwar2", set())
    assert not empty.margin_is_informative  # 数据不足，确实不画
    assert not empty.raw_margin_values  # 但原因是"没有数据"，绘图侧据此区分

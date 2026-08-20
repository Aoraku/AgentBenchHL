"""候选相对**固定人类池**的 Elo：累积锚定极大似然估计。

为什么需要它
------------
逐轮指标里原来有两个 Elo 位：

* ``elo_vs_opponent``：以**当轮对手**的已知 Elo 为锚、按当轮胜率做 logistic 反推。
  问题是有序课程（``ladder_up``）每清掉一个人就换对手，锚点跟着跳；
  再加上一轮只有 ``rollout_k × roles × seeds`` 局，方差很大。
  于是曲线上会看到"打赢弱的→分数高、换强的→分数掉"，那是**换尺子**造成的假下降，
  不是能力退步。
* ``fixed_pool_elo``：原来恒为 ``None``，注释说由慢通道回填，但仓库里没有回填实现。
  而这恰恰是唯一跨轮、跨游戏可比的量——"刷到 SOTA"只能用它说话。

做法
----
把该 run **迄今为止全部** complete 的官方对局汇总（对手可以各不相同），
用每个对手在人类池里的 ``measured`` Elo 作为**固定锚点**，对候选强度 θ 做一维
Bradley–Terry 极大似然估计：

.. math::

    P(\\text{候选胜}) = \\frac{1}{1 + 10^{(a_i - \\theta)/400}}

θ 与所有对手共享同一把尺子（人类池的 measured Elo 刻度），所以：

* 换对手不会让曲线跳——锚点变了但尺子没变；
* 对局越多越准（每轮都把历史局算进来，而不是只看当轮）；
* **不需要额外机时**：用的就是已经打完的那些局。

与"真慢通道"的关系：这不是重新跑一遍全池对战，而是基于已有对局的估计。
所以字段命名上诚实标注 ``method="anchored_mle"``、``matches`` 给出样本量，
供分析时判断可信度。

边界情况
--------
全胜 / 全败时似然没有有限最大值（θ→±∞）。这里加**两场虚拟平局**
（对手取锚点均值）做正则，等价于一个很弱的先验："在没有反例之前，
不要断言候选比对手强 1000 分以上"。样本一多，先验影响迅速衰减。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

__all__ = ["PoolEloEstimate", "estimate_pool_elo"]

# Elo 刻度：400 分 = 10 倍胜率比。
SCALE = 400.0
# 正则强度：等价于额外 N 场对"平均锚点"的平局。
PRIOR_DRAWS = 2.0
# θ 的搜索区间（相对锚点均值）。人类池 Elo 跨度约 -1000~2000，±3000 足够宽。
SEARCH_HALF_WIDTH = 3000.0
TOLERANCE = 1e-4


@dataclass(frozen=True)
class PoolEloEstimate:
    """一次估计的结果与它的可信度线索。"""

    elo: float
    matches: int
    anchored_matches: int
    score_rate: float
    anchor_mean: float
    method: str = "anchored_mle"

    def as_dict(self) -> dict[str, object]:
        return {
            "elo": round(self.elo, 2),
            "matches": self.matches,
            "anchored_matches": self.anchored_matches,
            "score_rate": round(self.score_rate, 4),
            "anchor_mean": round(self.anchor_mean, 2),
            "method": self.method,
        }


def _expected_score(theta: float, anchor: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (anchor - theta) / SCALE))


def _score_derivative(theta: float, samples: Sequence[tuple[float, float]]) -> float:
    """对数似然对 θ 的导数（去掉正的常数因子 ln10/400）。

    BT/Elo 模型下它就是"实际得分 − 期望得分"的总和，单调递减，
    所以可以直接二分求零点，不需要牛顿法的二阶导与步长控制。
    """

    total = 0.0
    for anchor, score in samples:
        total += score - _expected_score(theta, anchor)
    return total


def estimate_pool_elo(
    matches: Iterable[Mapping[str, object]],
    anchors: Mapping[str, float | None],
    *,
    score_key: str = "points",
    opponent_key: str = "opponent_id",
) -> PoolEloEstimate | None:
    """估计候选在人类池刻度上的 Elo。

    ``matches`` 是逐局记录（只用 ``status == "complete"`` 的局，调用方筛好），
    每条要有对手 id 与该局得分（1 胜 / 0.5 平 / 0 负）。
    ``anchors`` 是 ``{对手 id: 该对手的池内 Elo}``；**没有锚点的对手会被跳过**——
    宁可样本少，也不要把"未知强度"当成平均水平混进去。

    返回 ``None`` 表示没有任何可用样本（此时调用方应当继续记 ``None``，
    而不是编一个数）。
    """

    samples: list[tuple[float, float]] = []
    total = 0
    score_sum = 0.0
    for row in matches:
        total += 1
        opponent = row.get(opponent_key)
        if opponent is None:
            continue
        anchor = anchors.get(str(opponent))
        if anchor is None:
            continue
        raw = row.get(score_key)
        try:
            score = float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        score = min(1.0, max(0.0, score))
        samples.append((float(anchor), score))
        score_sum += score

    if not samples:
        return None

    anchor_mean = sum(anchor for anchor, _ in samples) / len(samples)
    # 正则：两场对"平均锚点"的虚拟平局，保证全胜/全败时 θ 仍然有限。
    regularised = list(samples)
    regularised.extend([(anchor_mean, 0.5)] * int(PRIOR_DRAWS))

    low = anchor_mean - SEARCH_HALF_WIDTH
    high = anchor_mean + SEARCH_HALF_WIDTH
    # 导数在 θ 上严格递减：low 处为正、high 处为负，二分必然收敛。
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _score_derivative(middle, regularised) > 0:
            low = middle
        else:
            high = middle
        if high - low < TOLERANCE:
            break

    return PoolEloEstimate(
        elo=0.5 * (low + high),
        matches=total,
        anchored_matches=len(samples),
        score_rate=score_sum / len(samples),
        anchor_mean=anchor_mean,
    )

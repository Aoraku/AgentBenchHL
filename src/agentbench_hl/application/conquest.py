"""顺序征服进度 —— "能稳定击败一个就切到下一个" 的可复现判据。

研究动机
--------
实验 5 的两个课程设置是**有序**的：

* ``ladder_up``   : 从第 10 名开始往上打（10 → 9 → … → 1）
* ``ladder_down`` : 从第 1 名开始往下打（1 → 2 → 3 → …）

"打过了就换人"这句话必须落成一个**确定性判据**，否则同一份日志会被解释成不同的
进度，曲线也就不可比。这里把判据固定成三个参数：

* ``min_matches`` : 本轮对当前目标至少打过几局（默认 2，含两侧座次）
* ``win_rate``    : 本轮对当前目标的得分率下限（默认 0.6；平局算 0.5）
* ``streak``      : 需要连续几轮达标（默认 1；设 2 即"稳定击败"更严格的版本）

进度只从**事件流**推导（按迭代顺序回放），因此断点续跑、事后复盘、跨机器复现都会
得到同一个游标位置。不做"任意赢过一个对手就 +1"这种与顺序无关的计数——那会让
agent 在第 3 轮偶然赢了个弱手就跳过两级目标。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

DEFAULT_MIN_MATCHES = 2
DEFAULT_WIN_RATE = 0.6
DEFAULT_STREAK = 1


@dataclass(frozen=True)
class RoundResult:
    """一轮里对某个对手的成绩（只记可用于判定的字段）。"""

    iteration: int
    opponent_id: str
    played: int
    points: float  # 胜=1 平=0.5 负=0 的累加

    @property
    def score_rate(self) -> float | None:
        return None if self.played <= 0 else self.points / self.played


@dataclass(frozen=True)
class ConquestState:
    cleared: int
    cleared_ids: tuple[str, ...]
    target_id: str | None
    target_index: int
    streak: int
    finished: bool

    @property
    def summary(self) -> dict[str, object]:
        return {
            "cleared": self.cleared,
            "cleared_ids": list(self.cleared_ids),
            "target_id": self.target_id,
            "target_index": self.target_index,
            "streak": self.streak,
            "finished": self.finished,
        }


@dataclass(frozen=True)
class AdvanceRule:
    min_matches: int = DEFAULT_MIN_MATCHES
    win_rate: float = DEFAULT_WIN_RATE
    streak: int = DEFAULT_STREAK

    def satisfied(self, result: RoundResult) -> bool:
        rate = result.score_rate
        if rate is None or result.played < self.min_matches:
            return False
        return rate >= self.win_rate


def round_results(rows: Iterable[Mapping[str, object]]) -> tuple[RoundResult, ...]:
    """把逐局记录聚合成"每轮 × 每对手"的成绩。

    ``rows`` 用 ``iteration`` / ``opponent_id`` / ``status`` / ``points`` 四个字段；
    非 ``complete`` 的局（基础设施故障）不计入分母，否则一次沙箱故障会把达标
    判定拉低，进度停滞的原因就变得不可解释。
    """

    buckets: dict[tuple[int, str], list[float]] = {}
    order: list[tuple[int, str]] = []
    for row in rows:
        if row.get("status") != "complete":
            continue
        iteration_value = row.get("iteration")
        opponent = row.get("opponent_id")
        if not isinstance(iteration_value, int) or not isinstance(opponent, str):
            continue
        key = (iteration_value, opponent)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        points = row.get("points")
        buckets[key].append(float(points) if isinstance(points, (int, float)) else 0.0)
    results = [
        RoundResult(
            iteration=iteration,
            opponent_id=opponent,
            played=len(buckets[(iteration, opponent)]),
            points=sum(buckets[(iteration, opponent)]),
        )
        for iteration, opponent in order
    ]
    results.sort(key=lambda item: (item.iteration, item.opponent_id))
    return tuple(results)


def evaluate(
    sequence: Sequence[str],
    results: Sequence[RoundResult],
    *,
    rule: AdvanceRule | None = None,
) -> ConquestState:
    """沿 ``sequence`` 推进游标，返回当前目标与已征服数。

    只有"对**当前目标**的成绩"才能推动游标；对其它对手的战绩一律忽略
    （诊断性对局、影子对局都不该影响课程进度）。
    """

    advance = rule or AdvanceRule()
    if not sequence:
        return ConquestState(0, (), None, 0, 0, True)

    index = 0
    streak = 0
    cleared: list[str] = []
    by_iteration: dict[int, dict[str, RoundResult]] = {}
    for result in results:
        by_iteration.setdefault(result.iteration, {})[result.opponent_id] = result

    for iteration in sorted(by_iteration):
        if index >= len(sequence):
            break
        target = sequence[index]
        result = by_iteration[iteration].get(target)
        if result is None:
            continue  # 这一轮没打当前目标（例如框架被覆盖或协议错误轮）
        if advance.satisfied(result):
            streak += 1
            if streak >= advance.streak:
                cleared.append(target)
                index += 1
                streak = 0
        else:
            streak = 0  # 达标必须连续，否则"稳定"二字没有意义

    finished = index >= len(sequence)
    return ConquestState(
        cleared=len(cleared),
        cleared_ids=tuple(cleared),
        target_id=None if finished else sequence[index],
        target_index=index,
        streak=streak,
        finished=finished,
    )

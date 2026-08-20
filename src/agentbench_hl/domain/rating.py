"""从对局图反解评分（Bradley–Terry 极大似然，Elo 刻度）。

**为什么不能只用胜率**：稀疏配对下每个选手遇到的对手强度不同，胜率不可比。
把整张对局图一起拟合才能得到自洽的评分。

模型（与 Elo 同一族）：

``P(a 胜 b) = 1 / (1 + 10^{-(r_a - r_b)/400})``

等价于 Bradley–Terry：``P = π_a / (π_a + π_b)``，``r = 400·log10(π)``。
平局按半胜半负计入（Elo 的标准处理）。

求解用 MM（minorization–maximization）迭代，单调收敛、无需学习率：

``π_a ← w_a / Σ_b [ n_ab / (π_a + π_b) ]``

其中 ``w_a`` 是 a 的总得分（胜=1、平=0.5），``n_ab`` 是 a 与 b 的对局数。

**必要的正则化**（诚实标注）：全胜或全负的选手在 MLE 下评分是 ±∞。这里给每个选手
加 ``prior_matches`` 场"对虚拟锚点（π=1）的半胜半负"，等价于把评分向池子中位数收缩。
默认 1 场；``prior_matches=0`` 时全胜/全负选手会被标记为 ``saturated`` 并剔除。

**锚定**：拟合只确定相对强弱（尺度已由 400 固定，位置不定）。若给出 ``anchors``
（A 的 manifest Elo），用**共同选手上的均值平移**把结果对齐到人类榜刻度，
不做尺度缩放（缩放会掩盖"人类榜与实测分布宽度不同"这个真实现象）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

ELO_SCALE = 400.0
_MAX_ITERATIONS = 5000
_TOLERANCE = 1e-10


@dataclass(frozen=True)
class PairwiseRecord:
    """一场（或若干场汇总的）对局：``player`` 对 ``opponent`` 拿到 ``points``。"""

    player: str
    opponent: str
    points: float
    matches: float = 1.0


@dataclass(frozen=True)
class RatingRow:
    player_id: str
    elo: float | None
    matches: float
    points: float
    winrate: float | None
    component: int
    saturated: bool
    anchor_elo: float | None
    note: str | None = None


def _components(neighbors: Mapping[str, set[str]]) -> dict[str, int]:
    seen: dict[str, int] = {}
    index = 0
    for start in sorted(neighbors):
        if start in seen:
            continue
        stack = [start]
        seen[start] = index
        while stack:
            node = stack.pop()
            for other in sorted(neighbors.get(node, ())):
                if other not in seen:
                    seen[other] = index
                    stack.append(other)
        index += 1
    return seen


def fit_ratings(
    records: Sequence[PairwiseRecord],
    *,
    anchors: Mapping[str, float] | None = None,
    prior_matches: float = 1.0,
) -> tuple[RatingRow, ...]:
    """对整张对局图做 BT 极大似然拟合，返回按 Elo 降序的评分表。"""

    points: dict[str, float] = {}
    played: dict[str, float] = {}
    pair_counts: dict[tuple[str, str], float] = {}
    neighbors: dict[str, set[str]] = {}
    for record in records:
        if record.player == record.opponent or record.matches <= 0:
            continue
        points[record.player] = points.get(record.player, 0.0) + record.points
        points[record.opponent] = (
            points.get(record.opponent, 0.0) + record.matches - record.points
        )
        played[record.player] = played.get(record.player, 0.0) + record.matches
        played[record.opponent] = played.get(record.opponent, 0.0) + record.matches
        key = (record.player, record.opponent) if record.player < record.opponent else (
            record.opponent,
            record.player,
        )
        pair_counts[key] = pair_counts.get(key, 0.0) + record.matches
        neighbors.setdefault(record.player, set()).add(record.opponent)
        neighbors.setdefault(record.opponent, set()).add(record.player)
    if not played:
        return ()

    component_of = _components(neighbors)
    sizes: dict[int, int] = {}
    for component in component_of.values():
        sizes[component] = sizes.get(component, 0) + 1
    main_component = max(sizes, key=lambda key: (sizes[key], -key))

    players = sorted(player for player in played if component_of[player] == main_component)
    if not players:
        return ()
    pi = dict.fromkeys(players, 1.0)
    opponents: dict[str, list[tuple[str, float]]] = {player: [] for player in players}
    for (left, right), count in pair_counts.items():
        if left not in pi or right not in pi:
            continue
        opponents[left].append((right, count))
        opponents[right].append((left, count))

    for _ in range(_MAX_ITERATIONS):
        shift = 0.0
        for player in players:
            numerator = points[player] + 0.5 * prior_matches
            denominator = prior_matches / (pi[player] + 1.0)
            for other, count in opponents[player]:
                denominator += count / (pi[player] + pi[other])
            if numerator <= 0.0 or denominator <= 0.0:
                continue
            updated = numerator / denominator
            shift = max(shift, abs(math.log(updated) - math.log(pi[player])))
            pi[player] = updated
        # 归一化到几何均值 1（BT 只确定相对值）。
        logs = sum(math.log(pi[player]) for player in players) / len(players)
        factor = math.exp(-logs)
        for player in players:
            pi[player] *= factor
        if shift < _TOLERANCE:
            break

    raw = {player: ELO_SCALE * math.log10(pi[player]) for player in players}
    offset = 0.0
    if anchors:
        shared = [player for player in players if player in anchors]
        if shared:
            offset = sum(anchors[player] - raw[player] for player in shared) / len(shared)

    rows: list[RatingRow] = []
    for player in sorted(played):
        in_main = component_of[player] == main_component
        total = played[player]
        scored = points[player]
        saturated = prior_matches <= 0 and (scored <= 0.0 or scored >= total)
        rows.append(
            RatingRow(
                player_id=player,
                elo=(round(raw[player] + offset, 2) if in_main and not saturated else None),
                matches=total,
                points=scored,
                winrate=(scored / total if total else None),
                component=component_of[player],
                saturated=saturated,
                anchor_elo=(anchors or {}).get(player),
                note=(
                    None
                    if in_main and not saturated
                    else ("saturated (all wins or all losses)" if saturated else "disconnected")
                ),
            )
        )
    rows.sort(key=lambda row: (row.elo is None, -(row.elo or 0.0), row.player_id))
    return tuple(rows)

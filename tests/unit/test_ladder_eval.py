"""全池实测评分的离线契约测试（赛程 + BT 拟合）。"""

from __future__ import annotations

import random

from agentbench_hl.application.ladder_eval import build_plan, pairing_offsets
from agentbench_hl.domain.rating import PairwiseRecord, fit_ratings


def test_pairing_is_connected_and_side_balanced() -> None:
    players = [f"p{index:02d}" for index in range(12)]
    plan = build_plan("g", players, ("P0", "P1"), degree=4, seeds=(7,))

    # 每对都出现两种座次（谁执 P0），避免先后手优势污染评分。
    pairs = {(player, opponent) for player, opponent, _, _ in plan.cases}
    assert all((opponent, player) in pairs for player, opponent in pairs)

    # d=1 的循环配对本身是一条哈密顿环 ⇒ 图连通（BT 可解的前提）。
    neighbours: dict[str, set[str]] = {player: set() for player in players}
    for player, opponent, _, _ in plan.cases:
        neighbours[player].add(opponent)
        neighbours[opponent].add(player)
    seen = {players[0]}
    stack = [players[0]]
    while stack:
        node = stack.pop()
        for other in neighbours[node]:
            if other not in seen:
                seen.add(other)
                stack.append(other)
    assert seen == set(players)


def test_plan_is_deterministic_and_scales_linearly() -> None:
    players = [f"p{index:02d}" for index in range(20)]
    first = build_plan("g", players, ("P0", "P1"), degree=6, seeds=(7, 11))
    second = build_plan("g", list(reversed(players)), ("P0", "P1"), degree=6, seeds=(7, 11))

    assert first.cases == second.cases  # 只取决于 id 集合，与输入顺序无关
    assert pairing_offsets(6) == (1, 2, 3)
    # n × ceil(degree/2) × 2 座次 × seeds
    assert first.total == 20 * 3 * 2 * 2


def test_bradley_terry_recovers_planted_ratings() -> None:
    truth = {f"p{index:02d}": 1000 + (index - 10) * 40 for index in range(20)}
    generator = random.Random(20260816)

    def probability(left: str, right: str) -> float:
        return 1 / (1 + 10 ** (-(truth[left] - truth[right]) / 400))

    plan = build_plan("g", sorted(truth), ("P0", "P1"), degree=12, seeds=(7, 11, 13))
    records = [
        PairwiseRecord(
            player,
            opponent,
            1.0 if generator.random() < probability(player, opponent) else 0.0,
        )
        for player, opponent, _, _ in plan.cases
    ]

    rows = fit_ratings(records, anchors={key: truth[key] for key in list(truth)[:5]})
    rated = [row for row in rows if row.elo is not None]

    assert len(rated) == len(truth)
    errors = [abs(row.elo - truth[row.player_id]) for row in rated]  # type: ignore[operator]
    # 采样噪声下的合理精度：平均误差应显著小于相邻档位差（40 分）的两倍。
    assert sum(errors) / len(errors) < 70
    # 排序应与真实强弱高度一致（最强者仍在前三）。
    assert rows[0].player_id in {"p19", "p18", "p17"}


def test_disconnected_and_saturated_players_are_flagged_not_faked() -> None:
    records = [
        PairwiseRecord("a", "b", 1.0),
        PairwiseRecord("a", "b", 1.0),
        # 与主分量完全不连通的一对：无法与其它人比较，必须标记而不是硬给分。
        PairwiseRecord("y", "z", 1.0),
        PairwiseRecord("y", "z", 0.0),
        PairwiseRecord("y", "z", 1.0),
        PairwiseRecord("y", "z", 0.0),
    ]

    rows = {row.player_id: row for row in fit_ratings(records, prior_matches=1.0)}

    assert rows["y"].elo is None and rows["y"].note == "disconnected"
    assert rows["a"].elo is not None
    # 全胜选手在有先验时评分有限；先验关掉时必须诚实标记饱和。
    saturated = {row.player_id: row for row in fit_ratings(records, prior_matches=0.0)}
    assert saturated["a"].elo is None
    assert saturated["a"].note == "saturated (all wins or all losses)"

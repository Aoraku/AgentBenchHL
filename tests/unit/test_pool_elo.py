"""``domain/pool_elo.py`` 的数值性质。

这层要保证的是"**换对手不会让曲线跳**"——实验 2 的有序课程每清掉一个人就换对手，
如果 Elo 估计跟着对手走，画出来就是"越强越掉分"的假象。
"""

from __future__ import annotations

import pytest

from agentbench_hl.domain.pool_elo import estimate_pool_elo

ANCHORS = {"strong": 2000.0, "mid": 1600.0, "weak": 1200.0, "unranked": None}


def _matches(opponent: str, wins: int, losses: int, draws: int = 0) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows += [{"opponent_id": opponent, "points": 1.0}] * wins
    rows += [{"opponent_id": opponent, "points": 0.0}] * losses
    rows += [{"opponent_id": opponent, "points": 0.5}] * draws
    return rows


def test_even_record_lands_on_the_anchor() -> None:
    """五成得分率 ⇒ 与对手同强度。这是整个刻度的定义点。"""

    estimate = estimate_pool_elo(_matches("mid", 5, 5), ANCHORS)
    assert estimate is not None
    assert estimate.elo == pytest.approx(1600.0, abs=1.0)
    assert estimate.anchored_matches == 10


def test_beating_a_stronger_opponent_scores_higher_than_beating_a_weaker_one() -> None:
    strong = estimate_pool_elo(_matches("strong", 7, 3), ANCHORS)
    weak = estimate_pool_elo(_matches("weak", 7, 3), ANCHORS)
    assert strong is not None and weak is not None
    assert strong.elo > weak.elo
    # 差距应当≈两个锚点之差（同样的得分率下，Elo 差是可加的）。
    assert (strong.elo - weak.elo) == pytest.approx(800.0, abs=20.0)


def test_switching_opponents_does_not_reset_the_estimate() -> None:
    """先打强的、再打弱的：估计应当落在两者之间，而不是跟着最新对手跳。

    对照：只看当轮的 ``elo_vs_opponent`` 在换到 weak 之后会直接掉到 1200 附近。
    """

    mixed = estimate_pool_elo(
        _matches("strong", 5, 5) + _matches("weak", 5, 5),
        ANCHORS,
    )
    assert mixed is not None
    assert 1200.0 < mixed.elo < 2000.0
    assert mixed.elo == pytest.approx(1600.0, abs=60.0)


def test_perfect_record_stays_finite() -> None:
    """全胜时似然没有有限最大值，必须靠正则收住，否则曲线会出现 ``inf``。"""

    estimate = estimate_pool_elo(_matches("mid", 8, 0), ANCHORS)
    assert estimate is not None
    assert 1600.0 < estimate.elo < 2600.0
    assert estimate.score_rate == 1.0


def test_all_losses_is_symmetric_to_all_wins() -> None:
    wins = estimate_pool_elo(_matches("mid", 8, 0), ANCHORS)
    losses = estimate_pool_elo(_matches("mid", 0, 8), ANCHORS)
    assert wins is not None and losses is not None
    assert (wins.elo - 1600.0) == pytest.approx(1600.0 - losses.elo, abs=1.0)


def test_more_evidence_moves_further_from_the_prior() -> None:
    """样本越多，正则的影响越小——8 连胜应当比 2 连胜给出更高的估计。"""

    few = estimate_pool_elo(_matches("mid", 2, 0), ANCHORS)
    many = estimate_pool_elo(_matches("mid", 8, 0), ANCHORS)
    assert few is not None and many is not None
    assert many.elo > few.elo


def test_opponents_without_anchor_are_skipped_not_guessed() -> None:
    """没有池内 Elo 的对手直接跳过。

    把"未知强度"当成平均水平混进来，会把估计悄悄拉向池子均值——
    宁可样本少，也不要引入一个看不见的偏差。
    """

    estimate = estimate_pool_elo(
        _matches("unranked", 9, 0) + _matches("mid", 1, 1),
        ANCHORS,
    )
    assert estimate is not None
    assert estimate.matches == 11
    assert estimate.anchored_matches == 2
    assert estimate.elo == pytest.approx(1600.0, abs=1.0)


def test_returns_none_without_any_usable_sample() -> None:
    assert estimate_pool_elo([], ANCHORS) is None
    assert estimate_pool_elo(_matches("unranked", 3, 0), ANCHORS) is None


def test_draws_count_as_half() -> None:
    draws = estimate_pool_elo(_matches("mid", 0, 0, draws=6), ANCHORS)
    assert draws is not None
    assert draws.elo == pytest.approx(1600.0, abs=1.0)
    assert draws.score_rate == pytest.approx(0.5)

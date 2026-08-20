"""线协议层面的策略比较 —— 域逻辑单测。

守两条纪律：
1. ε 正则 KL 的取值只依赖 (|A|, 是否换动作)，所以匿名填充支撑集是精确做法；
2. 候选提前崩溃造成的缺失决策**绝不能**被当成"与父版本一致"。
"""

from __future__ import annotations

import math

import pytest

from agentbench_hl.domain.metrics import epsilon_regularized_kl
from agentbench_hl.domain.wire_policy import (
    WireDecision,
    WireEpisode,
    compare_wire_policies,
    first_divergence,
    synthetic_support,
    wire_decision_samples,
)

EPSILON = 0.02


def _episode(*tokens: str, observations: tuple[str, ...] = ()) -> WireEpisode:
    ids = observations or tuple(f"obs{index}" for index in range(len(tokens)))
    return WireEpisode(
        match_id="m1",
        role="P0",
        decisions=tuple(
            WireDecision(index=index, observation_id=ids[index], action_token=token)
            for index, token in enumerate(tokens)
        ),
    )


def test_synthetic_support_has_declared_size_and_contains_both_actions() -> None:
    support = synthetic_support("a", "b", 6)

    assert len(support) == 6
    assert len(set(support)) == 6
    assert support[:2] == ("a", "b")


def test_synthetic_support_is_exact_not_an_approximation() -> None:
    # 换任何一组互不相同的填充符，KL 必须一模一样：这正是"只需要 |A|"的证明。
    left = epsilon_regularized_kl("a", "b", synthetic_support("a", "b", 5), EPSILON)
    right = epsilon_regularized_kl("a", "b", ("a", "b", "x", "y", "z"), EPSILON)

    assert left == pytest.approx(right)


def test_identical_action_needs_only_one_slot() -> None:
    support = synthetic_support("a", "a", 3)

    assert support[0] == "a"
    assert len(support) == 3
    assert epsilon_regularized_kl("a", "a", support, EPSILON) == 0.0


def test_support_smaller_than_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        synthetic_support("a", "b", 1)


def test_full_agreement_gives_zero_information_gain() -> None:
    reference = _episode("a", "b", "c")

    comparison = compare_wire_policies(
        reference, ("a", "b", "c"), support_size=6, epsilon=EPSILON
    )

    assert comparison.mean_kl_nats == 0.0
    assert comparison.disagreement_rate == 0.0
    assert first_divergence(reference, ("a", "b", "c")) is None


def test_partial_disagreement_scales_with_the_disagreement_rate() -> None:
    reference = _episode("a", "b", "c", "d")

    comparison = compare_wire_policies(
        reference, ("a", "x", "c", "y"), support_size=6, epsilon=EPSILON
    )
    unit = epsilon_regularized_kl("a", "b", synthetic_support("a", "b", 6), EPSILON)

    assert comparison.disagreement_rate == pytest.approx(0.5)
    assert comparison.mean_kl_nats == pytest.approx(0.5 * unit)
    assert first_divergence(reference, ("a", "x", "c", "y")) == 1


def test_larger_declared_support_gives_larger_kl() -> None:
    reference = _episode("a")

    small = compare_wire_policies(reference, ("b",), support_size=5, epsilon=EPSILON)
    large = compare_wire_policies(reference, ("b",), support_size=125, epsilon=EPSILON)

    assert small.mean_kl_nats is not None
    assert large.mean_kl_nats is not None
    assert large.mean_kl_nats > small.mean_kl_nats
    assert math.isfinite(large.mean_kl_nats)


def test_missing_candidate_decisions_are_truncated_not_counted_as_agreement() -> None:
    reference = _episode("a", "b", "c", "d")

    samples = wire_decision_samples(reference, ("a", "b"), support_size=6)
    comparison = compare_wire_policies(
        reference, ("a", "b"), support_size=6, epsilon=EPSILON
    )

    # 只比较了 2 个决策；后两个既不算一致也不算分歧。
    assert len(samples) == 2
    assert comparison.disagreement_rate == 0.0
    assert len(comparison.trace) == 2


def test_occupancy_shift_is_null_without_the_candidate_own_rollout() -> None:
    reference = _episode("a", "b")

    comparison = compare_wire_policies(
        reference, ("a", "x"), support_size=6, epsilon=EPSILON
    )

    # 重放发生在参考占据上，那里的"位移"恒为 0；诚实做法是记 null 而不是 0。
    assert comparison.occupancy_shift is None


def test_occupancy_shift_uses_each_version_own_states() -> None:
    reference = _episode("a", "b", observations=("s0", "s1"))

    comparison = compare_wire_policies(
        reference,
        ("a", "x"),
        support_size=6,
        epsilon=EPSILON,
        candidate_observation_ids=("s0", "s2"),
    )

    assert comparison.occupancy_shift == pytest.approx(0.5)


def test_episode_requires_identity() -> None:
    with pytest.raises(ValueError, match="role"):
        WireEpisode(match_id="m", role="", decisions=())

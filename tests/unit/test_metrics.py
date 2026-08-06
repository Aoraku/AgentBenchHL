from __future__ import annotations

import math

import pytest

from agentbench_hl.domain.metrics import (
    EloResult,
    epsilon_regularized_kl,
    fit_anchored_elo,
)


def test_deterministic_action_change_has_finite_kl() -> None:
    value = epsilon_regularized_kl(
        "HOLD",
        "BUILD:1,2",
        ("HOLD", "BUILD:1,2"),
        0.01,
    )

    assert value > 0
    assert math.isfinite(value)


def test_same_action_has_zero_kl() -> None:
    assert epsilon_regularized_kl(
        "HOLD",
        "HOLD",
        ("HOLD", "BUILD:1,2"),
        0.01,
    ) == pytest.approx(0.0)


def test_epsilon_kl_rejects_an_action_outside_hand_checked_legal_space() -> None:
    with pytest.raises(ValueError, match="legal"):
        epsilon_regularized_kl(
            "HOLD",
            "INVENTED_TACTIC",
            ("HOLD", "BUILD:1,2"),
            0.01,
        )


def results_for(version_id: str) -> tuple[EloResult, ...]:
    return (
        EloResult(version_id, "rank20", "P0", 1.0),
        EloResult(version_id, "rank20", "P1", 0.0),
    )


def test_candidate_elo_is_version_local() -> None:
    first = fit_anchored_elo(results_for("v001"), {"rank20": 1200.0})
    second = fit_anchored_elo(results_for("v002"), {"rank20": 1200.0})

    assert first == second
    assert math.isfinite(first.combined)

from __future__ import annotations

from agentbench_hl.adapters.antwar2.policy_probe import (
    PolicyDecision,
    PolicyEpisodeTrace,
    compare_policy_episode,
    occupancy_total_variation,
)


def test_policy_trace_compares_only_shared_atomic_decision_contexts() -> None:
    first_support = ("HOLD", "11:8:11", "31", "32")
    second_support = ("HOLD", "31", "32")
    parent = PolicyEpisodeTrace(
        match_id="episode-1",
        role="P0",
        decisions=(
            PolicyDecision(
                "episode-1:r0000:p0",
                ("11:8:11", "31"),
                (first_support, second_support, ("HOLD",)),
                "public-state-a",
            ),
        ),
    )
    candidate = PolicyEpisodeTrace(
        match_id="episode-1",
        role="P0",
        decisions=(
            PolicyDecision(
                "episode-1:r0000:p0",
                ("11:8:11", "32"),
                (first_support, second_support, ("HOLD",)),
                "public-state-a",
            ),
        ),
    )

    samples = compare_policy_episode(parent, candidate)

    assert tuple(item.state_id for item in samples) == (
        "episode-1:r0000:p0:a000",
        "episode-1:r0000:p0:a001",
    )
    assert samples[0].parent_action == samples[0].candidate_action == "11:8:11"
    assert samples[1].parent_action == "31"
    assert samples[1].candidate_action == "32"
    assert samples[1].legal_actions == second_support


def test_policy_trace_uses_hold_to_measure_sequence_termination() -> None:
    support = ("HOLD", "31")
    parent = PolicyEpisodeTrace(
        "episode-2",
        "P1",
        (PolicyDecision("state", (), (support,), "occupancy"),),
    )
    candidate = PolicyEpisodeTrace(
        "episode-2",
        "P1",
        (PolicyDecision("state", ("31",), (support, ("HOLD",)), "occupancy"),),
    )

    samples = compare_policy_episode(parent, candidate)

    assert len(samples) == 1
    assert samples[0].parent_action == "HOLD"
    assert samples[0].candidate_action == "31"


def test_occupancy_shift_compares_each_version_own_normalized_state_histogram() -> None:
    value = occupancy_total_variation(
        ("state-a", "state-a"),
        ("state-a", "state-b"),
    )

    assert value == 0.5

"""Deterministic-policy information and fixed-pool performance metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from agentbench_hl.domain.models import Usage


def epsilon_regularized_kl(
    old_action: str,
    new_action: str,
    legal_actions: tuple[str, ...],
    epsilon: float,
) -> float:
    """Measure deterministic action change through a declared epsilon channel."""

    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be strictly between zero and one")
    if not legal_actions or len(set(legal_actions)) != len(legal_actions):
        raise ValueError("legal actions must be non-empty and unique")
    if old_action not in legal_actions or new_action not in legal_actions:
        raise ValueError("both deterministic actions must be in the legal decision space")
    if old_action == new_action:
        return 0.0
    support_size = len(legal_actions)
    background = epsilon / support_size
    selected = 1.0 - epsilon + background
    old_distribution = {
        action: selected if action == old_action else background for action in legal_actions
    }
    new_distribution = {
        action: selected if action == new_action else background for action in legal_actions
    }
    return sum(
        probability * math.log(probability / new_distribution[action])
        for action, probability in old_distribution.items()
    )


@dataclass(frozen=True)
class EloResult:
    candidate_id: str
    opponent_id: str
    role: str
    points: float

    def __post_init__(self) -> None:
        if self.role not in {"P0", "P1"}:
            raise ValueError("Elo role must be P0 or P1")
        if not 0 <= self.points <= 1:
            raise ValueError("Elo points must be in [0, 1]")


@dataclass(frozen=True)
class EloEstimate:
    p0: float
    p1: float
    combined: float


def _fit_rating(
    results: tuple[EloResult, ...],
    human_ratings: dict[str, float],
) -> float:
    anchor = sum(human_ratings.values()) / len(human_ratings)
    observations: list[tuple[float, float, float]] = []
    for result in results:
        try:
            opponent_rating = float(human_ratings[result.opponent_id])
        except KeyError as exc:
            raise ValueError(f"missing frozen human rating: {result.opponent_id}") from exc
        observations.append((opponent_rating, result.points, 1.0))
    observations.extend(((anchor, 1.0, 0.5), (anchor, 0.0, 0.5)))

    def gradient(candidate_rating: float) -> float:
        total = 0.0
        for opponent_rating, points, weight in observations:
            expected = 1.0 / (1.0 + 10 ** ((opponent_rating - candidate_rating) / 400.0))
            total += weight * (points - expected)
        return total

    low = anchor - 4000.0
    high = anchor + 4000.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if gradient(midpoint) > 0:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def fit_anchored_elo(
    results: tuple[EloResult, ...],
    human_ratings: dict[str, float],
) -> EloEstimate:
    if not human_ratings:
        raise ValueError("human ratings cannot be empty")
    candidate_ids = {result.candidate_id for result in results}
    if len(candidate_ids) > 1:
        raise ValueError("candidate Elo must be fitted independently per version")
    return EloEstimate(
        p0=_fit_rating(tuple(item for item in results if item.role == "P0"), human_ratings),
        p1=_fit_rating(tuple(item for item in results if item.role == "P1"), human_ratings),
        combined=_fit_rating(results, human_ratings),
    )


def combine_usage(*values: Usage) -> Usage:
    def total(field: str) -> int | None:
        items = [getattr(value, field) for value in values]
        return None if all(item is None for item in items) else sum(item or 0 for item in items)

    wall_items = [value.wall_time_s for value in values]
    wall = (
        None if all(item is None for item in wall_items) else sum(item or 0 for item in wall_items)
    )
    return Usage(
        input_tokens=total("input_tokens"),
        cached_input_tokens=total("cached_input_tokens"),
        output_tokens=total("output_tokens"),
        reasoning_tokens=total("reasoning_tokens"),
        total_tokens=total("total_tokens"),
        wall_time_s=wall,
    )


@dataclass(frozen=True)
class IterationMetrics:
    research_iteration: int
    candidate_id: str
    champion_id: str
    behavioral_ig: float | None
    occupancy_shift: float | None
    action_disagreement: float | None
    elo: EloEstimate | None
    win_rate: float | None
    score_margin: float | None
    learning_usage: Usage = field(default_factory=Usage)
    evaluation_usage: Usage = field(default_factory=Usage)
    total_usage: Usage = field(default_factory=Usage)

    def to_row(self) -> dict[str, object]:
        return {
            "research_iteration": self.research_iteration,
            "candidate_id": self.candidate_id,
            "champion_id": self.champion_id,
            "behavioral_ig": self.behavioral_ig,
            "occupancy_shift": self.occupancy_shift,
            "action_disagreement": self.action_disagreement,
            "fixed_pool_elo": None if self.elo is None else self.elo.combined,
            "elo_p0": None if self.elo is None else self.elo.p0,
            "elo_p1": None if self.elo is None else self.elo.p1,
            "win_rate": self.win_rate,
            "score_margin": self.score_margin,
            "learning_tokens": self.learning_usage.total_tokens,
            "evaluation_tokens": self.evaluation_usage.total_tokens,
            "total_tokens": self.total_usage.total_tokens,
            "learning_wall_time_s": self.learning_usage.wall_time_s,
            "evaluation_wall_time_s": self.evaluation_usage.wall_time_s,
            "total_wall_time_s": self.total_usage.wall_time_s,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> IterationMetrics:
        elo = (
            None
            if row.get("fixed_pool_elo") is None
            else EloEstimate(
                p0=float(row["elo_p0"]),
                p1=float(row["elo_p1"]),
                combined=float(row["fixed_pool_elo"]),
            )
        )

        def usage(prefix: str) -> Usage:
            tokens = row.get(f"{prefix}_tokens")
            wall_time = row.get(f"{prefix}_wall_time_s")
            return Usage(
                total_tokens=None if tokens is None else int(tokens),
                wall_time_s=None if wall_time is None else float(wall_time),
            )

        return cls(
            research_iteration=int(row["research_iteration"]),
            candidate_id=str(row["candidate_id"]),
            champion_id=str(row["champion_id"]),
            behavioral_ig=(
                None if row.get("behavioral_ig") is None else float(row["behavioral_ig"])
            ),
            occupancy_shift=(
                None if row.get("occupancy_shift") is None else float(row["occupancy_shift"])
            ),
            action_disagreement=(
                None
                if row.get("action_disagreement") is None
                else float(row["action_disagreement"])
            ),
            elo=elo,
            win_rate=(None if row.get("win_rate") is None else float(row["win_rate"])),
            score_margin=(None if row.get("score_margin") is None else float(row["score_margin"])),
            learning_usage=usage("learning"),
            evaluation_usage=usage("evaluation"),
            total_usage=usage("total"),
        )

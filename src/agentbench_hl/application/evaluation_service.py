"""Hash-cached evaluation, regression gates, and human calibration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.ports.arena import Arena, MatchCase, MatchResult
from agentbench_hl.ports.event_store import EventStore


@dataclass(frozen=True)
class EvaluationObservation:
    case_id: str
    opponent_id: str
    role: str
    seed: int
    status: str
    result: str | None
    match_id: str | None = None
    points: float | None = None
    score_margin: float | None = None
    replay_path: Path | None = None


@dataclass(frozen=True)
class EvaluationResult:
    observations: tuple[EvaluationObservation, ...]
    target_solved: bool
    regressions_passed: bool
    promotable: bool
    frontier_eligible: bool
    retry_case_ids: tuple[str, ...]


def evaluate_candidate(
    *,
    observations: tuple[EvaluationObservation, ...],
    target_id: str,
    locked_regression_ids: tuple[str, ...],
    roles: tuple[str, ...],
    seeds: tuple[int, ...],
) -> EvaluationResult:
    by_case = {item.case_id: item for item in observations}

    def case_id(opponent: str, role: str, seed: int) -> str:
        return f"{opponent}:{role}:{seed}"

    target_cases = tuple(case_id(target_id, role, seed) for role in roles for seed in seeds)
    regression_cases = tuple(
        case_id(opponent, role, seed)
        for opponent in locked_regression_ids
        for role in roles
        for seed in seeds
    )
    expected = (*target_cases, *regression_cases)
    retry = tuple(
        item for item in expected if item not in by_case or by_case[item].status != "complete"
    )
    target_solved = all(
        item in by_case and by_case[item].status == "complete" and by_case[item].result == "win"
        for item in target_cases
    )
    regressions_passed = all(
        item in by_case and by_case[item].status == "complete" and by_case[item].result == "win"
        for item in regression_cases
    )
    complete = not retry
    return EvaluationResult(
        observations=observations,
        target_solved=target_solved,
        regressions_passed=regressions_passed,
        promotable=complete and target_solved and regressions_passed,
        frontier_eligible=complete,
        retry_case_ids=retry,
    )


@dataclass(frozen=True)
class HumanCalibrationMatch:
    player0: str
    player1: str
    points0: float

    def __post_init__(self) -> None:
        if not 0 <= self.points0 <= 1:
            raise ValueError("human calibration points must be in [0, 1]")
        if self.player0 == self.player1:
            raise ValueError("human calibration requires distinct players")


@dataclass(frozen=True)
class HumanCalibration:
    matrix_hash: str
    ratings: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ratings", MappingProxyType(dict(self.ratings)))


def fit_human_calibration(
    matches: tuple[HumanCalibrationMatch, ...],
) -> HumanCalibration:
    if not matches:
        raise ValueError("human calibration matrix cannot be empty")
    canonical = sorted((item.player0, item.player1, float(item.points0)) for item in matches)
    encoded = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    matrix_hash = hashlib.sha256(encoded).hexdigest()
    players = sorted({name for row in canonical for name in row[:2]})
    ratings = {player: 1000.0 for player in players}
    for _ in range(2000):
        gradients = {player: -0.0005 * (ratings[player] - 1000.0) for player in players}
        for player0, player1, points0 in canonical:
            expected0 = 1.0 / (1.0 + 10 ** ((ratings[player1] - ratings[player0]) / 400.0))
            error = points0 - expected0
            gradients[player0] += error
            gradients[player1] -= error
        updated = {player: ratings[player] + 0.25 * gradients[player] for player in players}
        mean = sum(updated.values()) / len(updated)
        ratings = {player: value - mean + 1000.0 for player, value in updated.items()}
    return HumanCalibration(
        matrix_hash,
        {player: round(ratings[player], 6) for player in players},
    )


@dataclass(frozen=True)
class EvaluationMatrix:
    opponent_ids: tuple[str, ...]
    roles: tuple[str, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class _EvaluationCasePlan:
    case_id: str
    opponent_id: str
    role: str
    seed: int
    cache_key: str
    cached: Mapping[str, object] | None


class EvaluationService:
    """Run or reuse only complete cases with an exact five-part content key."""

    def __init__(
        self,
        *,
        arena: Arena,
        event_store: EventStore,
        backend_hash: str,
        candidate_root: Callable[[str], Path],
        candidate_hash: Callable[[str], str],
        opponent_hashes: Mapping[str, str],
        # Native AntWar2 transport is stateful at the process/pipe level.
        # Serialize cases by default so a hung child can be timed out and
        # reaped deterministically; callers may still opt into parallelism.
        max_workers: int = 1,
    ) -> None:
        if max_workers < 1:
            raise ValueError("evaluation max_workers must be positive")
        self.arena = arena
        self.event_store = event_store
        self.backend_hash = backend_hash
        self.candidate_root = candidate_root
        self.candidate_hash = candidate_hash
        self.opponent_hashes = dict(opponent_hashes)
        self.max_workers = max_workers

    def evaluate_version(
        self,
        version_id: str,
        matrix: EvaluationMatrix,
        *,
        target_id: str,
        locked_regression_ids: tuple[str, ...],
    ) -> EvaluationResult:
        existing = self.event_store.read_all()
        completed = {
            str(event.payload["cache_key"]): event.payload
            for event in existing
            if event.event_type == "EvaluationCaseCompleted"
        }
        candidate_hash = self.candidate_hash(version_id)
        plans: list[_EvaluationCasePlan] = []
        for opponent_id in matrix.opponent_ids:
            for role in matrix.roles:
                for seed in matrix.seeds:
                    case_id = f"{opponent_id}:{role}:{seed}"
                    key_value = (
                        self.backend_hash,
                        candidate_hash,
                        self.opponent_hashes[opponent_id],
                        role,
                        seed,
                    )
                    cache_key = hashlib.sha256(
                        json.dumps(key_value, separators=(",", ":")).encode("utf-8")
                    ).hexdigest()
                    plans.append(
                        _EvaluationCasePlan(
                            case_id,
                            opponent_id,
                            role,
                            seed,
                            cache_key,
                            completed.get(cache_key),
                        )
                    )

        candidate_root = self.candidate_root(version_id)
        pending = tuple(plan for plan in plans if plan.cached is None)
        fresh: dict[str, MatchResult] = {}
        if pending:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(pending)),
                thread_name_prefix="agentbench-eval",
            ) as executor:
                futures = {
                    plan.case_id: executor.submit(
                        self.arena.run_case,
                        MatchCase(
                            version_id,
                            plan.opponent_id,
                            plan.role,
                            plan.seed,
                        ),
                        candidate_root,
                    )
                    for plan in pending
                }
                fresh = {plan.case_id: futures[plan.case_id].result() for plan in pending}

        observations: list[EvaluationObservation] = []
        for plan in plans:
            cached = plan.cached
            if cached is not None:
                observations.append(
                    EvaluationObservation(
                        plan.case_id,
                        plan.opponent_id,
                        plan.role,
                        plan.seed,
                        "complete",
                        str(cached["result"]),
                        str(cached["match_id"]),
                        float(cached["points"]),
                        float(cached["score_margin"]),
                        Path(str(cached["replay_path"])).resolve(),
                    )
                )
                continue
            result = fresh[plan.case_id]
            observation = EvaluationObservation(
                plan.case_id,
                plan.opponent_id,
                plan.role,
                plan.seed,
                result.status,
                result.result,
                f"{version_id}-{plan.opponent_id}-{plan.role.lower()}-s{plan.seed}",
                result.points,
                result.score_margin,
                result.replay_path,
            )
            observations.append(observation)
            if result.status == "complete":
                event = FinalizedEvent.create(
                    "EvaluationCaseCompleted",
                    {
                        "cache_key": plan.cache_key,
                        "backend_hash": self.backend_hash,
                        "candidate_hash": candidate_hash,
                        "opponent_hash": self.opponent_hashes[plan.opponent_id],
                        "case_id": plan.case_id,
                        "version_id": version_id,
                        "opponent_id": plan.opponent_id,
                        "role": plan.role,
                        "seed": plan.seed,
                        "match_id": observation.match_id,
                        "result": result.result,
                        "points": result.points,
                        "score_margin": result.score_margin,
                        "terminal_base_hp": result.terminal_base_hp,
                        "rounds": result.rounds,
                        "replay_path": str(result.replay_path),
                        "trace_path": (
                            None if result.trace_path is None else str(result.trace_path)
                        ),
                        "events_path": (
                            None if result.events_path is None else str(result.events_path)
                        ),
                        "process_returncodes": result.process_returncodes,
                    },
                    idempotency_key=f"evaluation-complete:{plan.cache_key}",
                )
                self.event_store.append(event)
            else:
                attempt = sum(
                    event.event_type == "EvaluationCaseIncomplete"
                    and event.payload.get("cache_key") == plan.cache_key
                    for event in existing
                )
                event = FinalizedEvent.create(
                    "EvaluationCaseIncomplete",
                    {
                        "cache_key": plan.cache_key,
                        "case_id": plan.case_id,
                        "version_id": version_id,
                        "opponent_id": plan.opponent_id,
                        "role": plan.role,
                        "seed": plan.seed,
                        "status": result.status,
                        "result": result.result,
                        "error": result.error,
                    },
                    idempotency_key=(f"evaluation-incomplete:{plan.cache_key}:{attempt}"),
                )
                self.event_store.append(event)
        return evaluate_candidate(
            observations=tuple(observations),
            target_id=target_id,
            locked_regression_ids=locked_regression_ids,
            roles=matrix.roles,
            seeds=matrix.seeds,
        )

    @staticmethod
    def calibrate_human_pool(
        matrix: tuple[HumanCalibrationMatch, ...],
    ) -> HumanCalibration:
        return fit_human_calibration(matrix)

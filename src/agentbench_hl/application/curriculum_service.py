"""Bottom-up official-rank curriculum with locked solved opponents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from agentbench_hl.adapters.antwar2.runtime import Opponent


@dataclass(frozen=True)
class CurriculumMatch:
    opponent_id: str
    role: str
    seed: int
    status: str
    result: str | None


@dataclass(frozen=True)
class OpponentProgress:
    opponent_id: str
    state: str
    completed_wins: int
    required_cases: int


@dataclass(frozen=True)
class CurriculumStatus:
    by_opponent: Mapping[str, OpponentProgress]
    locked_regression_ids: tuple[str, ...]
    all_runnable_solved: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_opponent", MappingProxyType(dict(self.by_opponent)))


class CurriculumComplete(RuntimeError):
    pass


class CurriculumService:
    def __init__(
        self,
        *,
        opponents: tuple[Opponent, ...],
        roles: tuple[str, ...],
        seeds: tuple[int, ...],
        matches: tuple[CurriculumMatch, ...],
        required_win_rate: float = 0.5,
    ) -> None:
        if not roles or any(role not in {"P0", "P1"} for role in roles):
            raise ValueError("curriculum roles must be P0/P1")
        if not seeds:
            raise ValueError("curriculum seeds cannot be empty")
        if not 0.0 <= required_win_rate <= 1.0:
            raise ValueError("curriculum required_win_rate must be in [0, 1]")
        self.opponents = tuple(sorted(opponents, key=lambda item: item.rank))
        self.roles = roles
        self.seeds = seeds
        self.matches = matches
        self.required_win_rate = required_win_rate

    def status(self) -> CurriculumStatus:
        progress: dict[str, OpponentProgress] = {}
        required = len(self.roles) * len(self.seeds)
        for opponent in self.opponents:
            if not opponent.runnable:
                progress[opponent.opponent_id] = OpponentProgress(
                    opponent.opponent_id,
                    "unrunnable",
                    0,
                    required,
                )
                continue
            expected = {(role, seed) for role in self.roles for seed in self.seeds}
            observed = {
                (item.role, item.seed): item
                for item in self.matches
                if item.opponent_id == opponent.opponent_id and (item.role, item.seed) in expected
            }
            wins = sum(
                item.status == "complete" and item.result == "win" for item in observed.values()
            )
            score = wins + 0.5 * sum(
                item.status == "complete" and item.result == "draw" for item in observed.values()
            )
            pass_rate = score / required if required else 0.0
            if any(item.status != "complete" for item in observed.values()):
                state = "incomplete"
            elif set(observed) == expected and pass_rate >= self.required_win_rate:
                state = "solved"
            else:
                state = "unsolved"
            progress[opponent.opponent_id] = OpponentProgress(
                opponent.opponent_id,
                state,
                wins,
                required,
            )
        locked = tuple(
            opponent.opponent_id
            for opponent in sorted(self.opponents, key=lambda item: item.rank, reverse=True)
            if progress[opponent.opponent_id].state == "solved"
        )
        runnable = [item for item in self.opponents if item.runnable]
        return CurriculumStatus(
            by_opponent=progress,
            locked_regression_ids=locked,
            all_runnable_solved=all(
                progress[item.opponent_id].state == "solved" for item in runnable
            ),
        )

    def default_target(self) -> Opponent:
        status = self.status()
        candidates = [
            opponent
            for opponent in self.opponents
            if opponent.runnable and status.by_opponent[opponent.opponent_id].state != "solved"
        ]
        if not candidates:
            raise CurriculumComplete("all runnable human opponents are solved")
        return max(candidates, key=lambda item: item.rank)

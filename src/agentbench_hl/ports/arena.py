"""Deterministic game-arena contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

Role = str
MatchStatus = Literal["complete", "incomplete"]
Outcome = Literal["win", "loss", "draw"]


@dataclass(frozen=True)
class ProcessSpec:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.argv or not self.argv[0]:
            raise ValueError("process argv cannot be empty")
        object.__setattr__(self, "argv", tuple(str(item) for item in self.argv))
        object.__setattr__(self, "cwd", Path(self.cwd).resolve())


@dataclass(frozen=True)
class MatchCase:
    candidate_id: str
    opponent_id: str
    role: Role
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("match role must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("match seed must be an integer")


@dataclass(frozen=True)
class MatchResult:
    """Game-agnostic outcome of one deterministic match.

    ``score_margin`` and ``rounds`` are cross-game scalars in the shared metrics
    schema.  Any additional, game-specific terminal state (for example AntWar2's
    ``terminal_base_hp``) is carried opaquely in ``payload`` so the framework
    core never has to know a single game's terminal semantics.
    """

    case: MatchCase
    status: MatchStatus
    result: Outcome | None
    points: float | None
    score_margin: float | None
    rounds: int | None
    payload: Mapping[str, Any] = field(default_factory=dict)
    replay_path: Path | None = None
    trace_path: Path | None = None
    events_path: Path | None = None
    error: str | None = None
    process_returncodes: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.status == "complete":
            if self.result is None or self.points is None:
                raise ValueError("complete match requires an outcome")
            if self.score_margin is None:
                raise ValueError("complete match requires a score margin")
            if self.rounds is None or self.error is not None:
                raise ValueError("complete match has invalid completion fields")
        elif self.status == "incomplete":
            if self.result is not None or self.points is not None:
                raise ValueError("incomplete match cannot be scored")
            if not self.error:
                raise ValueError("incomplete match requires an error")
        else:
            raise ValueError(f"unknown match status: {self.status}")


class Arena(Protocol):
    def run_case(self, case: MatchCase, candidate_root: Path) -> MatchResult: ...

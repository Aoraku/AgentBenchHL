"""Result value objects and the iteration driver for :mod:`run_service`.

These are the immutable outputs of a run's lifecycle steps plus the small
``advance_run`` driver.  They are split out of ``run_service`` to keep the
service module focused on orchestration behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from agentbench_hl.application.evaluation_service import EvaluationResult
from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.lineage import LineageState
from agentbench_hl.domain.metrics import IterationMetrics

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from agentbench_hl.application.run_service import RunService


@dataclass(frozen=True)
class RunResult:
    root: Path
    lineage: LineageState
    match_id: str
    metrics: IterationMetrics
    events: tuple[FinalizedEvent, ...]

    def event_count(self, event_type: str) -> int:
        return sum(item.event_type == event_type for item in self.events)


@dataclass(frozen=True)
class IterationAdvanceResult:
    version_id: str
    parent_id: str
    target_id: str
    selection: str
    evaluation: EvaluationResult
    metrics: IterationMetrics


@dataclass(frozen=True)
class CertificationResult:
    champion_id: str
    passed: bool
    total_cases: int
    wins: int
    incomplete_cases: tuple[str, ...]
    failed_cases: tuple[str, ...]


def advance_run(
    run: RunService,
    *,
    acts: int,
) -> tuple[IterationAdvanceResult, ...]:
    if isinstance(acts, bool) or acts < 1:
        raise ValueError("acts must be a positive integer")
    return tuple(run.advance_one_iteration() for _ in range(acts))

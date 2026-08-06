"""Persistent, isolated research-agent runtime contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class RunContext:
    objective: str
    initial_prompt: str
    base_instructions: str
    developer_instructions: str
    cwd: Path
    candidate_root: Path
    gamepack_root: Path
    research_root: Path
    human_pool_root: Path
    evaluator_root: Path
    runtime_workspace_roots: tuple[Path, ...]
    model: str
    model_provider: str
    writable_workspace_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "cwd",
            "candidate_root",
            "gamepack_root",
            "research_root",
            "human_pool_root",
            "evaluator_root",
        ):
            object.__setattr__(self, field, Path(getattr(self, field)).resolve())
        object.__setattr__(
            self,
            "runtime_workspace_roots",
            tuple(Path(item).resolve() for item in self.runtime_workspace_roots),
        )
        writable = self.writable_workspace_roots or (
            self.candidate_root,
            self.research_root,
        )
        object.__setattr__(
            self,
            "writable_workspace_roots",
            tuple(Path(item).resolve() for item in writable),
        )

    def with_runtime_workspace_roots(self, roots: tuple[Path, ...]) -> RunContext:
        return replace(self, runtime_workspace_roots=roots)

    def validate_isolation(self) -> None:
        required = {self.candidate_root, self.gamepack_root, self.research_root}
        roots = set(self.runtime_workspace_roots)
        missing = required - roots
        if missing:
            raise ValueError(f"runtime workspace roots omit required paths: {sorted(missing)}")
        forbidden = {
            self.human_pool_root: "human pool",
            self.evaluator_root: "evaluator certification",
        }
        for root in roots:
            if not root.is_absolute() or not root.exists():
                raise ValueError(f"runtime workspace root is unavailable: {root}")
            lowered = str(root).lower()
            if "handoff_next_agent" in lowered or "reference_policy" in lowered:
                raise ValueError(f"runtime root exposes reference policy material: {root}")
            for unsafe, label in forbidden.items():
                if root == unsafe or unsafe.is_relative_to(root) or root.is_relative_to(unsafe):
                    raise ValueError(f"runtime root exposes {label}: {root}")
        if self.cwd not in roots:
            raise ValueError("agent cwd must be an explicit isolated workspace root")
        writable = set(self.writable_workspace_roots)
        if not {self.candidate_root, self.research_root}.issubset(writable):
            raise ValueError("candidate and research roots must be writable")
        if not writable.issubset(roots):
            raise ValueError("writable roots must be declared runtime roots")
        if self.gamepack_root in writable:
            raise ValueError("frozen GamePack cannot be writable")


@dataclass
class AgentSession:
    thread_id: str
    goal_status: str
    ephemeral: bool
    active_turn_id: str | None = None


CheckpointPredicate = Callable[[object], bool]


class AgentRuntime(Protocol):
    def start(self, run_context: RunContext) -> AgentSession: ...

    def resume(self, session_id: str, run_context: RunContext) -> AgentSession: ...

    def run_until_checkpoint(
        self,
        session: AgentSession,
        run_context: RunContext,
        checkpoint_predicate: CheckpointPredicate,
    ) -> AgentSession: ...

    def pause(self, session: AgentSession) -> AgentSession: ...

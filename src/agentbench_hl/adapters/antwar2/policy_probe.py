"""Compare deterministic atomic decisions on the same frozen public states."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.domain.metrics import epsilon_regularized_kl


def action_key(operation: object | None) -> str:
    """Canonicalize one public SDK atomic operation without tactical labels."""

    if operation is None:
        return "HOLD"
    tokens = operation.to_protocol_tokens()
    return ":".join(str(int(token)) for token in tokens)


def enumerate_legal_atomic_actions(
    state: object,
    player: int,
    pending: tuple[object, ...] = (),
) -> tuple[str, ...]:
    """Enumerate exact legal atoms using the frozen SDK legality predicate."""

    from SDK.backend.model import Operation
    from SDK.utils.constants import TOWER_UPGRADE_TREE, VALID_CELLS, OperationType

    candidates = [Operation(OperationType.BUILD_TOWER, x, y) for x, y in VALID_CELLS]
    for tower in state.towers_of(player):
        candidates.extend(
            Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
            for target in TOWER_UPGRADE_TREE.get(tower.tower_type, ())
        )
        candidates.append(Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id))
    for operation_type in (
        OperationType.USE_LIGHTNING_STORM,
        OperationType.USE_EMP_BLASTER,
        OperationType.USE_DEFLECTOR,
        OperationType.USE_EMERGENCY_EVASION,
    ):
        candidates.extend(Operation(operation_type, x, y) for x, y in VALID_CELLS)
    candidates.extend(
        (
            Operation(OperationType.UPGRADE_GENERATION_SPEED),
            Operation(OperationType.UPGRADE_GENERATED_ANT),
        )
    )
    legal = {"HOLD"}
    legal.update(
        action_key(item) for item in candidates if state.can_apply_operation(player, item, pending)
    )
    return tuple(sorted(legal))


@dataclass(frozen=True)
class DecisionSample:
    state_id: str
    legal_actions: tuple[str, ...]
    parent_action: str
    candidate_action: str
    parent_occupancy: str
    candidate_occupancy: str


@dataclass(frozen=True)
class PolicyDecision:
    state_id: str
    actions: tuple[str, ...]
    legal_supports: tuple[tuple[str, ...], ...]
    occupancy_id: str

    def __post_init__(self) -> None:
        if not self.state_id or not self.occupancy_id:
            raise ValueError("policy decision requires state and occupancy IDs")
        if len(self.legal_supports) != len(self.actions) + 1:
            raise ValueError("policy decision requires one support per atom plus HOLD")
        for index, support in enumerate(self.legal_supports):
            if not support or len(set(support)) != len(support) or "HOLD" not in support:
                raise ValueError("policy decision support must be unique and include HOLD")
            if index < len(self.actions) and self.actions[index] not in support:
                raise ValueError("policy action is outside its legal atomic support")


@dataclass(frozen=True)
class PolicyEpisodeTrace:
    match_id: str
    role: str
    decisions: tuple[PolicyDecision, ...]

    def __post_init__(self) -> None:
        if not self.match_id:
            raise ValueError("policy episode requires a match ID")
        if self.role not in {"P0", "P1"}:
            raise ValueError("policy episode role must be P0 or P1")


def compare_policy_episode(
    parent: PolicyEpisodeTrace,
    candidate: PolicyEpisodeTrace,
) -> tuple[DecisionSample, ...]:
    """Compare atoms only while both policies share the same decision prefix."""

    if parent.match_id != candidate.match_id or parent.role != candidate.role:
        raise ValueError("policy traces must use the same episode and role")
    parent_ids = tuple(item.state_id for item in parent.decisions)
    candidate_ids = tuple(item.state_id for item in candidate.decisions)
    if parent_ids != candidate_ids:
        raise ValueError("policy traces must use the same ordered reference states")
    samples: list[DecisionSample] = []
    for parent_decision, candidate_decision in zip(
        parent.decisions, candidate.decisions, strict=True
    ):
        atom_index = 0
        while True:
            parent_action = (
                parent_decision.actions[atom_index]
                if atom_index < len(parent_decision.actions)
                else "HOLD"
            )
            candidate_action = (
                candidate_decision.actions[atom_index]
                if atom_index < len(candidate_decision.actions)
                else "HOLD"
            )
            parent_support = parent_decision.legal_supports[atom_index]
            candidate_support = candidate_decision.legal_supports[atom_index]
            if parent_support != candidate_support:
                raise ValueError("shared atomic prefix produced different legal supports")
            if candidate_action not in parent_support:
                raise ValueError("candidate action is outside the shared legal support")
            samples.append(
                DecisionSample(
                    state_id=f"{parent_decision.state_id}:a{atom_index:03d}",
                    legal_actions=parent_support,
                    parent_action=parent_action,
                    candidate_action=candidate_action,
                    parent_occupancy=parent_decision.occupancy_id,
                    candidate_occupancy=candidate_decision.occupancy_id,
                )
            )
            if parent_action != candidate_action or parent_action == "HOLD":
                break
            atom_index += 1
    return tuple(samples)


def probe_policy_episode(
    candidate_root: str | Path,
    replay_path: str | Path,
    *,
    match_id: str,
    role: str,
    timeout_s: float = 60.0,
    command_prefix: tuple[str, ...] = (),
) -> PolicyEpisodeTrace:
    candidate = Path(candidate_root).resolve()
    replay = Path(replay_path).resolve()
    worker = Path(__file__).with_name("policy_trace_worker.py").resolve()
    allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TMPDIR")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        (
            *command_prefix,
            sys.executable,
            str(worker),
            "--candidate",
            str(candidate),
            "--replay",
            str(replay),
            "--match-id",
            match_id,
            "--role",
            role,
        ),
        cwd=candidate,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )
    marker = "AGENTBENCH_POLICY_TRACE="
    payload = next(
        (
            line.removeprefix(marker)
            for line in completed.stdout.splitlines()
            if line.startswith(marker)
        ),
        None,
    )
    if completed.returncode != 0 or payload is None:
        diagnostic = " ".join(
            (completed.stderr or completed.stdout or "missing trace output").split()
        )[-4000:]
        raise RuntimeError(f"policy trace worker failed: {diagnostic}")
    value = json.loads(payload)
    if not isinstance(value, dict) or not isinstance(value.get("decisions"), list):
        raise ValueError("policy trace worker returned malformed output")
    decisions = tuple(
        PolicyDecision(
            state_id=str(item["state_id"]),
            actions=tuple(str(action) for action in item["actions"]),
            legal_supports=tuple(
                tuple(str(action) for action in support) for support in item["legal_supports"]
            ),
            occupancy_id=str(item["occupancy_id"]),
        )
        for item in value["decisions"]
        if isinstance(item, dict)
    )
    if len(decisions) != len(value["decisions"]):
        raise ValueError("policy trace worker returned a malformed decision")
    return PolicyEpisodeTrace(str(value["match_id"]), str(value["role"]), decisions)


@dataclass(frozen=True)
class StateDivergence:
    state_id: str
    kl_nats: float
    disagreed: bool


@dataclass(frozen=True)
class BehaviorComparison:
    trace: tuple[StateDivergence, ...]
    mean_kl_nats: float | None
    disagreement_rate: float | None
    occupancy_shift: float | None


def _occupancy_total_variation(samples: tuple[DecisionSample, ...]) -> float | None:
    if not samples:
        return None
    parent = Counter(item.parent_occupancy for item in samples)
    candidate = Counter(item.candidate_occupancy for item in samples)
    support = set(parent) | set(candidate)
    count = len(samples)
    return 0.5 * sum(abs(parent[label] / count - candidate[label] / count) for label in support)


def occupancy_total_variation(
    parent_state_ids: tuple[str, ...],
    candidate_state_ids: tuple[str, ...],
) -> float | None:
    if not parent_state_ids or not candidate_state_ids:
        return None
    parent = Counter(parent_state_ids)
    candidate = Counter(candidate_state_ids)
    support = set(parent) | set(candidate)
    return 0.5 * sum(
        abs(
            parent[state_id] / len(parent_state_ids)
            - candidate[state_id] / len(candidate_state_ids)
        )
        for state_id in support
    )


def compare_decisions(
    samples: tuple[DecisionSample, ...],
    *,
    epsilon: float,
) -> BehaviorComparison:
    trace = tuple(
        StateDivergence(
            state_id=sample.state_id,
            kl_nats=epsilon_regularized_kl(
                sample.parent_action,
                sample.candidate_action,
                sample.legal_actions,
                epsilon,
            ),
            disagreed=sample.parent_action != sample.candidate_action,
        )
        for sample in samples
    )
    if not trace:
        return BehaviorComparison((), None, None, None)
    return BehaviorComparison(
        trace=trace,
        mean_kl_nats=sum(item.kl_nats for item in trace) / len(trace),
        disagreement_rate=sum(item.disagreed for item in trace) / len(trace),
        occupancy_shift=_occupancy_total_variation(samples),
    )

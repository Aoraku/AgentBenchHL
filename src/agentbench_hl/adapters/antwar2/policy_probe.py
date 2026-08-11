"""Probe AntWar2 deterministic atomic decisions on frozen public states.

The game-agnostic comparison records and reducers now live in
``agentbench_hl.domain.policy``.  This adapter only implements the AntWar2
specific parts: enumerating legal atomic operations through the frozen SDK and
running the isolated policy-trace worker.  The domain types are re-exported for
backward compatibility with existing callers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agentbench_hl.domain.policy import (
    BehaviorComparison,
    DecisionSample,
    PolicyDecision,
    PolicyEpisodeTrace,
    StateDivergence,
    compare_decisions,
    compare_policy_episode,
    occupancy_total_variation,
)

__all__ = [
    "BehaviorComparison",
    "DecisionSample",
    "PolicyDecision",
    "PolicyEpisodeTrace",
    "StateDivergence",
    "action_key",
    "compare_decisions",
    "compare_policy_episode",
    "enumerate_legal_atomic_actions",
    "occupancy_total_variation",
    "probe_policy_episode",
]


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

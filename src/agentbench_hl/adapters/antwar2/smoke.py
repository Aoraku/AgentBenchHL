"""Candidate-side import and public-state legality smoke test."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_MARKER = "AGENTBENCH_SMOKE_RESULT="
_PROGRAM = r"""
import json
from ai import AI
from SDK.backend.state import create_python_backend_state

counts = {}
for player in (0, 1):
    agent = AI()
    agent.on_match_start(player, 7)
    state = create_python_backend_state(seed=7)
    operations = agent.choose_operations(state, player)
    if not isinstance(operations, list):
        raise TypeError("AI.choose_operations must return a list")
    accepted = []
    for operation in operations:
        if not state.can_apply_operation(player, operation, accepted):
            raise ValueError(f"role P{player} emitted an illegal operation: {operation}")
        accepted.append(operation)
    counts[f"P{player}"] = len(accepted)
print("AGENTBENCH_SMOKE_RESULT=" + json.dumps({"counts": counts}, sort_keys=True))
"""


@dataclass(frozen=True)
class ValidationResult:
    status: str
    error: str | None
    artifacts: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))


def verify_smoke(
    candidate_root: str | Path,
    timeout_s: float = 30.0,
    *,
    command_prefix: tuple[str, ...] = (),
) -> ValidationResult:
    candidate = Path(candidate_root).resolve()
    for relative in ("ai.py", "main.py", "common.py", "protocol.py", "SDK/__init__.py"):
        if not (candidate / relative).is_file():
            return ValidationResult("failed", f"candidate package is missing {relative}", {})
    allowed = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TMPDIR")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        completed = subprocess.run(
            (*command_prefix, sys.executable, "-c", _PROGRAM),
            cwd=candidate,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=float(timeout_s),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ValidationResult("failed", f"smoke failed: {exc}", {})
    payload = next(
        (
            line.removeprefix(_MARKER)
            for line in completed.stdout.splitlines()
            if line.startswith(_MARKER)
        ),
        None,
    )
    if completed.returncode != 0 or payload is None:
        diagnostic = " ".join(
            (completed.stderr or completed.stdout or "missing smoke result").split()
        )[-4000:]
        return ValidationResult(
            "failed",
            f"candidate smoke rejected: {diagnostic}",
            {"returncode": completed.returncode},
        )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        return ValidationResult(
            "failed",
            f"candidate smoke result is invalid JSON: {exc}",
            {"returncode": completed.returncode},
        )
    return ValidationResult(
        "complete",
        None,
        {
            "roles": ["P0", "P1"],
            "accepted_operation_counts": value["counts"],
            "policy_sha256": hashlib.sha256((candidate / "ai.py").read_bytes()).hexdigest(),
            "returncode": completed.returncode,
        },
    )

"""macOS Seatbelt 候选隔离（原 ``codex_goal.read_isolation`` 的行为）。"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.ports.isolation import IsolationRequest, IsolationUnavailable

SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def write_candidate_profile(path: str | Path, request: IsolationRequest) -> Path:
    """写出"禁隐藏读 + 禁写 + 禁网"的 Seatbelt 配置。"""

    normalized = request.normalized()
    if not normalized.denied_read_roots:
        raise ValueError("candidate isolation requires at least one denied read root")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["(version 1)", "(allow default)"]
    if not normalized.allow_network:
        lines.append("(deny network*)")
        # Seatbelt 的 network* 同时覆盖 unix socket，因此定点放通必须显式写出来。
        lines.extend(
            f"(allow network-outbound (literal {json.dumps(str(path))}))"
            for path in normalized.allowed_unix_sockets
        )
    lines.append("(deny file-write*)")
    lines.extend(
        f"(allow file-write* (subpath {json.dumps(str(root))}))"
        for root in normalized.writable_roots
    )
    lines.extend(
        f"(deny file-read* (subpath {json.dumps(str(root))}))"
        for root in normalized.denied_read_roots
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


@dataclass(frozen=True)
class SeatbeltIsolation:
    profile_path: Path
    request: IsolationRequest
    kind: str = "seatbelt"

    @classmethod
    def create(cls, profile_path: str | Path, request: IsolationRequest) -> SeatbeltIsolation:
        if sys.platform != "darwin":
            raise IsolationUnavailable("Seatbelt isolation requires macOS")
        if not Path(SANDBOX_EXEC).exists():
            raise IsolationUnavailable(f"{SANDBOX_EXEC} is unavailable")
        profile = write_candidate_profile(profile_path, request)
        return cls(profile_path=profile, request=request.normalized())

    def command_prefix(self) -> tuple[str, ...]:
        return (SANDBOX_EXEC, "-f", str(self.profile_path))

    def wrap(self, command: Sequence[str]) -> tuple[str, ...]:
        return (*self.command_prefix(), *command)

    def describe(self) -> Mapping[str, object]:
        return {
            "kind": self.kind,
            "profile": str(self.profile_path),
            "network": "denied" if not self.request.allow_network else "allowed",
            "allowed_unix_sockets": [str(item) for item in self.request.allowed_unix_sockets],
            "writes": "denied",
            "denied_read_roots": [str(item) for item in self.request.denied_read_roots],
        }

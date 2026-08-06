"""Outer macOS Seatbelt denial for hidden research inputs."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path


def write_read_isolation_profile(
    path: str | Path,
    *,
    denied_read_roots: Sequence[str | Path],
) -> Path:
    if sys.platform != "darwin":
        raise RuntimeError("strict read isolation currently requires macOS Seatbelt")
    roots = tuple(sorted({Path(item).resolve() for item in denied_read_roots}))
    if not roots:
        raise ValueError("read isolation requires at least one denied root")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = ["(version 1)", "(allow default)"]
    lines.extend(f"(deny file-read* (subpath {json.dumps(str(root))}))" for root in roots)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_candidate_isolation_profile(
    path: str | Path,
    *,
    denied_read_roots: Sequence[str | Path],
) -> Path:
    """Deny hidden reads, all writes, and networking for candidate execution."""

    if sys.platform != "darwin":
        raise RuntimeError("strict candidate isolation currently requires macOS Seatbelt")
    roots = tuple(sorted({Path(item).resolve() for item in denied_read_roots}))
    if not roots:
        raise ValueError("candidate isolation requires at least one denied read root")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-write*)",
    ]
    lines.extend(f"(deny file-read* (subpath {json.dumps(str(root))}))" for root in roots)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def isolated_app_server_command(
    command: Sequence[str],
    profile: str | Path,
) -> tuple[str, ...]:
    if not command:
        raise ValueError("isolated command cannot be empty")
    return ("/usr/bin/sandbox-exec", "-f", str(Path(profile).resolve()), *command)

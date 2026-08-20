"""按平台与配置选择候选隔离后端。

选择顺序（``backend="auto"``）：

1. macOS → Seatbelt；
2. Linux 且有 ``bwrap`` → bubblewrap；
3. Linux 且有 ``docker`` → 容器；
4. 都没有 → 抛 :class:`IsolationUnavailable`（**绝不静默降级**，因为静默降级会让
   "候选不能联网/不能偷看人类源码"这条科学前提失效）。

只有配置显式写 ``isolation.backend: disabled`` 才允许无隔离运行，且会在 run
清单里留痕（``kind=disabled``）。
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from agentbench_hl.adapters.isolation.bubblewrap import BubblewrapIsolation
from agentbench_hl.adapters.isolation.docker import DockerIsolation
from agentbench_hl.adapters.isolation.seatbelt import SeatbeltIsolation
from agentbench_hl.ports.isolation import (
    CandidateIsolation,
    DisabledIsolation,
    IsolationRequest,
    IsolationUnavailable,
)

BACKENDS = ("auto", "seatbelt", "bubblewrap", "docker", "disabled")


def select_candidate_isolation(
    request: IsolationRequest,
    *,
    backend: str = "auto",
    profile_path: str | Path | None = None,
    docker_image: str | None = None,
) -> CandidateIsolation:
    """返回一个满足 ``request`` 的隔离实现。"""

    if backend not in BACKENDS:
        raise ValueError(f"unknown isolation backend {backend!r}; choose from {BACKENDS}")

    if backend == "disabled":
        return DisabledIsolation(reason="isolation.backend=disabled in experiment config")

    if backend in ("auto", "seatbelt") and sys.platform == "darwin":
        if profile_path is None:
            raise ValueError("seatbelt isolation requires a profile_path")
        return SeatbeltIsolation.create(profile_path, request)
    if backend == "seatbelt":
        raise IsolationUnavailable("seatbelt isolation requires macOS")

    if backend in ("auto", "bubblewrap") and shutil.which("bwrap"):
        return BubblewrapIsolation.create(request)
    if backend == "bubblewrap":
        raise IsolationUnavailable("bubblewrap (bwrap) is unavailable on this host")

    if backend in ("auto", "docker") and shutil.which("docker"):
        if docker_image:
            return DockerIsolation.create(request, image=docker_image)
        return DockerIsolation.create(request)
    if backend == "docker":
        raise IsolationUnavailable("docker is unavailable on this host")

    raise IsolationUnavailable(
        "no strong candidate isolation available: install bubblewrap (Linux) or run on macOS; "
        "set isolation.backend=disabled only for local development"
    )

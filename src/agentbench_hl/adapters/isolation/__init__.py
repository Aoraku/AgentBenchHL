"""候选执行隔离的具体实现（按平台/配置选择）。

- ``seatbelt``  : macOS Seatbelt（``sandbox-exec``），原有行为，保持不变。
- ``bubblewrap``: Linux bubblewrap（``bwrap``），与 Seatbelt 语义等价：
                  整机只读 + 遮蔽隐藏材料 + 断网 + 仅 tmpfs 可写。
- ``docker``    : 容器隔离（``--network none --read-only``），无 bwrap 时的兜底。
- ``disabled``  : 显式关闭（仅开发/CI）。

选择顺序见 :func:`select_candidate_isolation`。
"""

from __future__ import annotations

from agentbench_hl.adapters.isolation.bubblewrap import BubblewrapIsolation
from agentbench_hl.adapters.isolation.docker import DockerIsolation
from agentbench_hl.adapters.isolation.seatbelt import SeatbeltIsolation
from agentbench_hl.adapters.isolation.select import select_candidate_isolation
from agentbench_hl.ports.isolation import (
    CandidateIsolation,
    DisabledIsolation,
    IsolationRequest,
    IsolationUnavailable,
)

__all__ = [
    "BubblewrapIsolation",
    "CandidateIsolation",
    "DisabledIsolation",
    "DockerIsolation",
    "IsolationRequest",
    "IsolationUnavailable",
    "SeatbeltIsolation",
    "select_candidate_isolation",
]

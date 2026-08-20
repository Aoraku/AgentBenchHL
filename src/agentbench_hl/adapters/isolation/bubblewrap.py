"""Linux bubblewrap 候选隔离 —— 与 macOS Seatbelt 语义等价。

等价性对照（已在 Debian 13 / 内核 6.12 实测）：

| 目标 | Seatbelt | bubblewrap |
| --- | --- | --- |
| 禁网 | ``(deny network*)`` | ``--unshare-net`` |
| 禁写 | ``(deny file-write*)`` | ``--ro-bind / /``（仅 tmpfs 可写） |
| 遮蔽隐藏材料 | ``(deny file-read* (subpath …))`` | 目录 → ``--tmpfs``；文件 → ``/dev/null`` |
| 临时可写 | ``/tmp`` 允许 | ``--tmpfs /tmp``（+ 把落在 /tmp 下的可读根重新挂回） |

额外收益：``--unshare-pid/ipc/uts`` + ``--die-with-parent`` 让候选进程无法逃逸或残留。
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.ports.isolation import IsolationRequest, IsolationUnavailable


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class BubblewrapIsolation:
    """用 ``bwrap`` 实施"整机只读 + 定点遮蔽 + 断网"的候选隔离。"""

    executable: str
    request: IsolationRequest
    kind: str = "bubblewrap"

    @classmethod
    def create(
        cls, request: IsolationRequest, *, executable: str | None = None
    ) -> BubblewrapIsolation:
        found = executable or shutil.which("bwrap")
        if not found:
            raise IsolationUnavailable(
                "bubblewrap (bwrap) is unavailable; install it (apt install bubblewrap) "
                "or select another isolation backend"
            )
        normalized = request.normalized()
        if not normalized.denied_read_roots:
            raise ValueError("candidate isolation requires at least one denied read root")
        return cls(executable=found, request=normalized)

    def command_prefix(self) -> tuple[str, ...]:
        request = self.request
        arguments: list[str] = [
            self.executable,
            "--unshare-ipc",
            "--unshare-pid",
            "--unshare-uts",
            "--die-with-parent",
            "--new-session",
            # 整机只读：候选能读系统与后端资产，但任何写入都失败。
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            # 唯一默认可写区域：私有 tmpfs。
            "--tmpfs",
            "/tmp",
        ]
        if not request.allow_network:
            arguments.append("--unshare-net")
        # 顺序很重要：先遮蔽隐藏材料，再把确实需要的路径挂回来。
        # 这样"遮蔽整个人类选手池 + 只放开本局对手包"是一条自然的表达。
        for root in request.denied_read_roots:
            if not root.exists():
                continue
            if root.is_dir():
                arguments.extend(("--tmpfs", str(root)))
            else:
                arguments.extend(("--ro-bind", "/dev/null", str(root)))
        for root in request.readable_roots:
            if root.exists():
                arguments.extend(("--ro-bind", str(root), str(root)))
        for root in request.writable_roots:
            root.mkdir(parents=True, exist_ok=True)
            arguments.extend(("--bind", str(root), str(root)))
        if request.scratch_dir is not None:
            # 顺序在 --tmpfs /tmp 之后：把宿主 scratch 目录挂回同名路径，
            # 否则沙箱里 TMPDIR 指向的目录不存在（make/g++ 会直接失败）。
            request.scratch_dir.mkdir(parents=True, exist_ok=True)
            arguments.extend(("--bind", str(request.scratch_dir), str(request.scratch_dir)))
            arguments.extend(("--setenv", "TMPDIR", str(request.scratch_dir)))
        return tuple(arguments)

    def wrap(self, command: Sequence[str]) -> tuple[str, ...]:
        return (*self.command_prefix(), *command)

    def describe(self) -> Mapping[str, object]:
        return {
            "kind": self.kind,
            "executable": self.executable,
            "network": "denied" if not self.request.allow_network else "allowed",
            # unix socket 是文件系统对象，不随 netns 走：断网后它是唯一可控的放通通道。
            "allowed_unix_sockets": [str(item) for item in self.request.allowed_unix_sockets],
            "writes": "denied (tmpfs + declared writable roots only)",
            "denied_read_roots": [str(item) for item in self.request.denied_read_roots],
            "writable_roots": [str(item) for item in self.request.writable_roots],
        }

"""容器候选隔离 —— 无 bwrap 时的兜底（也可用于跨主机执行面）。

``docker run --network none --read-only`` 与 Seatbelt 语义等价；隐藏材料通过
"不挂载"实现（容器内根本看不到宿主机路径），比遮蔽更强。

注意：候选包与后端可执行必须显式挂进容器（只读）。因此本实现要求调用方在
``IsolationRequest.readable_roots`` 里给出确切需要的路径。
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.ports.isolation import IsolationRequest, IsolationUnavailable

DEFAULT_IMAGE = "agentbench/candidate-runtime:latest"


@dataclass(frozen=True)
class DockerIsolation:
    executable: str
    image: str
    request: IsolationRequest
    cpus: float = 1.0
    memory: str = "2g"
    pids_limit: int = 256
    kind: str = "docker"

    @classmethod
    def create(
        cls,
        request: IsolationRequest,
        *,
        image: str = DEFAULT_IMAGE,
        executable: str | None = None,
        cpus: float = 1.0,
        memory: str = "2g",
        pids_limit: int = 256,
    ) -> DockerIsolation:
        found = executable or shutil.which("docker")
        if not found:
            raise IsolationUnavailable("docker is unavailable")
        normalized = request.normalized()
        if not normalized.readable_roots:
            raise ValueError("docker isolation requires explicit readable_roots to mount")
        return cls(
            executable=found,
            image=image,
            request=normalized,
            cpus=cpus,
            memory=memory,
            pids_limit=pids_limit,
        )

    def command_prefix(self) -> tuple[str, ...]:
        arguments: list[str] = [
            self.executable,
            "run",
            "--rm",
            "-i",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,exec,size=512m",
            f"--cpus={self.cpus}",
            f"--memory={self.memory}",
            f"--pids-limit={self.pids_limit}",
            "--security-opt",
            "no-new-privileges",
            "-u",
            "65534:65534",
        ]
        if not self.request.allow_network:
            arguments.extend(("--network", "none"))
        for root in self.request.readable_roots:
            arguments.extend(("-v", f"{root}:{root}:ro"))
        arguments.extend(("-w", str(self.request.readable_roots[0]), self.image))
        return tuple(arguments)

    def wrap(self, command: Sequence[str]) -> tuple[str, ...]:
        return (*self.command_prefix(), *command)

    def describe(self) -> Mapping[str, object]:
        return {
            "kind": self.kind,
            "image": self.image,
            "network": "denied" if not self.request.allow_network else "allowed",
            "writes": "denied (read-only rootfs + tmpfs)",
            "limits": {"cpus": self.cpus, "memory": self.memory, "pids": self.pids_limit},
            "mounts": [str(item) for item in self.request.readable_roots],
        }


def container_paths(*roots: Path) -> tuple[Path, ...]:
    """容器内路径与宿主机一致（我们按同路径挂载），便于复现与审计。"""

    return tuple(Path(item).resolve() for item in roots)

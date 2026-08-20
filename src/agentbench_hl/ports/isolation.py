"""候选执行隔离契约（游戏无关、平台无关）。

HL 的科学有效性依赖两件事：Goal 只能看到 GamePack 允许的公开材料；候选策略在
对局中不能联网、不能写盘、不能偷看隐藏材料（人类源码 / 认证矩阵 / 参考策略 /
凭据）。**如何**实施隔离是平台细节（macOS Seatbelt / Linux bubblewrap / 容器），
因此抽象成本 port，由 ``adapters/isolation/`` 提供实现。

框架 core 与各游戏 factory 只依赖本契约，不出现任何平台字符串。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class IsolationRequest:
    """一次隔离请求：谁可读、谁必须不可读、临时可写目录放哪。

    - ``readable_roots``：候选运行确实需要读的路径（候选包、后端可执行、公开 SDK）。
      实现可以选择"整机只读 + 定点遮蔽"，只要保证 ``denied_read_roots`` 不可读。
    - ``denied_read_roots``：必须不可读的隐藏材料（含单个文件，如 ``.env``）。
    - ``scratch_dir``：允许写的临时目录；None 表示实现自行提供（如 tmpfs /tmp）。
    - ``allowed_unix_sockets``：即使在 ``allow_network=False`` 下也**允许连接**的
      unix socket（唯一用途：把 harness 定点放通到本机代理，见
      :mod:`agentbench_hl.adapters.isolation.uds_gateway`）。unix socket 不属于
      任何网络命名空间，因此这条放通不会带回"能上公网"的能力。
    """

    denied_read_roots: tuple[Path, ...]
    readable_roots: tuple[Path, ...] = ()
    writable_roots: tuple[Path, ...] = ()
    scratch_dir: Path | None = None
    allow_network: bool = False
    allowed_unix_sockets: tuple[Path, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def normalized(self) -> IsolationRequest:
        return IsolationRequest(
            denied_read_roots=tuple(
                sorted({Path(item).resolve() for item in self.denied_read_roots})
            ),
            readable_roots=tuple(sorted({Path(item).resolve() for item in self.readable_roots})),
            writable_roots=tuple(sorted({Path(item).resolve() for item in self.writable_roots})),
            scratch_dir=None if self.scratch_dir is None else Path(self.scratch_dir).resolve(),
            allow_network=self.allow_network,
            allowed_unix_sockets=tuple(
                sorted({Path(item).resolve() for item in self.allowed_unix_sockets})
            ),
            metadata=dict(self.metadata),
        )


@runtime_checkable
class CandidateIsolation(Protocol):
    """把一条候选执行命令包装进隔离环境。"""

    kind: str

    def command_prefix(self) -> tuple[str, ...]:
        """返回要前置到候选命令前的参数（可为空元组=无包装）。"""
        ...

    def wrap(self, command: Sequence[str]) -> tuple[str, ...]:
        """包装完整命令。"""
        ...

    def describe(self) -> Mapping[str, object]:
        """写进 run 清单的可审计描述（用于复现与审计）。"""
        ...


class IsolationUnavailable(RuntimeError):
    """当前主机无法提供满足请求的强隔离。"""


@dataclass(frozen=True)
class DisabledIsolation:
    """显式关闭隔离（仅供本地开发/CI；run 清单会记录 kind=disabled）。"""

    reason: str = "explicitly disabled"
    kind: str = "disabled"

    def command_prefix(self) -> tuple[str, ...]:
        return ()

    def wrap(self, command: Sequence[str]) -> tuple[str, ...]:
        return tuple(command)

    def describe(self) -> Mapping[str, object]:
        return {"kind": self.kind, "reason": self.reason, "network": "unrestricted"}

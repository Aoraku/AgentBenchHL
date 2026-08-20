"""CPU 租约 —— 让每一局对局获得独占核，保证墙钟超时判定公平。

**为什么必须有**：A 的对战器用**墙钟**做每步超时判定。如果多局对局共享同一批核，
计算重的选手会因为邻居抢占而被判"超时"，于是"慢但合法"的策略被系统性误杀
（实测：rank05 vs rank01 串行 150 s 正常完成 308 回合；4 局并行时同一对局全部
判超时）。跑分一旦受并行度影响，整个基准就失去可比性。

**做法**：用文件锁做**跨进程** CPU 租约。每局开始前租下 ``cpus_per_match`` 个核，
用 ``taskset`` 绑定；结束后释放。web 进程内的多个 job 与独立的 ``abhl`` 子进程
共享同一套租约目录，因此不会互相超订。

**降级**：租不到核时（超时）仍然会跑，但会在对局 payload 里标注 ``cpu_pinned=false``，
方便事后剔除受污染的样本 —— 绝不静默。
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

DEFAULT_LEASE_ROOT = Path(os.environ.get("ABHL_CPU_LEASE_ROOT", "/tmp/abhl-cpu-leases"))
# 预留给控制面/系统的核（默认前两个）。
RESERVED_CPUS = int(os.environ.get("ABHL_RESERVED_CPUS", "2"))


def available_cpus() -> tuple[int, ...]:
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - 非 Linux
        allowed = list(range(os.cpu_count() or 1))
    return tuple(allowed[RESERVED_CPUS:]) or tuple(allowed)


@contextlib.contextmanager
def lease_cpus(
    count: int,
    *,
    lease_root: Path | None = None,
    timeout_s: float = 600.0,
    poll_s: float = 0.5,
) -> Iterator[tuple[int, ...]]:
    """租下 ``count`` 个独占核；租不到则在超时后以空元组降级。"""

    if count <= 0:
        yield ()
        return
    root = Path(lease_root or DEFAULT_LEASE_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    pool = available_cpus()
    if not pool:
        yield ()
        return
    deadline = time.time() + timeout_s
    handles: list = []
    acquired: list[int] = []
    try:
        while True:
            for cpu in pool:
                if len(acquired) >= count:
                    break
                handle = (root / f"cpu-{cpu}.lock").open("w")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    handle.close()
                    continue
                handles.append(handle)
                acquired.append(cpu)
            if len(acquired) >= count or time.time() >= deadline:
                break
            # 没凑齐就全部退还，避免多进程互相持有一半造成活锁。
            for handle in handles:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            handles, acquired = [], []
            time.sleep(poll_s)
        yield tuple(acquired)
    finally:
        for handle in handles:
            with contextlib.suppress(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def taskset_prefix(cpus: Sequence[int]) -> tuple[str, ...]:
    """把 CPU 列表变成 ``taskset`` 前缀（空列表 = 不绑定）。"""

    if not cpus:
        return ()
    return ("taskset", "-c", ",".join(str(cpu) for cpu in sorted(cpus)))

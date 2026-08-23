"""慢评测（后台全池评测）的生命周期管理。

它解决什么问题
--------------
``evaluation.background_pool`` 这个配置项长期**只被解析、从来没有被消费**：
配置里写 ``true`` 什么也不会发生，慢评测一直靠人手动敲
``scripts/pool_elo_worker.py``。于是"每 3 轮自动测一次中间版本在全池的排名"
这件事在每个 run 上都要重做一遍，而漏做的表现是**图上少一条线**——
不报错，只是 Elo 面板里没有那条实测曲线。

本模块把它接上：``background_pool: true`` 时，启动脚本调用 :func:`spawn`
起一个独立的 worker 进程，与迭代同生共死。

为什么必须是独立进程（而不是 driver 里的一段代码）
------------------------------------------------
1. **不能共写事件账本**。主迭代与慢评测各写各的文件：worker 只写
   ``pool-elo/``，主账本 ``events.jsonl`` 由迭代进程独占。两个进程追加同一个
   账本会让对局记录交错，而这从曲线上看不出来。
2. **不能拖慢迭代**。一个版本打完 229 人池要几百局；放在迭代循环里就变成
   "每 3 轮卡住半小时"。worker 自带 CPU 水位控制（``--headroom``），
   忙的时候自己停下等。
3. **要能单独重启**。慢评测崩了不该影响迭代，反之亦然。

抽样口径
--------
``--best-only --iteration-stride N``：只测每轮被选中的最佳候选（那才是演进
主线，落选候选是探索分支），且每 N 轮取一版。相邻轮次的池内 Elo 差异远小于
±50 的标准误，抽样不改变曲线形状但成本降一个数量级。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: 每几轮取一个版本做全池评测。
#:
#: 3 是成本与分辨率的平衡点：32 轮 → 11 个版本 × 约 458 局 ≈ 5k 局，
#: 在 32 核上与迭代并行跑得完；stride=1 会变成 32 版 ≈ 15k 局，
#: 慢评测会一直追不上迭代，图上永远只有前几个点。
DEFAULT_STRIDE = 3


@dataclass(frozen=True)
class SlowEvalPlan:
    """一次慢评测的启动计划（便于测试断言，不必真起进程）。"""

    command: tuple[str, ...]
    log_path: Path
    run_root: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "log_path": str(self.log_path),
            "run_root": str(self.run_root),
        }


def build_plan(
    *,
    run_root: Path,
    agentbench_root: Path,
    game: str,
    repository_root: Path,
    seeds: tuple[int, ...] = (7,),
    stride: int = DEFAULT_STRIDE,
    challenger_track: str | None = None,
    python_executable: str | None = None,
    parallel: int | None = None,
    headroom: int = 8,
) -> SlowEvalPlan:
    """拼出 worker 的启动命令。

    ``parallel`` 不给时按"剩余核数 / 每局 3 核"估：慢评测只能用主迭代
    剩下的机时，抢机时会让 agent 思考变慢，而那占全程约 84%。
    """

    worker = repository_root / "scripts" / "pool_elo_worker.py"
    if not worker.is_file():
        raise FileNotFoundError(f"pool_elo_worker.py not found: {worker}")

    if parallel is None:
        cores = shutil.os.cpu_count() or 8  # type: ignore[attr-defined]
        parallel = max(1, (cores - headroom) // 3)

    command = [
        python_executable or sys.executable,
        "-u",
        str(worker),
        "--run-root",
        str(run_root),
        "--agentbench-root",
        str(agentbench_root),
        "--game",
        game,
        "--seeds",
        ",".join(str(item) for item in seeds),
        # 只测演进主线，且每 stride 轮一版（第 1 轮总保留）。
        "--best-only",
        "--iteration-stride",
        str(stride),
        "--parallel",
        str(parallel),
        "--headroom",
        str(headroom),
    ]
    if challenger_track:
        # 分轨游戏（rollman 等）必须指定挑战者自己扮演哪一轨，
        # 否则会出现同轨互殴（ghost 打 ghost），对局在协议层就没意义。
        command.extend(["--challenger-track", challenger_track])

    return SlowEvalPlan(
        command=tuple(command),
        log_path=run_root.parent / f"{run_root.name}.pool-elo.log",
        run_root=run_root,
    )


def already_running(run_root: Path) -> int | None:
    """这个 run 是否已经有慢评测 worker 在跑；返回它的 pid。

    为什么必须有这道检查：慢评测有两个入口 —— 新 run 由
    ``background_pool: true`` 自动挂，已经在跑的 run 用
    ``scripts/attach_slow_eval.sh`` 手动挂。两者撞车时会有两个 worker
    **并发写同一个 ``pool-elo/`` 目录**：同一版本的对局被重复调度、
    ``matches.jsonl`` 交错追加、``challenger-elo.json`` 互相覆盖。
    这些都不会报错，只会让机时白烧、数据可疑。

    读 ``/proc`` 而不是 ``pgrep -f``：pattern 里含 ``--run-root``（以 ``-``
    开头）会被 pgrep 当成选项，而绕过之后它又会匹配到调用者自己的命令行。
    两种误判都实测踩过。

    ``/proc`` 不存在时（macOS 等非 Linux 开发机）返回 ``None`` —— 宁可漏检
    也不能崩：这个函数在 run 的启动路径上，抛异常会连带打死整个 run，
    而它本身只是一道防重复的护栏。
    """

    proc = Path("/proc")
    if not proc.is_dir():
        return None

    target = str(run_root)
    self_pid = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == self_pid:
            continue
        try:
            command = (
                (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            )
        except (OSError, PermissionError):
            continue
        if "pool_elo_worker.py" in command and target in command:
            return pid
    return None


def spawn(plan: SlowEvalPlan) -> subprocess.Popen[bytes] | None:
    """起 worker 进程（detached，日志落盘）。

    已经有 worker 在跑同一个 run 时返回 ``None``（不重复起，见
    :func:`already_running`）。

    ``start_new_session=True``：慢评测不应该跟着启动它的 shell 一起被 Ctrl-C
    掉。它的生命周期由 run 决定，而不是由那个终端决定。
    """

    if already_running(plan.run_root) is not None:
        return None
    plan.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = plan.log_path.open("ab")
    return subprocess.Popen(  # noqa: S603 - 命令由本模块构造
        plan.command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )

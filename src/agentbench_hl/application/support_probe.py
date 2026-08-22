"""逐决策点真实 |A(s)| 的提供者（按游戏分发）。

为什么需要这一层
--------------
行为 IG 的 KL 在 ε-smoothing 下有闭式解 ``(m−u)·ln(m/u)``，其中 ``u = ε/|A|``、
``m = (1−ε)+u`` —— **|A| 直接定标**。用「操作类型字母表」当支持集（antwar/antwar2 都是
常量 10）会系统性偏离真值：实测 antwar 一整局真实 |A(s)| 中位数只有 **4**
（均值 4.2、最小 1、最大 40），**98% 的决策点上常量偏大**，于是每个决策点的 KL 都被压低。

这一层把「怎么拿到精确合法集」这件游戏特有的事收拢起来，
上层 ``application/behavioral_ig.py`` 只认一个 ``SupportProvider`` 协议，
没有探针的游戏就诚实回落到字母表并上报 ``support_mode=opcode_alphabet``。

两个游戏的探针机制不同，但都遵守同一条纪律：**状态必须来自真后端产出的回放**，
不能靠客户端 SDK 自己往前推。
antwar 的官方 SDK 经济结算与后端不一致（见 ``AgentBench/games/antwar/known_issues.md``：
``downgrade_tower_income`` 在塔已被移除后才调用，拆除等级 1 塔返回 −1 而后端返还
``12×2^(N−1)``），自推会逐回合累积误差；实测第 149 回合就漂了 49 金币。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

WORKER_TIMEOUT_S = 600.0

#: worker 在 stdout 上约定的输出前缀。
_MARKER = "AGENTBENCH_POLICY_TRACE="


def _worker_path(game: str) -> Path | None:
    root = Path(__file__).resolve().parents[1] / "adapters" / game / "policy_trace_worker.py"
    return root if root.is_file() else None


def _run_worker(
    worker: Path,
    candidate: Path,
    replay: Path,
    role: str,
    *,
    agentbench_root: Path | None = None,
) -> list[dict]:
    """在**子进程**里跑探针。

    必须是子进程：探针要 ``import ai``（候选自己写的模块）并把候选目录塞进
    ``sys.path``。在框架进程里做这件事会污染框架自身的模块表，
    而且连续两轮的候选同名模块会互相覆盖 —— 那种 bug 极难查。

    ``AGENTBENCH_ROOT`` 必须显式传给子进程：部分游戏（rollman）的候选包
    **不自带**后端 ``core/``（棋盘与合法动作枚举都在里面），探针要靠这个
    环境变量去 A 仓定位。候选快照在 ``runs/<run>/snapshots/`` 下，
    从那里向上找不到 AgentBench，所以没有这个变量探针就只能失败——
    而失败会被上层当作"没有探针"静默回落到字母表近似。
    """

    environment = dict(os.environ)
    if agentbench_root is not None:
        environment["AGENTBENCH_ROOT"] = str(Path(agentbench_root).resolve())

    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--candidate",
            str(candidate),
            "--replay",
            str(replay),
            "--match-id",
            "ig-probe",
            "--role",
            role,
        ],
        capture_output=True,
        text=True,
        timeout=WORKER_TIMEOUT_S,
        check=False,
        cwd=str(candidate),
        env=environment,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        raise RuntimeError(
            "policy trace worker failed: " + (detail[-1] if detail else "no stderr")
        )
    for line in reversed((completed.stdout or "").splitlines()):
        if line.startswith(_MARKER):
            payload = json.loads(line[len(_MARKER):])
            decisions = payload.get("decisions")
            return decisions if isinstance(decisions, list) else []
    raise RuntimeError("policy trace worker produced no AGENTBENCH_POLICY_TRACE line")


def support_sizes(
    game: str,
    candidate: Path,
    replay: Path,
    role: str,
    *,
    agentbench_root: Path | None = None,
) -> Sequence[int]:
    """该局每个决策点的真实 |A(s)|；拿不到就返回空序列（上层回落到字母表）。

    一个决策点可能提交多个原子动作，``legal_supports`` 因此是个列表。
    这里取**第一个**原子动作面对的合法集：线协议侧的一个「决策」对应选手写出的
    一个回复帧，也就是它在该回合做决定的那一刻，与第一个原子动作的上下文一致。
    """

    worker = _worker_path(game)
    if worker is None:
        return ()
    sizes: list[int] = []
    for decision in _run_worker(
        worker, candidate, replay, role, agentbench_root=agentbench_root
    ):
        supports = decision.get("legal_supports")
        if not isinstance(supports, list) or not supports:
            return ()
        first = supports[0]
        if not isinstance(first, list) or not first:
            return ()
        sizes.append(len(first))
    return sizes


def provider_for(game: str, *, agentbench_root: Path | None = None):
    """按游戏拿一个 ``SupportProvider``；该游戏没有探针就返回 None。

    ``agentbench_root`` 会透传给探针子进程（见 ``_run_worker``）：
    不自带后端 ``core/`` 的游戏靠它定位 A 仓，缺了就只能静默回落到近似口径。
    """

    if _worker_path(game) is None:
        return None

    def provide(candidate: Path, replay: Path, role: str) -> Sequence[int]:
        return support_sizes(
            game, candidate, replay, role, agentbench_root=agentbench_root
        )

    return provide

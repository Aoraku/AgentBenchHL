"""容器边界护栏 —— 每次 run 起来时**验证**容器里没有本不该有的东西。

为什么需要一道运行时检查，而不是"生成脚本已经过滤过了就完事"
------------------------------------------------------------
容器的内容来自三条独立的路径，任何一条回归都会让边界破功，而且**不会报错**，
只会静默地改变实验口径：

1. ``scripts/gen_candidate_support.py`` 的 deny 清单（脚手架从 A 的官方 SDK 过滤）；
2. ``GamePack.materialize()`` → ``frozen-gamepack``（曾经把整份脚手架又复制一遍进
   ``workspace/gamepack/candidate_support/``）；
3. ``config.goal.seed_policy_path``（种子策略目录会整份 copytree 进工作区）。

历史事实：第 2 条就漏过一次，容器里同时存在两份脚手架，其中一份带着
``tools/run_local_match.py``。当时没有任何报错，只是那一轮 850s 墙钟里有 530s
被 agent 拿去在容器内自对弈了 —— 表现为"模型好慢"，而不是"隔离坏了"。

所以边界必须在**装配完成后、agent 第一次思考之前**被真的扫一遍。它是 run 的
前置条件，不通过就不该开跑（一次长跑要几十小时，跑完才发现口径污染的代价太大）。

这道检查关心什么
----------------
它只管一件事：**agent 有没有能力自己完成一局对局**。三类东西给它这个能力：

* 本地对局器 / AI 实验室（``run_local_match.py``、``ai_lab.py`` …）；
* 训练脚本与可交互环境（``alphazero.py``、``train*.py``、``training/`` …）；
* 现成的完整对手（官方示例 AI、贪心参考实现）—— 有了对手，自对弈才有意义。

它**不**关心官方协议层里的状态/合法动作模块（``SDK/backend/`` 之类）：那是候选
在评测器里起得来的前提，人类选手当年拿到的也是同一份，剔掉等于改变竞赛条件。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: 被禁的文件名（不分目录，命中即算越界）。逐条列明而不是用宽 glob ——
#: ``model.py`` 看着像 RL 网络，但 antwar2 的 ``SDK/backend/model.py`` 是协议必需的
#: ``Operation`` 定义，宽匹配会把整个游戏judge成越界。
FORBIDDEN_NAMES: dict[str, str] = {
    # ① 本地对局 / 实验室
    "run_local_match.py": "local_match_runner",
    "run_packaged_match.py": "local_match_runner",
    "ai_lab.py": "local_ai_lab",
    "ai_lab_gui.py": "local_ai_lab",
    "replay_style_lab.py": "local_replay_lab",
    "evaluate_models.py": "local_evaluation",
    # ② 训练
    "alphazero.py": "rl_training",
    "train.py": "rl_training",
    "train_example.py": "rl_training",
    "train_mcts.py": "rl_training",
    "train_mcts_10epoch.py": "rl_training",
    "train_sweep.py": "rl_training",
    "ai_rl.py": "rl_policy",
    # ③ 现成的完整策略：会让第 0 轮的"裸策略"变成照抄
    "official_ai.py": "ready_made_strategy",
}

#: 被禁的目录名（连同子树）。
FORBIDDEN_DIRS: dict[str, str] = {
    "tools": "local_match_runner",
    "training": "rl_training",
    "ai_greedy": "ready_made_strategy",
}

#: 扫描时跳过的目录：agent 自己的产出与框架回写的证据不参与边界判定。
#: ``feedback/`` 是评测器回传的合法内容；``research/`` 是 agent 的经验；
#: ``.agentbench/`` 是两条通道本身；``__pycache__`` 是运行副产物。
SKIPPED_DIRS = frozenset({".agentbench", "feedback", "research", "__pycache__", ".git"})


@dataclass(frozen=True)
class Violation:
    """一处越界：容器里出现了让 agent 能自己打比赛的东西。"""

    path: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind}


class ContainerBoundaryError(RuntimeError):
    """容器边界破了。装配阶段直接失败，不允许带着污染的容器开跑。"""


def _walk(root: Path) -> Iterator[Path]:
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root)
        if any(part in SKIPPED_DIRS for part in relative.parts):
            continue
        yield item


def scan(workspace: str | Path) -> tuple[Violation, ...]:
    """扫描容器，返回全部越界项（空元组 = 边界完好）。"""

    root = Path(workspace)
    if not root.is_dir():
        return ()
    violations: list[Violation] = []
    for item in _walk(root):
        relative = item.relative_to(root).as_posix()
        if item.is_dir():
            kind = FORBIDDEN_DIRS.get(item.name)
            if kind is not None:
                violations.append(Violation(relative + "/", kind))
            continue
        kind = FORBIDDEN_NAMES.get(item.name)
        if kind is not None:
            violations.append(Violation(relative, kind))
    return tuple(violations)


def assert_sealed(workspace: str | Path) -> tuple[Violation, ...]:
    """边界不完好就抛错。返回值保留给调用方做记账（正常时是空元组）。"""

    violations = scan(workspace)
    if violations:
        detail = "\n".join(f"  - {item.path}  [{item.kind}]" for item in violations)
        raise ContainerBoundaryError(
            "容器边界被破坏：工作区里出现了让 agent 能自己打比赛/自己训练的东西。\n"
            f"{detail}\n"
            "这不是性能问题，是实验有效性问题：agent 一旦能在容器内自对弈，"
            "它实际见过的轨迹数就不是框架发给它的 k 条，`trajectories_seen` 这条"
            "横坐标失效，实验三（HL vs RL）的对照前提也被污染。\n"
            "修法：重跑 `python scripts/gen_candidate_support.py` 刷新脚手架"
            "（deny 清单在该脚本的 SPECS 里），再跑 `python scripts/gen_gamepacks.py` "
            "刷新指纹；若来自 goal.seed_policy_path，请自行清理该目录。"
        )
    return violations

"""容器边界的契约测试 —— 把「agent 不能自己打比赛」这件事钉死在 CI 里。

为什么必须有这层测试，而不是靠人眼看一眼工作区
--------------------------------------------
容器的内容由三条独立路径拼出来（脚手架生成、frozen-gamepack 快照、seed_policy），
其中任何一条回归都**不会报错**，只会静默改变实验口径：agent 拿到本地对战工具后
会在容器里自对弈，于是它实际见过的轨迹数远超框架发给它的 k 条，
`trajectories_seen` 这条横坐标失效、实验三（HL vs RL）的对照前提被污染。

这类回归在事后极难发现（曲线照样在涨，只是涨的原因变了），所以这里从两侧同时钉：

* **静态**：8 个游戏的候选脚手架里搜不到本地对局器/训练脚本/现成策略；
* **行为**：护栏函数真的会对越界目录抛错，而不是只打印警告。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentbench_hl.application.container_guard import (
    ContainerBoundaryError,
    assert_sealed,
    scan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GAMEPACKS = REPO_ROOT / "gamepacks"

GAMES = ("antwar", "antwar2", "aquawar", "generals", "lostspace", "miracle", "rollman", "snakego")

#: 协议层里**必须保留**的东西。剔多了会让候选在评测器里起不来，表现为
#: 「该游戏所有候选静默 0 回合判负」——这是本项目最贵的坑，所以正向也要钉。
REQUIRED_ENTRY_FILES = ("main.py", "_bootstrap.py", "CANDIDATE_CONTRACT.md", "ai_example.py")


@pytest.mark.parametrize("game", GAMES)
def test_candidate_support_has_no_local_match_or_training(game: str) -> None:
    support = GAMEPACKS / game / "candidate_support"
    assert support.is_dir(), f"{game} 缺少候选脚手架"
    violations = scan(support)
    assert violations == (), (
        f"{game} 的候选脚手架里出现了越界文件："
        + ", ".join(f"{item.path}[{item.kind}]" for item in violations)
    )


@pytest.mark.parametrize("game", GAMES)
def test_candidate_support_keeps_protocol_layer(game: str) -> None:
    """反向约束：不能为了"干净"把候选起不来的东西也剔掉。"""

    support = GAMEPACKS / game / "candidate_support"
    for name in REQUIRED_ENTRY_FILES:
        assert (support / name).is_file(), f"{game} 的脚手架缺少 {name}"


@pytest.mark.parametrize("game", GAMES)
def test_support_provenance_records_denied_paths(game: str) -> None:
    """被剔掉的东西必须留账，否则下次没人知道为什么少了文件。"""

    provenance = json.loads(
        (GAMEPACKS / game / "candidate_support" / "SUPPORT_PROVENANCE.json").read_text(
            encoding="utf-8"
        )
    )
    assert "container_denied" in provenance, f"{game} 的 provenance 没有 container_denied 字段"
    for entry in provenance["container_denied"]:
        assert entry["path"] and entry["reason"]


def test_antwar2_dropped_the_known_offenders() -> None:
    """antwar2 是唯一自带完整本地实验室的游戏，逐条确认它们真的没了。"""

    support = GAMEPACKS / "antwar2" / "candidate_support"
    for offender in (
        "tools",
        "SDK/backend/../training",
        "SDK/training",
        "SDK/alphazero.py",
        "SDK/train_mcts.py",
        "SDK/evaluate_models.py",
        "SDK/native_adapter.py",
        "official_ai.py",
        "ai_greedy",
    ):
        assert not (support / offender).exists(), f"antwar2 容器里仍有 {offender}"
    # 协议层必须还在，否则候选根本起不来。
    for required in (
        "common.py",
        "protocol.py",
        "official_main.py",
        "SDK/backend/engine.py",
        "SDK/backend/model.py",
        "SDK/utils/actions.py",
    ):
        assert (support / required).is_file(), f"antwar2 协议层缺了 {required}"


def test_guard_rejects_a_polluted_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "tools").mkdir(parents=True)
    (workspace / "tools" / "run_local_match.py").write_text("# self play", encoding="utf-8")
    with pytest.raises(ContainerBoundaryError) as error:
        assert_sealed(workspace)
    assert "run_local_match.py" in str(error.value)


def test_guard_accepts_a_clean_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "gamepack").mkdir(parents=True)
    (workspace / "gamepack" / "rules.md").write_text("# rules", encoding="utf-8")
    (workspace / "main.py").write_text("import _bootstrap", encoding="utf-8")
    (workspace / "leaderboard.json").write_text("{}", encoding="utf-8")
    assert assert_sealed(workspace) == ()


def test_guard_ignores_agent_own_output(tmp_path: Path) -> None:
    """agent 自己写的候选与评测器回传的证据不该被误判成越界。

    这一条是刻意钉的：如果护栏把 `feedback/` 或 `research/` 也扫进去，
    agent 只要在经验里提一句 `train.py` 就会让整个 run 起不来。
    """

    workspace = tmp_path / "workspace"
    for relative in (
        ".agentbench/rollouts/v001/main.py",
        "feedback/req-1/replay.json",
        "research/EXPERIENCE.md",
        "__pycache__/main.cpython-313.pyc",
    ):
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x", encoding="utf-8")
    # 即使这些目录里出现了同名的越界文件，也不算容器越界。
    polluted = workspace / "feedback" / "req-1" / "train.py"
    polluted.write_text("# 评测器产物里的同名文件", encoding="utf-8")
    assert assert_sealed(workspace) == ()


# ---------------------------------------------------------------------------
# 真容器：把 8 个游戏的容器真的搭出来，再逐条对架构图验收
# ---------------------------------------------------------------------------
#
# 前面那些测试只看脚手架目录。但容器是**三条路径拼**出来的（脚手架 + frozen-gamepack
# 快照 + seed_policy），只查脚手架会漏掉最阴的一条：`GamePack.materialize()` 会把
# `candidate_support` 也快照进 frozen-gamepack，整份 copytree 进容器就等于**两份**
# 脚手架，其中那份死副本正是被剔文件的回归入口。所以必须搭真容器再扫。

#: 架构图 §Isolated Container 里那 6 样输入在容器内的落点。
REQUIRED_INPUTS = (
    "gamepack/rules.md",  # Game Rule
    "gamepack/decision_space.yaml",  # Decision Space
    "gamepack/replay_skill.md",  # Replay Tutorial
    "gamepack/replay_format.md",  # Replay Tutorial（字段规格）
    "CANDIDATE_CONTRACT.md",  # 提交格式
    "ai_example.py",  # 格式示例（策略是占位）
    "main.py",  # 入口
)


def _agentbench_root() -> Path | None:
    value = os.environ.get("AGENTBENCH_ROOT")
    if value and (Path(value) / "games").is_dir():
        return Path(value).resolve()
    sibling = REPO_ROOT.parent / "AgentBench"
    return sibling.resolve() if (sibling / "games").is_dir() else None


@pytest.fixture
def assembled_container(tmp_path: Path):
    """把某个游戏的容器按生产路径真的搭一遍。"""

    from agentbench_hl.adapters.contract.factory import _materialize_bootstrap
    from agentbench_hl.gamepack import GamePack

    def build(game: str) -> Path:
        root = _agentbench_root()
        if root is None:
            pytest.skip("找不到可用的 AgentBench(A) 仓")
        pack = GamePack.load(GAMEPACKS / game, agentbench_root=root)
        frozen = tmp_path / game / "frozen-gamepack"
        pack.materialize(frozen)
        bootstrap = tmp_path / game / "bootstrap"
        _materialize_bootstrap(pack, frozen, bootstrap, seed_policy=None)
        return bootstrap

    return build


@pytest.mark.parametrize("game", GAMES)
def test_assembled_container_is_sealed(game: str, assembled_container) -> None:
    """真容器里不能有本地对局器/训练脚本/现成策略。"""

    workspace = assembled_container(game)
    violations = scan(workspace)
    assert violations == (), (
        f"{game} 装配出来的容器越界："
        + ", ".join(f"{item.path}[{item.kind}]" for item in violations)
    )


@pytest.mark.parametrize("game", GAMES)
def test_assembled_container_has_the_six_inputs(game: str, assembled_container) -> None:
    """反向约束：该给的 6 样输入一样都不能少，否则 agent 无从下手。"""

    workspace = assembled_container(game)
    missing = [name for name in REQUIRED_INPUTS if not (workspace / name).is_file()]
    assert not missing, f"{game} 的容器缺少必需输入：{missing}"


@pytest.mark.parametrize("game", GAMES)
def test_assembled_container_has_no_duplicate_scaffold(game: str, assembled_container) -> None:
    """容器里只能有**一份**脚手架。

    `gamepack/candidate_support/` 是历史上的第二条泄漏路径：frozen-gamepack 会快照
    脚手架（为了留指纹），整份 copytree 进容器就多出一份死副本。它没有任何用途，
    却会让被剔除的 `tools/`、训练脚本从这里悄悄回归。
    """

    workspace = assembled_container(game)
    assert not (workspace / "gamepack" / "candidate_support").exists(), (
        f"{game} 的容器里出现了第二份脚手架（gamepack/candidate_support/）；"
        "frozen-gamepack 拷进工作区时必须排除 candidate_support"
    )


def test_container_has_no_rival_source(assembled_container) -> None:
    """容器里不能有任何人类选手的代码 —— 图上写明 agent 看不到对手。"""

    workspace = assembled_container("antwar2")
    suspicious = [
        item.relative_to(workspace).as_posix()
        for item in workspace.rglob("*")
        if item.is_dir() and item.name in {"pool", "players", "rival-source", "human-pool"}
    ]
    assert not suspicious, f"容器里出现了疑似选手池目录：{suspicious}"

#!/usr/bin/env python3
"""装配候选脚手架（``candidate_support``）：**vendor 官方 SDK + 统一入口**。

设计原则
--------
对战器以 ``python main.py`` 启动候选，所以候选包必须是**完整可运行的选手程序**，
不是"策略函数"。以前的方案是我们自己为每个游戏重写一份协议 codec —— 这条路是错的：

* 一份写错的 codec 会让该游戏**所有**候选静默"0 回合判负"，主表上表现为
  "这个模型在这个游戏完全不行"的**假结论**；
* 人类选手当年拿到的是**官方 SDK**，我们自己重写一份，候选与人类的条件就不一样了。

所以现在的做法是：把 A 仓 ``games/<game>/public_sdk/``（由
``AgentBench/scripts/extract_public_sdk.py`` 从**审计通过的 Python 选手**里
按逐字节复用票数提取出来的官方原版）原样搬进候选包，只额外放两个文件：

* ``_bootstrap.py``：8 个游戏逐字节相同，负责 ``sys.path`` 与可执行诊断；
* ``main.py``：逐游戏 30–60 行，把候选的 ``ai.AI`` 接到官方 SDK 的驱动点。
  官方的回合循环**一行都不重写**（能注入就注入，能直接调就直接调）。

统一契约
--------
所有游戏都是：**候选写 ``ai.py``，里面定义 ``class AI``**。这不是我们发明的约定，
而是 saiblo 官方 SDK 本来的约定（antwar2 官方 ``main.py`` 第一行就是
``from ai import AI``）。``AI`` 里该实现哪个方法由游戏语义决定，写在
``candidate_interface`` 与该游戏 ``sdk_interface.md`` 里——观测/动作直接就是官方 SDK
的原生对象，**不再包一层我们自己的数据结构**，因为那层翻译正是错误的来源。

用法
----
    python scripts/gen_candidate_support.py --check     # 只检查是否与 A 同步
    python scripts/gen_candidate_support.py             # 生成/刷新全部
    python scripts/gen_candidate_support.py --game snakego
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
GAMEPACKS = REPO_ROOT / "gamepacks"
SHARED = GAMEPACKS / "_shared"
RUNNERS = SHARED / "candidate_runners"
VENDOR = SHARED / "candidate_vendor"
BOOTSTRAP = SHARED / "candidate_support" / "_bootstrap.py"
SELFCHECK = SHARED / "candidate_support" / "selfcheck.py"
# 容器内自检的判定逻辑 = 框架侧前置校验的同一份代码（纯 stdlib，零内部依赖），
# 直接拷进容器命名为 selfcheck_lib.py，避免"容器里过了、框架侧又打回"的两套标准。
PREFLIGHT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "agentbench_hl"
    / "application"
    / "candidate_preflight.py"
)
AGENTBENCH_ROOT = Path(
    __import__("os").environ.get("AGENTBENCH_ROOT", REPO_ROOT.parent / "AgentBench")
)

# 官方 SDK 里与候选入口**同名**的文件必须改名，否则会遮蔽候选自己要写的 ai.py，
# 或者被我们的 main.py 覆盖掉。改名只改文件名，内容一个字节都不动。
DEFAULT_RENAMES: dict[str, str] = {
    "main.py": "official_main.py",
    "ai.py": "official_ai.py",
}

# ---------------------------------------------------------------------------
# 容器边界（隔离容器里**绝不允许**出现的东西）
# ---------------------------------------------------------------------------
# 目标框架里，容器内只有 6 样输入 + agent 自己，对外只有两条通道：
#   Action = {new code, selected rival} @k   →  Evaluator
#   Feedback = {result, replay info} @k      ←  Evaluator
# 「对局」完全由 Evaluator 负责，agent 不该**有能力**自己打比赛。
#
# 官方 SDK 原样 vendor 会把三类东西一起带进来，每一类都会破坏一个具体的实验口径：
#
#   ① 本地对局 / AI 实验室  ⇒ agent 在容器里自对弈任意多局，它实际见过的经验量
#      根本不是框架发给它的 k 条轨迹，横坐标 `trajectories_seen` 直接失效；
#      实测一轮 850s 里约 530s（63%）就烧在这件本不该发生的事上。
#   ② 训练脚本（AlphaZero / MCTS / Gym 环境）⇒ 「HL 迭代」偷偷变成
#      「HL + 本地 RL 训练」，实验三（HL vs RL）的对照前提被污染。
#   ③ 现成的完整策略（官方示例 AI、贪心参考实现）⇒ 第 0 轮的「裸策略」不是从规则
#      推导出来的，而是抄来的，「从规则出发」这个起点不成立。
#
# 与之相对，**必须保留**的是协议层：统一入口、官方回合循环、以及被它们 import 的
# 状态/动作模块。剔多了会让候选在评测器里根本起不来，表现为「该游戏所有候选静默
# 0 回合判负」——这是本项目踩过的最贵的坑，所以这里**逐条列明**而不是用宽 glob。
# 反例：`**/model.py` 看着像 RL 网络，但 antwar2 的 `SDK/backend/model.py`
# 是协议必需的 Operation 定义，一刀切会让 antwar2 全军覆没。
#
# 匹配规则：条目既可以是文件相对路径，也可以是目录（连同子树一起剔除）。
DENY_LOCAL_MATCH = "local_match_or_lab"
DENY_TRAINING = "training_or_rl"
DENY_READY_STRATEGY = "ready_made_strategy"

# 所有游戏共用：官方示例策略一律不进容器（协议层的 official_main.py 保留）。
COMMON_DENY: dict[str, str] = {
    "official_ai.py": DENY_READY_STRATEGY,
}


@dataclass(frozen=True)
class GameSpec:
    """一个游戏的候选脚手架装配规格。"""

    game: str
    interface: str
    # 非对称游戏按角色轨各发一份 SDK（rollman：吃豆人交 1 个方向、幽灵交 3 个）
    tracks: tuple[str, ...] = ()
    renames: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_RENAMES))
    # 官方 SDK 里不该进候选包的文件（协议无关、或与候选入口冲突）
    drop: tuple[str, ...] = ()
    # 容器边界：路径 -> 剔除理由（写进 SUPPORT_PROVENANCE.json 的 dropped 记账）
    deny: dict[str, str] = field(default_factory=dict)
    # `ai_example.py` 的方法体：**只演示接口**，策略部分必须是明显的占位。
    example_body: str = ""

    def container_deny(self) -> dict[str, str]:
        merged = dict(COMMON_DENY)
        merged.update(self.deny)
        return merged

    def sdk_dir_name(self, track: str | None) -> str:
        return "public_sdk" if track is None else f"public_sdk-{track}"

    def support_dir_name(self, track: str | None = None) -> str:
        """候选包目录名。

        **非对称游戏也只出一个包。** 这一点是被 B 侧的运行时结构逼出来的：
        ``goal_led_service`` 派活时是 ``for role in roles: for seed in seeds:``，
        整轮共用**同一个** ``candidate_root``（一个候选快照要打完所有角色）。
        所以"每轨一个候选包"根本没有机会被分别投放；而 manifest 里写成
        ``{track: dir}`` 还会让 ``gamepack.path_for`` 直接抛
        ``GamePackError: field 'candidate_support' must be a non-empty string``
        ——rollman 一挂实验就崩在搭工作区那一步。

        正确形态是**单包内按轨放两套官方 SDK**（``<track>_sdk/``），
        入口在运行时按 ``AGENTBENCH_ROLE`` 选择。
        """

        return "candidate_support"

    def sdk_prefix(self, track: str | None) -> str:
        """该轨的官方 SDK 在候选包内的相对前缀（对称游戏是包根）。"""

        return "" if track is None else f"{track}_sdk"


SPECS: dict[str, GameSpec] = {
    "antwar": GameSpec(
        "antwar",
        interface="AI.decide",
        example_body=(
            "        # 占位策略：永远返回「不移动」。真正的策略要你自己从 rules.md 推。\n"
            "        return []\n"
        ),
    ),
    # antwar2 的 BaseAgent 是 ABC，唯一抽象方法是 choose_bundle：
    # 只覆盖 choose_operations 会因"抽象方法未实现"而**无法实例化**。
    "antwar2": GameSpec(
        "antwar2",
        interface="AI.choose_bundle",
        # 官方 antwar2 SDK 是 8 个游戏里唯一「自带完整本地实验室」的：
        # tools/ 能直接跑本地对局，SDK/ 里还塞了 AlphaZero + Gym 环境 + 贪心参考 AI。
        # 协议层真正需要的只有 common.py / protocol.py / official_main.py /
        # SDK/backend/{core,engine,forecast,model,runtime,state}.py / SDK/utils/*，
        # 下面这些都在它们的 import 图之外（native_adapter 与 training 只在
        # `load_backend(prefer_native=True)` 和 `SDK.__getattr__` 的惰性分支里出现）。
        deny={
            "tools": DENY_LOCAL_MATCH,
            "SDK/alphazero.py": DENY_TRAINING,
            "SDK/training": DENY_TRAINING,
            "SDK/train_example.py": DENY_TRAINING,
            "SDK/train_mcts.py": DENY_TRAINING,
            "SDK/train_mcts_10epoch.py": DENY_TRAINING,
            "SDK/train_sweep.py": DENY_TRAINING,
            "SDK/evaluate_models.py": DENY_LOCAL_MATCH,
            "SDK/native_adapter.py": DENY_TRAINING,
            "ai_greedy": DENY_READY_STRATEGY,
            "ai_greedy.py": DENY_READY_STRATEGY,
        },
        example_body=(
            "        # 占位策略：从官方枚举出的**合法**动作包里取第一个。\n"
            "        # 这只是为了证明协议接得上，强度为零；策略要你自己从 rules.md 推。\n"
            "        bundles = bundles or self.list_bundles(state, player)\n"
            "        return bundles[0]\n"
        ),
    ),
    "generals": GameSpec(
        "generals",
        interface="AI.decide",
        # test_sync.py 是官方自带的后端同步测试，属于「本地跑一局验证」的工具。
        deny={"generals_impact_game/test_sync.py": DENY_LOCAL_MATCH},
        example_body=(
            "        # 占位策略：直接结束本回合操作（命令 8 = 结束）。\n"
            "        return []\n"
        ),
    ),
    "lostspace": GameSpec(
        "lostspace",
        interface="AI.play",
        # 官方 SDK 是**单文件**，改名要有语义：它既是协议层也是数据结构层
        renames={"main.py": "lostspace_sdk.py"},
        example_body=(
            "        # 占位策略：什么都不做，直接结束本回合。\n"
            '        return ("finish", None)\n'
        ),
    ),
    "miracle": GameSpec(
        "miracle",
        interface="AI.play",
        example_body=(
            "        # 占位策略：过牌 / 结束回合。策略要你自己从 rules.md 推。\n"
            "        return None\n"
        ),
    ),
    "rollman": GameSpec(
        "rollman",
        interface="AI.decide",
        tracks=("rollman", "ghost"),
        # rollman 官方 SDK 直接带了 PyTorch 策略网络与训练脚本（train.py / model.py /
        # ai_rl.py）。协议层需要的是 official_main.py / ai_to_judger.py /
        # core/{gamedata,GymEnvironment,board,pacman,ghost,utils}.py / utils/utils.py。
        # 注意 core/GymEnvironment.py **必须留**：official_main.py 直接 import 它。
        deny={
            "train.py": DENY_TRAINING,
            "model.py": DENY_TRAINING,
            "ai_rl.py": DENY_TRAINING,
        },
        example_body=(
            "        # 占位策略：吃豆人返回单个方向、幽灵返回三个方向（用 self.role 分支）。\n"
            '        return [0, 0, 0] if self.role == "ghost" else [0]\n'
        ),
    ),
    "snakego": GameSpec(
        "snakego",
        interface="AI.judge",
        # ⚠️ sampleAI.py **不能剔**：官方的回合循环就在 `sampleAI.run()` 里，
        # 入口靠依赖注入替换其中的 AI 名字。它同时带了一个示例策略，属于
        # 「协议层与示例策略同一文件」的既有事实，剔掉等于没有协议层。
        example_body=(
            "        # 占位策略：永远向下移动（操作码 1）。撞墙就死，强度为零。\n"
            "        return 1\n"
        ),
    ),
    # aquawar 是唯一的例外：官方"Python SDK"其实是 pybind11 C++ 扩展
    # （sdk/py_ai_sdk.cpp + CMakeLists，pybind11_DIR 还是占位符），
    # 所以池里 194 个审计通过的选手全是 C++、没有任何 Python 选手。
    # 协议层改用 _shared/candidate_vendor/aquawar/aquawar_sdk.py（纯 Python 移植），
    # 官方那几个依赖该扩展的 .py 全部剔除——留着只会诱导候选照抄然后 import 失败。
    "aquawar": GameSpec(
        "aquawar",
        interface="AI.act",
        drop=("main.py", "Action.py", "Action_sample.py", "sdk"),
        example_body=(
            "        # 占位策略：返回第一个合法动作。策略要你自己从 rules.md 推。\n"
            "        return actions[0] if actions else None\n"
        ),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def runner_path(game: str) -> Path:
    path = RUNNERS / f"{game}.py"
    if not path.is_file():
        raise SystemExit(
            f"{game}: 缺少候选入口驱动 {path}。\n"
            "每个游戏都需要一份 30–60 行的驱动，把 ai.AI 接到官方 SDK 的驱动点。"
        )
    return path


def contract_document(spec: GameSpec, runner: Path) -> str:
    """把驱动器的模块 docstring 抽出来，作为候选包里的契约说明。

    这样"给 agent 看的说明"与"真正执行的代码"是同一个来源，不会漂移。
    """

    import ast

    tree = ast.parse(runner.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree) or ""
    header = f"# 候选契约：{spec.game}"
    if spec.tracks:
        header += f"（非对称，角色轨：{'、'.join(spec.tracks)}）"
    return (
        f"{header}\n\n"
        f"> 由 `scripts/gen_candidate_support.py` 从 `_shared/candidate_runners/{spec.game}.py`\n"
        f"> 的模块文档生成，**与实际执行的入口代码同源**。\n"
        f"> `candidate_interface` = `{spec.interface}`\n\n"
        f"{docstring}\n"
    )


def _matches(relative: str, entry: str) -> bool:
    """条目既可以精确匹配一个文件，也可以匹配一整个子树（目录前缀）。"""

    return relative == entry or relative.startswith(entry + "/")


def example_document(spec: GameSpec) -> str:
    """生成 ``ai_example.py``：**只演示接口**，策略部分是明显的占位。

    为什么不能直接放官方示例 AI（``official_ai.py`` / ``ai_greedy/``）：
    第 0 轮的要求是「只有规则 → 裸策略」。放一个已经能打的完整实现，
    agent 第 0 轮大概率直接继承/照抄它（历史上 miracle 的探针就是
    ``class AI(OfficialSampleAI)``），于是「从规则出发」这个起点不成立，
    整条学习曲线的零点就没有意义了。

    文件名故意**不是** ``ai.py``：入口只加载 ``ai.py``，所以这份示例不会被误当成候选。
    """

    body = spec.example_body or "        raise NotImplementedError\n"
    method = spec.interface.split(".", 1)[1]
    base = "(BaseAgent)" if spec.game == "antwar2" else ""
    imports = "from common import BaseAgent\n\n\n" if spec.game == "antwar2" else ""
    signature = {
        "choose_bundle": "self, state, player, bundles=None",
        "decide": "self, *args, **kwargs",
        "judge": "self, snake, ctx",
        "play": "self, *args, **kwargs",
        "act": "self, *args, **kwargs",
    }.get(method, "self, *args, **kwargs")
    return (
        f'"""{spec.game} 候选**格式示例** —— 由 gen_candidate_support.py 生成。\n\n'
        "这份文件只回答一个问题：**「怎么写才符合提交格式」**。\n"
        "策略部分是刻意留白的占位（返回第一个合法动作 / 固定动作），强度为零。\n\n"
        "把它另存为 `ai.py` 再动手写你自己的策略。入口只会加载 `ai.py`，\n"
        "所以这份示例本身永远不会被当成候选提交。\n"
        f'"""\n\n'
        "from __future__ import annotations\n\n"
        f"{imports}"
        f"class AI{base}:\n"
        f"    def {method}({signature}):\n"
        f"{body}"
    )


def assemble(spec: GameSpec, *, check: bool) -> tuple[bool, str]:
    """装配一个游戏的候选脚手架（非对称游戏也只出**一个**包）。返回 (是否有变化, 说明)。"""

    tracks: tuple[str | None, ...] = spec.tracks or (None,)
    sdk_roots: dict[str | None, Path] = {}
    for track in tracks:
        sdk_root = AGENTBENCH_ROOT / "games" / spec.game / spec.sdk_dir_name(track)
        if not sdk_root.is_dir():
            return False, (
                f"缺少官方 SDK：{sdk_root}\n"
                f"      先在 A 仓跑：python scripts/extract_public_sdk.py --export "
                f"--game {spec.game}"
            )
        sdk_roots[track] = sdk_root

    runner = runner_path(spec.game)
    destination = GAMEPACKS / spec.game / spec.support_dir_name()
    staging = destination.parent / (destination.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    deny = spec.container_deny()
    dropped: list[str] = []
    denied: list[dict[str, str]] = []
    renamed: list[str] = []
    for track, sdk_root in sdk_roots.items():
        prefix = spec.sdk_prefix(track)
        for source in sorted(sdk_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(sdk_root).as_posix()
            if relative == "SDK_PROVENANCE.json":
                continue
            labelled = f"{prefix}/{relative}" if prefix else relative
            top = relative.split("/")[0]
            if relative in spec.drop or top in spec.drop:
                dropped.append(labelled)
                continue
            target_name = spec.renames.get(relative, relative)
            # 容器边界：本地对局工具 / 训练脚本 / 现成策略一律不进容器。
            # ⚠️ 必须同时按**改名后**的名字匹配：官方的 ``ai.py``（示例策略）会被改成
            # ``official_ai.py``，只查原名会让它悄悄溜进容器（miracle 就是这么漏的）。
            reason = next(
                (
                    why
                    for entry, why in deny.items()
                    if _matches(relative, entry) or _matches(target_name, entry)
                ),
                None,
            )
            if reason is not None:
                denied.append({"path": labelled, "reason": reason})
                continue
            if target_name != relative:
                change = f"{relative} → {target_name}"
                renamed.append(f"{prefix}/{change}" if prefix else change)
            target = staging / prefix / target_name if prefix else staging / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    shutil.copy2(BOOTSTRAP, staging / "_bootstrap.py")
    shutil.copy2(runner, staging / "main.py")
    (staging / "ai_example.py").write_text(example_document(spec), encoding="utf-8")
    # 容器内自检：让 agent 在提交前自己判掉"启动即崩"这类失败（1 秒 vs 一整轮）。
    # 这不是自评测——不打对局、不估胜率，只回答"判题器能不能把它拉起来"。
    shutil.copy2(SELFCHECK, staging / "selfcheck.py")
    shutil.copy2(PREFLIGHT, staging / "selfcheck_lib.py")

    # 移植层（目前只有 aquawar 用到）：官方 SDK 不是纯 Python 时，
    # 由我们自己写的协议层补位。它**不是**官方原版，provenance 里单列。
    vendored: list[str] = []
    vendor_root = VENDOR / spec.game
    if vendor_root.is_dir():
        for source in sorted(vendor_root.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(vendor_root).as_posix()
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            vendored.append(relative)

    (staging / "CANDIDATE_CONTRACT.md").write_text(
        contract_document(spec, runner), encoding="utf-8"
    )

    official: dict[str, object] = {}
    for track, sdk_root in sdk_roots.items():
        provenance = json.loads((sdk_root / "SDK_PROVENANCE.json").read_text(encoding="utf-8"))
        official[spec.sdk_prefix(track) or "."] = {
            "source": str(sdk_root.relative_to(AGENTBENCH_ROOT)),
            "track": track,
            "tree_sha256": provenance.get("tree_sha256"),
            "source_player_id": provenance.get("source_player_id"),
            "python_verified_authors": provenance.get("python_verified_authors"),
        }
    (staging / "SUPPORT_PROVENANCE.json").write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "game": spec.game,
                "tracks": [track for track in tracks if track is not None],
                "candidate_interface": spec.interface,
                "official_sdk": official,
                "renamed": renamed,
                "dropped": dropped,
                "container_denied": denied,
                "ported_protocol_layer": vendored,
                "generated_files": [
                    "_bootstrap.py",
                    "main.py",
                    "ai_example.py",
                    "selfcheck.py",
                    "selfcheck_lib.py",
                    "CANDIDATE_CONTRACT.md",
                ],
                "note": (
                    "官方 SDK 按容器边界过滤后 vendor（仅改名，未改内容）；"
                    "container_denied 记录被剔除的本地对局工具/训练脚本/现成策略——"
                    "对局只能通过 Action→Evaluator→Feedback 发生，agent 不该有能力自己打比赛；"
                    "ai_example.py 只演示接口，策略是占位，保证第 0 轮的裸策略来自规则而非照抄；"
                    "非对称游戏在单包内按 <track>_sdk/ 放多套，入口按 AGENTBENCH_ROLE 选择；"
                    "ported_protocol_layer 非官方原版，是我们的纯 Python 移植；"
                    "候选只需写 ai.py。装配逻辑见 scripts/gen_candidate_support.py"
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    new_hash = tree_sha256(staging)
    old_hash = tree_sha256(destination) if destination.is_dir() else None
    changed = new_hash != old_hash

    if check:
        shutil.rmtree(staging)
        status = "漂移" if changed else "同步"
        return changed, f"{status}（tree_sha256={new_hash[:16]}…）"

    if destination.exists():
        shutil.rmtree(destination)
    staging.rename(destination)
    files = sum(1 for item in destination.rglob("*") if item.is_file())
    detail = f"{files} 个文件，tree_sha256={new_hash[:16]}…"
    if denied:
        detail += f"，容器边界剔除 {len(denied)} 项"
    if dropped:
        detail += f"，剔除 {len(dropped)} 项"
    if renamed:
        detail += f"，改名 {len(renamed)} 项"
    return changed, detail


def update_manifest(spec: GameSpec, *, check: bool) -> str:
    """把 ``candidate_support`` / ``candidate_interface`` 写进 GamePack manifest。"""

    path = GAMEPACKS / spec.game / "manifest.yaml"
    if not path.is_file():
        return f"跳过 manifest（{path} 不存在）"
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    before = dict(document)
    # 永远是**字符串**：gamepack.path_for() 只接受字符串，写成 {track: dir} 会让
    # 该游戏在搭 Goal 工作区时直接抛 GamePackError。非对称游戏的多套 SDK 放在
    # 同一个包内的 <track>_sdk/ 子目录里，由入口按 AGENTBENCH_ROLE 选择。
    document["candidate_support"] = spec.support_dir_name()
    document["candidate_interface"] = spec.interface
    if document == before:
        return "manifest 无需变更"
    if check:
        return "manifest 需要更新（candidate_support/candidate_interface）"
    path.write_text(
        yaml.safe_dump(document, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    return "manifest 已更新"


GENERATED_BEGIN = "<!-- BEGIN candidate_support (generated by scripts/gen_candidate_support.py) -->"
GENERATED_END = "<!-- END candidate_support -->"


def _sdk_module_index(support_root: Path) -> list[str]:
    """候选包里 vendor 的官方 SDK 模块清单（供 agent 快速定位）。"""

    generated = {
        "main.py",
        "_bootstrap.py",
        "ai_example.py",
        "selfcheck.py",
        "selfcheck_lib.py",
        "CANDIDATE_CONTRACT.md",
        "SUPPORT_PROVENANCE.json",
    }
    names: list[str] = []
    for item in sorted(support_root.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(support_root).as_posix()
        if relative in generated:
            continue
        names.append(relative)
    return names


def update_sdk_interface(spec: GameSpec, *, check: bool) -> str:
    """在 ``sdk_interface.md`` 里维护一个**自动生成**的候选契约小节。

    人工写的游戏知识（决策空间、字段语义等）一个字都不动，只在文件末尾维护
    ``<!-- BEGIN candidate_support -->`` 标记块。这样"给 agent 看的接口说明"
    永远等于"实际执行的入口代码"，不会各写一套然后慢慢漂移。
    """

    path = GAMEPACKS / spec.game / "sdk_interface.md"
    if not path.is_file():
        return "跳过 sdk_interface.md（文件不存在）"

    support_root = GAMEPACKS / spec.game / spec.support_dir_name()
    lines = [
        GENERATED_BEGIN,
        "",
        "## 候选脚手架与填空契约（自动生成）",
        "",
        f"`candidate_interface` = `{spec.interface}`。候选包里已经放好**官方 SDK**"
        "（与人类选手当年拿到的逐字节相同），协议层不需要你实现。",
        "",
        "**你只需要写 `ai.py`，在其中定义 `class AI`。**",
        "",
    ]
    if spec.tracks:
        lines += [
            "这是**非对称游戏**，两个座位玩的不是同一个游戏，官方也发了两套 SDK。"
            "候选包里按角色轨分成 "
            + "、".join(f"`{spec.sdk_prefix(track)}/`" for track in spec.tracks)
            + "，入口 `main.py` 会读环境变量 `AGENTBENCH_ROLE`（取值 "
            + " / ".join(f"`{track}`" for track in spec.tracks)
            + "）自动选择对应那套，并把角色写到 `self.role` 上。",
            "",
            "**同一份 `ai.py` 要能应付两个角色**——一次 run 里你的代码会被派去打每个角色。"
            "用 `self.role` 分支即可。",
            "",
        ]
    if support_root.is_dir():
        modules = _sdk_module_index(support_root)
        lines.append("### 候选包内容")
        lines.append("")
        lines.append("| 文件 | 说明 |")
        lines.append("|---|---|")
        lines.append("| `main.py` | 入口（生成物）：把你的 `ai.AI` 接到官方 SDK 的驱动点 |")
        lines.append("| `_bootstrap.py` | 公共引导（生成物）：`sys.path`/工作目录与失败诊断 |")
        lines.append(
            "| `ai_example.py` | **格式示例**（生成物）：只演示接口，策略是占位。"
            "另存为 `ai.py` 后再写你自己的策略 |"
        )
        lines.append(
            "| `CANDIDATE_CONTRACT.md` | **先读这个**：该游戏 `AI` 要实现哪些方法、参数是什么 |"
        )
        lines.append(
            "| `selfcheck.py` | **提交前跑它**（生成物）：`python3 selfcheck.py` 检查每个候选"
            "能否导入、接口是否正确、能否被判题器拉起来。不打对局，判定口径与框架侧完全一致 |"
        )
        # 非对称游戏两套 SDK 加起来几十个文件，逐个列反而看不清；按轨折叠成一行。
        folded: dict[str, int] = {}
        for name in modules:
            prefix = name.split("/", 1)[0]
            if spec.tracks and prefix in {spec.sdk_prefix(track) for track in spec.tracks}:
                folded[prefix] = folded.get(prefix, 0) + 1
                continue
            note = "官方 SDK"
            if name == "official_main.py":
                note = "官方入口（协议层，被 `main.py` 复用）"
            elif name == "aquawar_sdk.py":
                note = "**纯 Python 移植的协议层**（非官方原版，见文件头说明）"
            lines.append(f"| `{name}` | {note} |")
        for prefix, count in sorted(folded.items()):
            lines.append(
                f"| `{prefix}/` | 该角色轨的官方 SDK 协议层，{count} 个文件"
                "（`official_main.py` 是官方入口）|"
            )
        lines.append("")
        lines.append(
            "> 容器里**没有**本地对战工具、游戏后端可执行程序、训练脚本，也没有任何"
            "现成的完整策略实现。这是刻意的：对局只能通过 `.agentbench/action.json`"
            "交给评测器完成，你写的策略强度只由评测器回传的 Feedback 定义。"
            "被剔除的清单记在 `SUPPORT_PROVENANCE.json` 的 `container_denied` 里。"
        )
        lines.append("")
    lines.append(GENERATED_END)
    block = "\n".join(lines)

    text = path.read_text(encoding="utf-8")
    if GENERATED_BEGIN in text and GENERATED_END in text:
        head, _, rest = text.partition(GENERATED_BEGIN)
        _, _, tail = rest.partition(GENERATED_END)
        updated = head.rstrip() + "\n\n" + block + tail
    else:
        updated = text.rstrip() + "\n\n" + block + "\n"
    if updated == text:
        return "sdk_interface.md 无需变更"
    if check:
        return "sdk_interface.md 需要更新（候选契约小节漂移）"
    path.write_text(updated, encoding="utf-8")
    return f"sdk_interface.md 已更新（{len(updated)} 字节）"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", help="只处理指定游戏（可重复）")
    parser.add_argument("--check", action="store_true", help="只检查是否与 A 仓官方 SDK 同步")
    args = parser.parse_args(argv)

    games = args.game or sorted(SPECS)
    drift = False
    failures: list[str] = []
    for game in games:
        spec = SPECS.get(game)
        if spec is None:
            print(f"=== {game}\n  没有装配规格（SPECS 里未定义）")
            failures.append(game)
            continue
        print(f"=== {game}  candidate_interface={spec.interface}")
        try:
            changed, detail = assemble(spec, check=args.check)
        except SystemExit as error:
            print(f"  ✗ {error}")
            failures.append(game)
            continue
        label = spec.support_dir_name()
        if detail.startswith("缺少官方 SDK"):
            print(f"  ✗ {label}: {detail}")
            failures.append(game)
            continue
        drift = drift or (changed and args.check)
        print(f"  {'~' if changed else '='} {label}: {detail}")
        print("  " + update_manifest(spec, check=args.check))
        print("  " + update_sdk_interface(spec, check=args.check))

    if failures:
        print("\n未完成：" + ", ".join(failures))
        return 1
    if args.check and drift:
        print("\n检测到漂移：候选脚手架与 A 仓官方 SDK 不一致，重跑本脚本刷新。")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

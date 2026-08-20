#!/usr/bin/env python3
"""从 A 仓生成/刷新所有游戏的 GamePack（引用式，零资源重复）。

用法：
    AGENTBENCH_ROOT=/path/to/AgentBench python scripts/gen_gamepacks.py [--game X ...]
    python scripts/gen_gamepacks.py --check      # 只检查漂移，不写文件

生成原则：
1. **规则 / 决策空间 / 回放字段说明**：一律写成 ``@agentbench:`` 引用，
   唯一权威源是 A；B 不留副本，A 更新后 B 立刻跟着变。
2. **B 独有材料**（回放阅读技能、SDK 协议说明、Goal 章程）：生成到 GamePack 内，
   并记录来源文件的 sha256；A 变更后 ``--check`` 会报漂移，提醒重新生成/人工校对。
3. 已有的人工材料（如 antwar2 手写的 replay_skill / candidate_support）**不覆盖**。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GAMEPACKS = REPO_ROOT / "gamepacks"

PRESERVE = ("replay_skill.md", "sdk_interface.md", "GOAL_CHARTER.md", "candidate_support")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_docstring(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return None
    return ast.get_docstring(tree)


def render_sdk_interface(game: str, arena_doc: str | None, roles: list[str]) -> str:
    protocol = arena_doc or "（该游戏的对战器未提供协议说明，请人工补充。）"
    return f"""# {game} · 选手程序接口（SDK / 协议）

> 本文件由 `scripts/gen_gamepacks.py` 从 A 的 `games/{game}/evaluator/arena.py` 的协议
> 说明生成。协议是**公开**信息（不是任何人的策略），可安全提供给 Goal 阅读。

## 你要交付什么

一个可运行的选手包，入口是 `main.py`：

- 进程启动后进入循环：读一帧输入 → 计算 → 写一帧输出；
- 角色：{roles}；同一份代码要能在任一角色下运行；
- 只能用标准库与包内自带模块；**不联网**、**不写文件**（沙箱会拒绝）；
- 任何格式错误 / 超时 / 崩溃都会被判负（对战器返回 `game_error`），
  所以**第 0 版的首要目标是协议完全正确**，强度其次。

## 对局传输协议（来自 A 的对战器）

{protocol}

## 自检建议

1. 先写一个只做合法动作的最小版本，确认能跑完整局（拿到 `complete` 而不是 `game_error`）；
2. 再逐步加入策略；每次改动都用一局诊断对局验证；
3. 读 `gamepack/replay_skill.md` 学会从回放定位自己的失误。
"""


def render_replay_skill(game: str, replay_format_ref: str) -> str:
    return f"""# {game} · 回放阅读指南

> 字段语义的权威来源是 A 的 `{replay_format_ref}`（已随 GamePack 一起冻结到
> `gamepack/replay_format.md`）。本文件教你**怎么用**这些字段做诊断。

## 三步诊断法

1. **先看终局**：谁赢、分差多少、多少回合结束。分差小 = 战术问题；分差大 = 战略问题。
2. **再看转折点**：沿时间轴找双方分差/资源差变化最快的那几个回合，只精读这几段。
3. **最后看动作**：在转折回合里对比"我的动作"与"对手的动作"，回答三个问题：
   - 我当时的合法动作集里有更好的选项吗（查 `gamepack/decision_space.yaml`）？
   - 对手在做什么模式化的事（重复出现的动作序列）？
   - 我的失误是**规则理解错误**、**评估函数错误**还是**执行顺序错误**？

## 写进经验文档的格式

每条经验写成可验证的形式：

```
假设：<对手在 X 情况下总是 Y>
证据：<回放 request_id / 回合号 / 字段>
改动：<我在策略里做了什么>
结果：<下一轮胜率/分差变化>
```

只保留被证据支持的结论；被推翻的假设要显式标记为"已否证"，避免反复踩坑。
"""


def render_goal_charter(game: str) -> str:
    return f"""# {game} · Goal 章程

## 唯一目标

在 {game} 上刷出 SOTA：相对人类选手池取得尽可能高的 Elo 与胜率。

## 你能看到

- `gamepack/rules.md`：规则（A 的权威版本）
- `gamepack/decision_space.yaml`：决策空间与合法动作语义
- `gamepack/replay_format.md` + `gamepack/replay_skill.md`：回放字段与阅读方法
- `leaderboard.json`：人类排行榜（对手 id / rank / Elo）
- `feedback/<request_id>/`：你自己每轮对局的回放与结果
- `research/`：你自己写的迭代经验
- 你自己历次候选代码（`.agentbench/rollouts/` 与 run 的 `snapshots/`）

## 你看不到

- 任何对手的源码（除非本轮实验显式开启消融）
- 认证/评测矩阵、参考策略、其它 run 的记忆
- 互联网

## 工作循环

0. 第 0 轮：只读规则与协议，写出**格式绝对正确**的裸策略 v000。
1. 每轮产出 k 个有机制差异的候选，写 `.agentbench/action.json` 请求官方对局。
2. 读回放 → 更新经验 → 改策略 → 下一轮。
3. 每次改动都要有回放证据支撑，避免无根据的大改写。
"""


def build_manifest(
    game: str,
    *,
    roles: list[str],
    has_rules: bool,
    has_decision_space: bool,
    has_replay_format: bool,
    has_replay_skill: bool,
    has_candidate_support: bool,
    source_digests: dict[str, str],
    existing: dict[str, object] | None = None,
) -> str:
    """合并生成 manifest：**保留**已有的人工字段，只覆盖生成器管理的字段。"""

    import yaml

    document: dict[str, object] = dict(existing or {})
    document["schema_version"] = "1.1"
    document["game"] = game
    if has_rules:
        document["rules"] = f"@agentbench:games/{game}/rules.md"
    if has_decision_space:
        document["decision_space"] = f"@agentbench:games/{game}/decision_space.yaml"
    if has_replay_format:
        document["replay_format"] = f"@agentbench:games/{game}/replay_format.md"
    # 回放阅读指南属于**游戏语义**，事实源在 A（games/<game>/replay_skill.md）。
    # B 只在 A 没有该游戏指南时才回落到本地存根，避免两仓各存一份、各自漂移。
    if has_replay_skill:
        document["replay_skill"] = f"@agentbench:games/{game}/replay_skill.md"
    else:
        document["replay_skill"] = "replay_skill.md"
    document["sdk_interface"] = "sdk_interface.md"
    document["goal_charter"] = "GOAL_CHARTER.md"
    if has_candidate_support:
        document["candidate_support"] = "candidate_support"
    document["roles"] = list(roles)
    allowed = [
        "frozen_gamepack",
        "public_sdk",
        "public_leaderboard",
        "active_candidate",
        "run_local_public_replays",
        "run_local_experience",
    ]
    forbidden = [
        "internet",
        "human_source",
        "certification_matrix",
        "reference_policy_versions",
        "reference_policy_replays",
        "cross_run_codex_memory",
    ]
    document["learning_isolation"] = {"allowed": allowed, "forbidden": forbidden}
    document["source_digests"] = dict(sorted(source_digests.items()))
    header = (
        "# 引用式 GamePack（由 scripts/gen_gamepacks.py 生成/刷新）。\n"
        "# 规则 / 决策空间 / 回放字段说明的唯一权威源是 A 仓：值以 @agentbench: 开头即引用。\n"
        "# public_leaderboard = 图上的 Human Ranking，仅含 opponent_id/rank/Elo。\n"
        "# source_digests 记录 A 侧来源指纹，`--check` 可检测漂移。\n"
    )
    return header + yaml.safe_dump(document, allow_unicode=True, sort_keys=True)


def game_roles(agentbench_root: Path, game: str) -> list[str]:
    import yaml

    path = agentbench_root / "games" / game / "game.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    roles = document.get("roles") if isinstance(document, dict) else None
    return [str(item) for item in roles] if isinstance(roles, list) and roles else ["P0", "P1"]


def generate(agentbench_root: Path, game: str, *, check: bool) -> dict[str, object]:
    game_dir = agentbench_root / "games" / game
    pack = GAMEPACKS / game
    rules = game_dir / "rules.md"
    decision_space = game_dir / "decision_space.yaml"
    replay_format = game_dir / "replay_format.md"
    replay_skill = game_dir / "replay_skill.md"
    arena = game_dir / "evaluator" / "arena.py"
    roles = game_roles(agentbench_root, game)

    digests: dict[str, str] = {}
    for label, path in (
        ("rules.md", rules),
        ("decision_space.yaml", decision_space),
        ("replay_format.md", replay_format),
        ("replay_skill.md", replay_skill),
        ("evaluator/arena.py", arena),
    ):
        if path.is_file():
            digests[label] = sha256(path)

    report: dict[str, object] = {
        "game": game,
        "roles": roles,
        "has_rules": rules.is_file(),
        "has_decision_space": decision_space.is_file(),
        "has_replay_format": replay_format.is_file(),
        "has_arena_protocol": module_docstring(arena) is not None,
        "written": [],
        "drift": [],
    }

    if check:
        manifest_path = pack / "manifest.yaml"
        if not manifest_path.is_file():
            report["drift"].append("gamepack missing")  # type: ignore[union-attr]
            return report
        import yaml

        document = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        recorded = document.get("source_digests") or {}
        for label, digest in digests.items():
            if recorded.get(label) != digest:
                report["drift"].append(label)  # type: ignore[union-attr]
        return report

    pack.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    skill = pack / "replay_skill.md"
    if replay_skill.is_file():
        # A 已有该游戏的回放阅读指南（事实源）⇒ manifest 走引用，删掉 B 的本地存根，
        # 否则同一份材料两仓各存一份，必然漂移。
        if skill.exists():
            skill.unlink()
            written.append(f"-{skill.name}")
    elif not skill.exists():
        skill.write_text(
            render_replay_skill(game, f"games/{game}/replay_format.md"), encoding="utf-8"
        )
        written.append(skill.name)
    sdk = pack / "sdk_interface.md"
    if not sdk.exists():
        sdk.write_text(
            render_sdk_interface(game, module_docstring(arena), roles), encoding="utf-8"
        )
        written.append(sdk.name)
    charter = pack / "GOAL_CHARTER.md"
    if not charter.exists():
        charter.write_text(render_goal_charter(game), encoding="utf-8")
        written.append(charter.name)

    manifest = pack / "manifest.yaml"
    existing: dict[str, object] = {}
    if manifest.is_file():
        import yaml

        loaded = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    manifest.write_text(
        build_manifest(
            game,
            roles=roles,
            has_rules=rules.is_file(),
            has_decision_space=decision_space.is_file(),
            has_replay_format=replay_format.is_file(),
            has_replay_skill=replay_skill.is_file(),
            has_candidate_support=(pack / "candidate_support").is_dir(),
            source_digests=digests,
            existing=existing,
        ),
        encoding="utf-8",
    )
    written.append(manifest.name)

    # 引用化之后，GamePack 内的规则/决策空间副本就是漂移源，删掉。
    for stale in ("rules.md", "decision_space.yaml", "replay_format.md"):
        target = pack / stale
        if target.is_file() and (game_dir / stale).is_file():
            target.unlink()
            written.append(f"-{stale}")

    report["written"] = written
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", action="append", default=None)
    parser.add_argument("--agentbench-root", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)

    root = arguments.agentbench_root or Path(os.environ.get("AGENTBENCH_ROOT", ""))
    if not root or not (root / "games").is_dir():
        print("error: set AGENTBENCH_ROOT (or --agentbench-root) to the AgentBench repository")
        return 2
    root = root.resolve()
    games = arguments.game or sorted(
        item.name
        for item in (root / "games").iterdir()
        if item.is_dir() and not item.name.startswith("_") and (item / "game.yaml").is_file()
    )
    drifted = 0
    for game in games:
        report = generate(root, game, check=arguments.check)
        drift = report.get("drift") or []
        status = "DRIFT" if drift else ("ok" if not arguments.check else "fresh")
        if drift:
            drifted += 1
        print(
            f"[{status:5}] {game:10} roles={report['roles']} "
            f"rules={report['has_rules']} ds={report['has_decision_space']} "
            f"replay={report['has_replay_format']} protocol={report['has_arena_protocol']} "
            f"{'drift=' + ','.join(drift) if drift else report.get('written') or ''}"
        )
    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

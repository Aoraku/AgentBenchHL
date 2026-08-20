"""B → A 的回放翻译桥。

**A 是唯一事实源**：回放字段的语义是游戏知识，翻译实现只能有一份，放在
``AgentBench/src/agentbench/replay/`` 与 ``games/<game>/evaluator/narrate.py``。
B 这一侧只做两件事：把 A 的实现找出来调用、把结果写进 Feedback 目录。

**这里绝不实现任何游戏语义**。一旦 B 也开始解析回放字段，两仓就会各说各话：
A 改了回放解码、B 不知道，反馈里的数字就悄悄错了 —— 而这种错不会报错，只会让
agent 基于错误证据迭代几十轮。

为什么这层桥值得存在（而不是让 agent 自己在容器里读回放）
--------------------------------------------------------
antwar2 一局裸回放是 6.9 MB 的纯数字。让 agent 自己解析的代价实测是：
一轮 850s 墙钟里约 530s（63%）花在写解析脚本和跑本地验证上，而这些代码每轮都要重写、
每个 run 都要重写、还各写各的（同一份数据被 N 个 agent 用 N 种可能是错的方式解读）。
翻译一次、以自然语言交付，把这 63% 还给"想策略"这件事本身。

A 是零依赖包，也不一定装进 B 的环境，所以按需把 ``<AGENTBENCH_ROOT>/src`` 加进
``sys.path`` 再导入（与 ``application/decision_space.py`` 同一套做法）。
翻译失败一律降级为一段说明文本，**绝不让整轮反馈丢掉** —— 反馈通道断掉比反馈不完整
严重得多。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Feedback 目录里自然语言回放的文件名。
NARRATION_FILENAME = "replay.md"


def _agentbench_root(explicit: str | Path | None) -> Path | None:
    if explicit:
        return Path(explicit).resolve()
    value = os.environ.get("AGENTBENCH_ROOT")
    return Path(value).resolve() if value else None


def _ensure_importable(root: Path) -> None:
    source = root / "src"
    if source.is_dir():
        text = str(source)
        if text not in sys.path:
            sys.path.insert(0, text)


def narrate_case(
    game: str,
    replay_path: str | Path | None,
    destination: Path,
    *,
    match_id: str,
    perspective: str | None = None,
    opponent_id: str = "",
    detail: str = "digest",
    official_winner: str | None = None,
    official_rounds: int | None = None,
    diagnostic: str = "",
    agentbench_root: str | Path | None = None,
) -> tuple[Path | None, str]:
    """把一局回放翻译成自然语言并写到 ``destination``。

    返回 ``(落盘路径 | None, 一句可审计的说明)``。

    ``replay_path`` 允许为 None：候选第 0 回合被判负时根本没有回放，这时 A 侧仍会
    产出一段"为什么没有回放 + 对战器诊断"的说明，那恰恰是这一轮最该读的东西。
    """

    root = _agentbench_root(agentbench_root)
    if root is None:
        return None, "AGENTBENCH_ROOT 未设置，无法调用 A 仓的回放翻译"
    _ensure_importable(root)
    try:
        from agentbench.replay import narrate  # noqa: PLC0415 - 仅在写反馈时需要
    except Exception as error:  # noqa: BLE001 - 导入失败要如实记账，不能静默
        return None, f"导入 A 仓 agentbench.replay 失败：{type(error).__name__}: {error}"

    try:
        narration = narrate(
            game,
            replay_path if replay_path is not None else destination.parent / "__missing__.json",
            match_id=match_id,
            perspective=perspective,
            opponent_id=opponent_id,
            detail=detail,  # type: ignore[arg-type]
            official_winner=official_winner,
            official_rounds=official_rounds,
            diagnostic=diagnostic,
            games_root=root / "games",
        )
    except Exception as error:  # noqa: BLE001 - 翻译失败不能让整轮反馈丢掉
        return None, f"A 仓回放翻译失败：{type(error).__name__}: {error}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(narration.text, encoding="utf-8")
    note = f"A 仓 agentbench.replay 翻译（detail={detail}，{len(narration.text)} 字符）"
    return destination, note

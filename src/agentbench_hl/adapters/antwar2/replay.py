"""Ground official AntWar2 replay numbers in public-state semantics —— 复用 A。

三仓铁律：回放解码语义是仓库 A 的资产。本模块从 A 的
`games/antwar2/evaluator/replay.py` **re-export** `decode_replay` 及相关数据类型
（薄封装），实现全部在 A。B 只在 A 结果之上做 HL 特有加工（回放技能翻译等）。
"""

from __future__ import annotations

from ._agentbench_evaluator import load_a_submodule

_replay = load_a_submodule("replay")

CanonicalFrame = _replay.CanonicalFrame
AtomicEvent = _replay.AtomicEvent
CriticalWindow = _replay.CriticalWindow
StrategicClaim = _replay.StrategicClaim
ReplayReport = _replay.ReplayReport
decode_replay = _replay.decode_replay

__all__ = [
    "AtomicEvent",
    "CanonicalFrame",
    "CriticalWindow",
    "ReplayReport",
    "StrategicClaim",
    "decode_replay",
]

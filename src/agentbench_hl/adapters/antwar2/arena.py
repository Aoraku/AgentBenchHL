"""Length-prefixed AntWar2 match transport —— 复用 A 的对局执行内核。

三仓铁律：对局裁决（长度前缀 stdio 协议、角色感知结果、公开轨迹导出）是仓库 A
的资产。B 不再维护一份独立实现，本模块从 A 的
`games/antwar2/evaluator/arena.py` **re-export** 所需符号（薄封装）。

调用方（factory / run_service）继续使用 B 的 `ports.arena` 的 `ProcessSpec`/
`MatchCase`（与 A 的等价 dataclass 字段一致，鸭子兼容），A 的 arena 返回的结果同样
暴露 `status`/`result`/`score_margin`/`payload`/`rounds` 等契约字段。
"""

from __future__ import annotations

from ._agentbench_evaluator import load_a_submodule

_arena = load_a_submodule("arena")

AntWar2MatchError = _arena.AntWar2MatchError
AntWar2Arena = _arena.AntWar2Arena
Runner = _arena.Runner
run_native_match = _arena.run_native_match
match_result_from_replay = _arena.match_result_from_replay
write_public_trace = _arena.write_public_trace

__all__ = [
    "AntWar2Arena",
    "AntWar2MatchError",
    "Runner",
    "match_result_from_replay",
    "run_native_match",
    "write_public_trace",
]

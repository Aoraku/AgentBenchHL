"""通用（游戏无关）契约适配器 —— 任何有 GamePack 的游戏零代码接入 HL。

组成：
- :mod:`pool`         人类选手池 / 公开排行榜（唯一源 = A 的 manifest.tsv）
- :mod:`match_worker` 隔离沙箱内的单局执行器（只调 A 的 ``evaluate()``）
- :mod:`arena`        B 的 Arena 实现（并行安全、三态忠实）
- :mod:`factory`      装配一次 Goal-led run
"""

from __future__ import annotations

from agentbench_hl.adapters.contract.arena import ContractArena
from agentbench_hl.adapters.contract.factory import (
    ContractAdapterFactory,
    GoalRunBundle,
    build_goal_run,
    game_roles,
)
from agentbench_hl.adapters.contract.pool import (
    PoolPlayer,
    load_pool,
    public_leaderboard,
    ranked_ladder,
    runnable_players,
)

__all__ = [
    "ContractAdapterFactory",
    "ContractArena",
    "GoalRunBundle",
    "PoolPlayer",
    "build_goal_run",
    "game_roles",
    "load_pool",
    "public_leaderboard",
    "ranked_ladder",
    "runnable_players",
]

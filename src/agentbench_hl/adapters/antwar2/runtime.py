"""Frozen AntWar2 resources, builds, and human-pool audit —— 复用 A 的对战器内核。

三仓铁律：游戏语义（编译后端、审计人类池、物化选手）是仓库 A 的资产。B 不再
维护一份独立实现，本模块从 A 的 `games/antwar2/evaluator/runtime.py` **re-export**
所需符号（薄封装），经 `_agentbench_evaluator` 定位 A 仓库（AGENTBENCH_ROOT）。

保留 B 侧稳定的符号名以最小化调用方/测试改动；实现全部在 A。
"""

from __future__ import annotations

from ._agentbench_evaluator import load_a_evaluator

_a = load_a_evaluator()

AntWar2RuntimeError = _a.AntWar2RuntimeError
AntWar2Layout = _a.AntWar2Layout
FrozenBackend = _a.FrozenBackend
Opponent = _a.Opponent
sha256_file = _a.sha256_file
tree_sha256 = _a.tree_sha256
safe_extract = _a.safe_extract
validate_cached_backend = _a.validate_cached_backend
build_backend = _a.build_backend
audit_human_pool = _a.audit_human_pool
materialize_bootstrap = _a.materialize_bootstrap

__all__ = [
    "AntWar2Layout",
    "AntWar2RuntimeError",
    "FrozenBackend",
    "Opponent",
    "audit_human_pool",
    "build_backend",
    "materialize_bootstrap",
    "safe_extract",
    "sha256_file",
    "tree_sha256",
    "validate_cached_backend",
]

"""Locate and load AgentBench (A) 的 AntWar2 对战器内核，供 B 复用。

三仓铁律：游戏语义（编译后端、跑对局、解码回放）是 A 的资产。B 不再维护一份
独立的对局裁决实现，而是通过本模块定位 A 仓库，加载 A 的
`games/antwar2/evaluator/` 包，再由 B 的 adapter 薄封装（re-export）所需符号。

定位 A 的顺序：
1. 环境变量 ``AGENTBENCH_ROOT``（与 config.paths.agentbench_root 同源）；
2. 三仓同级布局下 B 仓库根的兄弟目录 ``../AgentBench``。
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

_EVALUATOR_PACKAGE = "_agentbench_a_antwar2_evaluator"


class AgentBenchEvaluatorNotFound(RuntimeError):
    """无法定位 A 仓库的 AntWar2 对战器内核。"""


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("AGENTBENCH_ROOT")
    if env:
        roots.append(Path(env).expanduser())
    # B 仓库根：src/agentbench_hl/adapters/antwar2/_agentbench_evaluator.py 上溯 5 级。
    repo_root = Path(__file__).resolve().parents[4]
    roots.append(repo_root.parent / "AgentBench")
    return roots


def _resolve_evaluator_dir() -> Path:
    for root in _candidate_roots():
        evaluator_init = root / "games" / "antwar2" / "evaluator" / "__init__.py"
        if evaluator_init.is_file():
            return evaluator_init.parent
    tried = ", ".join(str(root) for root in _candidate_roots())
    raise AgentBenchEvaluatorNotFound(
        "cannot locate AgentBench (A) antwar2 evaluator; set AGENTBENCH_ROOT to the "
        f"AgentBench repository root (tried: {tried})"
    )


@lru_cache(maxsize=1)
def load_a_evaluator() -> ModuleType:
    """加载 A 的 `games/antwar2/evaluator` 包并返回模块（可复用其全部符号）。"""

    evaluator_dir = _resolve_evaluator_dir()
    # A 的 evaluator 依赖 `agentbench.core`（契约），把 A 的 src 加入路径。
    a_src = evaluator_dir.parents[2] / "src"
    if a_src.is_dir() and str(a_src) not in sys.path:
        sys.path.insert(0, str(a_src))

    if _EVALUATOR_PACKAGE in sys.modules:
        return sys.modules[_EVALUATOR_PACKAGE]

    spec = importlib.util.spec_from_file_location(
        _EVALUATOR_PACKAGE,
        evaluator_dir / "__init__.py",
        submodule_search_locations=[str(evaluator_dir)],
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AgentBenchEvaluatorNotFound(f"cannot load evaluator package from {evaluator_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_EVALUATOR_PACKAGE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_EVALUATOR_PACKAGE, None)
        raise
    return module


def load_a_submodule(name: str) -> ModuleType:
    """加载 A 的 evaluator 包下的子模块（如 ``arena`` / ``replay`` / ``runtime``）。"""

    load_a_evaluator()  # 确保包已注册且 A 的 src 在路径上。
    return importlib.import_module(f"{_EVALUATOR_PACKAGE}.{name}")

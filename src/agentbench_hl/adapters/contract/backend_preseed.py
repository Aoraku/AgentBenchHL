"""后端依赖预置 —— 需要联网的准备工作必须在**沙箱外**做一次。

问题
----
A 的部分游戏在跑第一局时会现场 pip 安装后端依赖（例如 LostSpace 需要
``antlr4-python3-runtime==4.9.1``，装到 ``build_root/backend-deps/site``）。
但我们所有对局都跑在**禁网沙箱**里，于是这一步永远失败：

    infra_error: failed to install antlr4-python3-runtime for the LostSpace backend
    (offline?) ...

结果 lostspace 全池 192 人审计 **0 通过**——看起来像"这个游戏的对战器坏了"，
实际上只是准备阶段被断网拦住。

约定
----
若 A 的某游戏 evaluator 模块（或其 ``runtime`` 子模块）暴露下列任一函数：

* ``bootstrap_backend_dependencies(build_root)``
* ``prepare_backend_dependencies(build_root)``

B 就在**沙箱外**（可联网）用同一个 ``build_root`` 先调用一次。这些函数普遍带
stamp 文件跳过逻辑，因此沙箱内再次调用时会命中缓存、不再联网。

这是一个**可选**约定：没有钩子的游戏什么都不做；钩子失败也不阻塞对局
（真正的失败会在对局诊断里如实出现，而不是被这里吞掉）。
"""

from __future__ import annotations

import fcntl
import importlib
import sys
from contextlib import suppress
from pathlib import Path

HOOK_NAMES = ("bootstrap_backend_dependencies", "prepare_backend_dependencies")
STAMP_NAME = ".backend-preseed"


def _hook(agentbench_root: Path, game: str):
    """在 A 的 evaluator 模块里找依赖引导钩子（找不到返回 None）。"""

    source_root = agentbench_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    games_root = agentbench_root / "games"
    if str(games_root) not in sys.path:
        sys.path.insert(0, str(games_root))

    modules = []
    with suppress(ImportError):
        modules.append(importlib.import_module(f"{game}.evaluator.runtime"))
    with suppress(ImportError):
        modules.append(importlib.import_module(f"{game}.evaluator"))
    for module in modules:
        for name in HOOK_NAMES:
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate
    return None


def preseed_backend(agentbench_root: Path, game: str, build_root: Path) -> str | None:
    """沙箱外预置后端依赖。

    Returns:
        ``None`` = 无需预置或已预置；否则返回失败原因（调用方只记录，不终止对局）。
    """

    build_root.mkdir(parents=True, exist_ok=True)
    stamp = build_root / STAMP_NAME
    if stamp.is_file():
        return None
    hook = _hook(Path(agentbench_root), game)
    if hook is None:
        stamp.write_text("no hook\n", encoding="utf-8")
        return None
    # 并行审计里多个 arena 共用同一个 build_root：用 flock 串行化，避免同时 pip。
    lock_path = build_root / ".preseed.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            if stamp.is_file():
                return None
            try:
                hook(build_root)
            except Exception as error:  # noqa: BLE001 - 预置失败不该让对局无法发起
                return f"{type(error).__name__}: {error}"
            stamp.write_text(f"{game}\n", encoding="utf-8")
            return None
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

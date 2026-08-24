"""``runtime.match_timeout_s`` 必须真的传到对战器**内层**。

它以前只杀外层进程
------------------
``ContractArena`` 拿到配置里的 ``match_timeout_s``（默认 1800s）后，只把它用作
worker 子进程的 ``subprocess.run(timeout=...)``；而真正判"这一局打太久"的是各游戏
evaluator 自己的 ``timeout_s`` 参数，worker 的 ``_load_evaluator`` 却**只探测
build_root / artifact_root，从不传 timeout_s**。

于是各游戏一律用自带默认值，实测：

    antwar2 / aquawar / miracle 180s，lostspace 240s，
    antwar / rollman 300s，generals 600s，snakego 900s

配置写 1800 完全没用，内层总是先到。后果正是手册警告过的那种——长局被判成候选
的失败：实测 ``s8k4-miracle`` 有两局 ``match timed out after 180.000s``，
在账本里是 0 回合 + ``result=loss``，读起来像"这个模型不会玩 miracle"。

这组测试锁住两件事：请求里带着 timeout、worker 会把它传给认这个参数的对战器。
"""

from __future__ import annotations

import inspect
import sys
from types import ModuleType
from typing import Any

from agentbench_hl.adapters.contract.arena import MATCH_TIMEOUT_GRACE_S
from agentbench_hl.adapters.contract.match_worker import _load_evaluator


class _Evaluator:
    """签名与真实对战器一致的替身。"""

    def __init__(
        self,
        game_dir: Any,
        *,
        build_root: Any = None,
        artifact_root: Any = None,
        timeout_s: float = 180.0,
    ) -> None:
        self.game_dir = game_dir
        self.build_root = build_root
        self.artifact_root = artifact_root
        self.timeout_s = timeout_s


class _EvaluatorWithoutTimeout:
    """有些对战器没有 timeout_s（不能因此报错）。"""

    def __init__(self, game_dir: Any, *, artifact_root: Any = None) -> None:
        self.game_dir = game_dir
        self.artifact_root = artifact_root


def _load(monkeypatch, factory, timeout_s, tmp_path):
    """绕开真实注册表，只测参数装配这一段。"""

    class _Plugin:
        evaluator_factory = staticmethod(lambda game_dir: factory(game_dir))

    registry = ModuleType("agentbench.core.registry")
    registry.get_plugin = lambda game, games_root: _Plugin()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "agentbench.core.registry", registry)
    return _load_evaluator(
        tmp_path, "miracle", tmp_path / "build", tmp_path / "art", timeout_s
    )


def test_configured_timeout_reaches_the_evaluator(monkeypatch, tmp_path) -> None:
    """核心断言：配置的 1800s 必须落到对战器的 timeout_s 上，而不是各游戏的 180s。"""

    evaluator = _load(monkeypatch, _Evaluator, 1800.0, tmp_path)
    assert evaluator.timeout_s == 1800.0, "配置的单局上限没传进去 —— 内层还会按 180s 杀局"


def test_missing_timeout_keeps_the_game_default(monkeypatch, tmp_path) -> None:
    """老的 request.json 没有 timeout_s，重放时要保持各游戏默认，不能崩。"""

    evaluator = _load(monkeypatch, _Evaluator, None, tmp_path)
    assert evaluator.timeout_s == 180.0


def test_evaluator_without_timeout_parameter_still_loads(monkeypatch, tmp_path) -> None:
    """不认 timeout_s 的对战器不能因为多传参数而失败。"""

    evaluator = _load(monkeypatch, _EvaluatorWithoutTimeout, 1800.0, tmp_path)
    assert isinstance(evaluator, _EvaluatorWithoutTimeout)
    assert evaluator.artifact_root is not None


def test_outer_subprocess_timeout_leaves_room_for_the_inner_one() -> None:
    """外层必须严格宽于内层，否则拿不到内层那份可定位的诊断。

    内层超时会报"哪个进程还活着、返回码多少"（lostspace 的死锁就是这么定位的）；
    外层超时只会报一句"exceeded Ns wall limit"，什么都指不出来。
    """

    assert MATCH_TIMEOUT_GRACE_S > 0
    # 余量至少要能容纳 C++ 选手的编译上限（compile_cpp_package 的 300s）。
    assert MATCH_TIMEOUT_GRACE_S >= 300.0


def test_load_evaluator_accepts_timeout_argument() -> None:
    """签名回归：漏掉这个参数就是上面那个静默故障。"""

    assert "timeout_s" in inspect.signature(_load_evaluator).parameters

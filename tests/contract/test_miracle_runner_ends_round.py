"""miracle 候选入口必须保证**每个回合都收尾**，否则整局作废。

背景
----
miracle 的一个回合是"若干操作 + 一条 ``endround``"，后端在 ``endround`` 到达前
一直阻塞读 stdin。而这份入口原来只是::

    while True:
        agent.update_game_info()
        agent.play()

也就是把"一定要收尾"整个压在 LLM 写的 ``play()`` 上。``play()`` 里任何一条提前
``return``（没好棋 / 条件不满足 / 异常分支）都会让对局**永久挂住**。

代价不是"这一局输了"，而是**这一局什么信息都没有**：卡到超时 → 0 回合 →
``result=loss`` / ``score_margin=0`` / ``evaluator_status=game_error``，
没有回放、没有分差梯度，agent 下一轮根本不知道错在哪。
实测 ``s8k4-miracle`` 的 ``v001_holylight_press`` **两个座次都**
``match timed out after 180.000s``。

实测对照（同一个"忘记收尾"的候选，同一个对手）：

* 修复前：126.9s，``GAME_ERROR: player 0 timed out``
* 修复后：7.0s，``COMPLETE``

8 个游戏里只有 miracle 把收尾交给候选（lostspace 的官方 SDK 自己会收尾），
所以只有它需要这层安全网。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

RUNNER = (
    Path(__file__).resolve().parents[2] / "gamepacks/_shared/candidate_runners/miracle.py"
)


class _AiClient:
    """官方基类的替身：只需要 end_round 这一个协议动作。"""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def end_round(self) -> None:
        self.sent.append("endround")

    def choose_cards(self) -> None:
        return None

    def update_game_info(self) -> None:
        return None

    def play(self) -> None:
        return None


def _run(ai_class: type, turns: int) -> tuple[Any, list[str]]:
    """把入口跑 ``turns`` 个回合，然后用 EOFError 收尾（= 对局结束）。"""

    logs: list[str] = []
    agent_box: list[Any] = []
    remaining = {"turns": turns}

    def construct(cls: type) -> Any:
        agent = cls()
        original_update = agent.update_game_info

        def update() -> None:
            if remaining["turns"] <= 0:
                raise EOFError
            remaining["turns"] -= 1
            original_update()

        agent.update_game_info = update  # type: ignore[method-assign]
        agent_box.append(agent)
        return agent

    bootstrap = ModuleType("_bootstrap")
    bootstrap.install_path = lambda: None  # type: ignore[attr-defined]
    bootstrap.load_ai_class = lambda expected: ai_class  # type: ignore[attr-defined]
    bootstrap.construct = construct  # type: ignore[attr-defined]
    bootstrap.guard = lambda fn, what: fn()  # type: ignore[attr-defined]
    bootstrap.log = logs.append  # type: ignore[attr-defined]

    ai_client = ModuleType("ai_client")
    ai_client.AiClient = _AiClient  # type: ignore[attr-defined]
    # 入口会给官方的 read_opt 打 EOF 补丁（官方版读到 EOF 会 int('') 抛 ValueError，
    # 被当成"候选抛异常"而不是"对局结束"），所以桩里要有它依赖的这几个属性。
    ai_client.read_opt = lambda: {}  # type: ignore[attr-defined]
    ai_client.sys = sys  # type: ignore[attr-defined]
    ai_client.json = __import__("json")  # type: ignore[attr-defined]

    saved = {name: sys.modules.get(name) for name in ("_bootstrap", "ai_client")}
    sys.modules["_bootstrap"] = bootstrap
    sys.modules["ai_client"] = ai_client
    try:
        spec = importlib.util.spec_from_file_location("miracle_runner_under_test", RUNNER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        assert module.main() == 0
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return agent_box[0], logs


def test_forgotten_end_round_is_supplied() -> None:
    """``play()`` 不收尾时，入口必须替它补上——否则后端永久阻塞。"""

    class Forgetful(_AiClient):
        def play(self) -> None:
            return  # 就是这条提前 return 让对局挂死

    agent, logs = _run(Forgetful, turns=3)
    assert agent.sent == ["endround"] * 3, "每个回合都必须有且只有一条 endround"
    assert any("没有调用 end_round" in item for item in logs), "补了就必须在 stderr 点名"


def test_explicit_end_round_is_not_duplicated() -> None:
    """``play()`` 自己收尾时不能再补一条。

    多发的那条会被后端当成**下一回合**的操作读掉，那一回合凭空被结束——
    静默且比死锁更难查。
    """

    class Correct(_AiClient):
        def play(self) -> None:
            self.end_round()

    agent, logs = _run(Correct, turns=3)
    assert agent.sent == ["endround"] * 3
    assert not any("没有调用 end_round" in item for item in logs)


def test_double_end_round_in_one_turn_is_blocked() -> None:
    """一个回合里调两次也只发一条（同样是"被当成下一回合操作"的隐患）。"""

    class Twice(_AiClient):
        def play(self) -> None:
            self.end_round()
            self.end_round()

    agent, logs = _run(Twice, turns=2)
    assert agent.sent == ["endround"] * 2
    assert any("重复调用" in item for item in logs)


def test_runner_is_in_sync_with_the_gamepack_copy() -> None:
    """gamepacks/miracle 下的副本必须与共享模板一致，否则修了也不生效。"""

    copy = Path(__file__).resolve().parents[2] / "gamepacks/miracle/candidate_support/main.py"
    if not copy.is_file():  # pragma: no cover - 生成物缺失时跳过
        pytest.skip("gamepack 副本不存在")
    assert copy.read_text(encoding="utf-8") == RUNNER.read_text(encoding="utf-8")

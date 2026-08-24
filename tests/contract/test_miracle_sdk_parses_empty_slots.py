"""miracle 候选 SDK 必须能解析**空生物槽**，否则任何候选都会当场崩。

背景
----
后端在生物槽为空时，``players[i][3]`` 里会出现一个空条目 ``[]``。而官方
``gameunit.py`` 的守卫只写了 ``if cc_list is None``——挡住了"参数没给"，
没挡住"给了个空列表"，于是下一行::

    self.type = UNIT_TYPE[cc_list[0]]
    → IndexError: list index out of range

崩的位置是 ``update_game_info()`` **解析局面**这一步，所以**与候选写的策略无关**：
任何候选遇到同一局面都会崩。候选进程当场退出（runner 记退出码 20），
对局记成 0 回合 + ``result=loss``。

实测 fix3-miracle：``v000_air_dominance`` 两个座次都是这样死的，
是那 16 局里唯一的 2 处失败；其余 14 局正常（回合数 19~80、分差真实）。

作者的本意从 ``None`` 分支就能看出来——空就当默认值；这里把判据放宽到"空即默认"。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SDK_DIR = (
    Path(__file__).resolve().parents[2] / "gamepacks/miracle/candidate_support"
)


def _load(name: str) -> ModuleType:
    path = SDK_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"miracle_{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # gameunit 内部按名字 import 同目录模块，需要把目录放进 path。
    if str(SDK_DIR) not in sys.path:
        sys.path.insert(0, str(SDK_DIR))
    spec.loader.exec_module(module)
    return module


gameunit = _load("gameunit")


@pytest.mark.parametrize("empty", [[], None])
def test_empty_creature_slot_falls_back_to_default(empty) -> None:
    """空槽必须按默认值解析，而不是 IndexError。"""

    capacity = gameunit.CreatureCapacity(empty)
    assert capacity.available_count == 0
    assert capacity.cool_down_list == []


def test_normal_slot_still_parses() -> None:
    """放宽判据不能影响正常数据。"""

    capacity = gameunit.CreatureCapacity([1, 4, [2]])
    assert capacity.available_count == 4
    assert capacity.cool_down_list == [2]


def test_player_state_with_an_empty_slot_parses() -> None:
    """整份局面里混着空槽时也要能解析——这正是线上崩的那种局面。"""

    player = gameunit.Player(
        0,
        [
            [[0, 2, 8, 6, 0, 0, 0, [-1, -1, -1]]],  # 神器
            6,  # 当前法力
            6,  # 最大法力
            [[0, 0, [3]], [], [3, 0, []]],  # ← 中间是空槽
            [],  # 最新召唤
        ],
    )
    assert len(player.creature_capacity) == 3
    assert player.creature_capacity[1].available_count == 0

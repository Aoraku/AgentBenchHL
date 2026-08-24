"""lostspace 候选 SDK 的回合应答必须与后端"是否等待回复"逐位对齐。

为什么这组测试值得单独存在
--------------------------
lostspace 后端 ``GameController.turn()`` 只对两种状态阻塞等待玩家回复::

    if status == PlayerStatus.Alive.value or status == PlayerStatus.WaitForEsacape.value:
        self.inround()          # ← 阻塞

后端 ``player.py`` 的状态定义是 5 个：``0活着，1挂了，2逃了，3下一回合跳过，
4逃生等待中``。而 SDK 的 ``STATUS`` 枚举**只写到 3**，回复条件是 ``{0, 2}``。
两处错位各自造成一类故障：

* **状态 4（逃生等待中）**：后端在等、SDK 不回 → **死锁**。对局卡到超时，
  账本上记成 0 回合 / result=loss / evaluator_status=game_error，而监控只报得出
  "0 回合对局占比高 —— 候选大概率协议格式错"，指向完全无辜的候选。
  实测 ``s8k4-lostspace`` 16 局废了 12 局；候选一旦摸到逃生舱就进状态 4，
  所以几乎必然触发。修好后同一局 11 秒打完（COMPLETE）。
* **状态 2（已逃离）**：后端不等、SDK 却发 finish → 这条消息留在后端 stdin 里，
  等下一次轮到自己时被当成**本回合的行动**读掉，回合被凭空结束。
  这条是静默的，比死锁更难查。

所以这里锁的不是"某个分支写对了"，而是**"该回复的时候一定回复、不该回复的时候
一定不回复"** 这个不变式。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SDK_PATH = (
    Path(__file__).resolve().parents[2]
    / "gamepacks/lostspace/candidate_support/lostspace_sdk.py"
)

#: 后端会阻塞等待回复的状态（GameController.turn 里的判断）。
BACKEND_WAITS = {0, 4}
#: 后端 ``src/config.py`` 的 ``PlayerStatus`` 全部取值（已在 saiblo 上核对原版）：
#: 0 活着 / 1 挂了 / 2 逃了 / 3 下一回合跳过 / 4 逃生等待中 / 5 Error。
#: 官方 SDK 只声明了 0~3 —— 官方 SDK 与官方后端本身就不一致。
ALL_STATUSES = (0, 1, 2, 3, 4, 5)


def _load_sdk() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lostspace_sdk_under_test", SDK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sdk = _load_sdk()


def _roundbegin(status: int) -> dict:
    """一条最小但字段齐全的 roundbegin 消息（``start_turn`` 会逐个字段读）。"""

    return {
        "type": "roundbegin",
        "state": 7,
        "inturn": 0,
        "status": status,
        "hp": 200,
        "keys": [0],
        "tools": {"LandMine": [0, 0], "Sticky": [0, 0], "Kit": 0, "Transport": 0},
        "others": [
            {"player_id": index, "status": 0, "keys": [], "hp": 200} for index in (1, 2, 3)
        ],
        "view": [],
    }


def _client(status: int) -> tuple[object, list[dict], list[int]]:
    client = sdk.AIClient()
    client.player.id = 0
    sent: list[dict] = []
    played: list[int] = []
    client.send_opt = lambda payload: sent.append(payload)  # type: ignore[method-assign]
    client.play = lambda: played.append(1)  # type: ignore[method-assign]
    client.update_view = lambda view: None  # type: ignore[method-assign]
    client.root = _roundbegin(status)
    return client, sent, played


def test_status_enum_covers_every_backend_state() -> None:
    """枚举必须覆盖后端全部 5 个状态，少一个就会漏掉一条分支。"""

    values = {member.value for member in sdk.STATUS}
    assert set(ALL_STATUSES) <= values, f"缺状态：{set(ALL_STATUSES) - values}"


@pytest.mark.parametrize("status", sorted(BACKEND_WAITS))
def test_replies_whenever_the_backend_blocks(status: int) -> None:
    """后端在等的状态（0 活着 / 4 逃生等待中）必须回一条 finish，否则死锁。"""

    client, sent, _ = _client(status)
    client.start_turn()
    assert sent == [{"type": "finish"}], f"status={status} 没有结束回合 → 后端会一直等"


@pytest.mark.parametrize("status", [s for s in ALL_STATUSES if s not in BACKEND_WAITS])
def test_stays_silent_when_the_backend_does_not_wait(status: int) -> None:
    """后端不等的状态（1 挂了 / 2 逃了 / 3 跳过）不能发消息。

    多发的 finish 会滞留在后端 stdin，下次轮到自己时被当成行动读掉。
    """

    client, sent, _ = _client(status)
    client.start_turn()
    assert sent == [], f"status={status} 多发了 {sent} —— 会被当成下一回合的行动"


def test_only_acts_while_alive() -> None:
    """只有活着才能行动；逃生等待中要结束回合但**不能**行动。"""

    alive, _, played_alive = _client(0)
    alive.start_turn()
    assert played_alive == [1]

    waiting, sent, played_waiting = _client(4)
    waiting.start_turn()
    assert played_waiting == [], "逃生等待中不能行动"
    assert sent == [{"type": "finish"}]


def _node(sdk_module, pos: list[int]):
    node = sdk_module.Node()
    node.pos = pos
    node.interprops = []
    return node


def test_dying_with_a_box_does_not_crash_the_candidate() -> None:
    """带箱子死亡时把箱子记到当前格——原来这里索引了两次，必定 TypeError。

    原代码::

        node_index = self.view.nodes[self.pos_to_node_index(...)]   # 已经是 Node
        self.view.nodes[node_index].interprops.append("Box")        # 又拿 Node 当下标

    候选进程当场退出（runner 记退出码 20），后端还在等它这一回合的回复，
    于是对局卡到超时、记 0 回合 / result=loss。**与候选策略无关**：任何候选
    带着箱子死一次就会崩。实测 fix3-lostspace 16 局里 5 局是这么废的。
    """

    client = sdk.AIClient()
    here = _node(sdk, [1, 2, 3])
    client.view.nodes = [here]
    client.player.pos = [1, 2, 3]
    client.player.spawn_pos = [0, 0, 1]
    client.root = {"type": "death", "box": True}

    client.in_turn()

    assert here.interprops == ["Box"]
    assert client.player.status == sdk.STATUS.DEAD


def test_box_is_not_misfiled_when_the_tile_is_out_of_view() -> None:
    """``pos_to_node_index`` 找不到时返回 -1，不能拿 -1 去写最后一个格子。

    那样不会崩，只会把箱子记在错误的格子上——错误的局面比崩溃更难查。
    """

    client = sdk.AIClient()
    elsewhere = _node(sdk, [9, 9, 9])
    client.view.nodes = [elsewhere]
    client.player.pos = [1, 2, 3]
    client.player.spawn_pos = [0, 0, 1]
    client.root = {"type": "death", "box": True}

    client.in_turn()

    assert elsewhere.interprops == [], "位置不在视野里时不能凭空记一个箱子"


def test_not_my_turn_is_a_no_op() -> None:
    """别人的回合只更新轮次，不能回复（回复会污染对方的回合）。"""

    client, sent, played = _client(0)
    client.root["inturn"] = 2
    client.start_turn()
    assert sent == []
    assert played == []

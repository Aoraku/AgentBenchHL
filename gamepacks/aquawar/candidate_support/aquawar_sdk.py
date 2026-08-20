"""aquawar 选手侧 SDK —— **纯 Python 移植**（不是官方原版，这点必须说清）。

为什么是移植而不是 vendor
-------------------------
其余 7 个游戏的候选脚手架都直接 vendor 官方 Python SDK。aquawar 不行：
它的"官方 Python SDK"（``AI_SDK/Python/sdk/``）其实是一个 **pybind11 C++ 扩展**
（``py_ai_sdk.cpp`` + ``ai_client.pyi`` 类型存根 + ``CMakeLists.txt``，
里面 ``pybind11_DIR`` 还是个占位符要选手自己填）。这也正是 aquawar 池里
**194 个审计通过的选手全是 C++、一个 Python 选手都没有**的原因。

在禁网沙箱里为每个候选 cmake+pybind11 编译一次既脆弱又浪费机时，
所以这里把**协议层**用纯 Python 重写一遍。移植范围严格限定为线协议与消息装配：

* 逐字节对齐官方 ``ai_client.hpp`` 的 ``listen`` / ``sendLen`` / ``sendrecv_msg``
  与 ``run`` 状态机（``Action_Pick`` / ``Action_Assert`` / ``Action_Action`` /
  ``Action_Finish``）；
* **不**移植官方那一大堆便利函数（``get_lowest_health_enemy`` 等）。局面以官方
  ``GameInfo`` 的**原始字段**呈现（``EnemyFish`` / ``EnemyHP`` / ``MyFish`` /
  ``MyHP`` / ``MyATK``），信息量不打折，也不会引入"翻译层错误"。

线协议（来自官方 ``ai_client.hpp``）
------------------------------------
* **读**：判题器发来的是 JSON，**没有长度前缀**。官方做法是逐字符累积、
  每次尝试解析，能解析成功就算读完一帧——这里照抄这个语义（``_read_frame``）。
* **写**：``[4 字节大端长度][JSON]``，且 JSON 里的 ``\\n`` / ``\\r``
  **必须先删掉**（官方 ``sendrecv_msg`` 就是这么做的）。

消息格式
--------
判题器 → 选手，按 ``Action`` 分派：

===========  ================================================
``Action``   随帧字段
===========  ================================================
``Pick``     ``RemainFishs``, ``FirstMover``
``Assert``   ``GameInfo``, ``EnemyAction``, ``MyAction``, ``EnemyAssert``
``Action``   ``GameInfo``, ``AssertReply``
``Finish``   ``Result``（``"Win"``/其它），可能带上述若干字段
===========  ================================================

选手 → 判题器：

* Pick：``{"Action":"Pick","ChooseFishs":[...]}``，若选了 >12 的编号 ``x``
  表示**拟态**，则追加 ``12`` 并附 ``"ImitateFish": x-12``；
* Assert：``{"Action":"Assert","Pos":p,"ID":i}``；不断言或位置非法时必须发
  ``{"Action":"Null"}``；
* Action：``{"Action":"Action","Type":t,"MyPos":p,...}``。``Type==0`` 普通攻击带
  ``EnemyPos``；``Type==1`` 技能带 ``EnemyList`` / ``MyList``
  （``enemy_target == -2`` 表示 AOE，展开为**所有存活敌方**）；
* Finish：``{"Action":"Finish"}``。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

FISH_COUNT = 4
IMITATE_BASE = 12


@dataclass
class Fish:
    """一条鱼。``id == -1`` 表示身份未知（敌方未被断言出来时就是这样）。"""

    id: int = -1
    hp: int = 0
    atk: int = 0

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class Action:
    """一次行动。

    :param type: ``0`` 普通攻击 / ``1`` 主动技能
    :param my_pos: 出手的己方鱼位置（0..3）
    :param enemy_target: 敌方目标位置；``-2`` = AOE（展开为所有存活敌方），``-1`` = 无
    :param friend_target: 友方目标位置；``-1`` = 无
    """

    type: int = 0
    my_pos: int = 0
    enemy_target: int = -1
    friend_target: int = -1


@dataclass
class Game:
    """交给 AI 的局面快照。

    ``raw`` 是判题器这一帧的原始 JSON，官方字段一个不少；上面几个列表是把
    ``GameInfo`` 拆好的常用视图，避免每次都自己索引。
    """

    my_fish: list[Fish] = field(default_factory=lambda: [Fish() for _ in range(FISH_COUNT)])
    enemy_fish: list[Fish] = field(default_factory=lambda: [Fish() for _ in range(FISH_COUNT)])
    remain_fish: list[int] = field(default_factory=list)
    first_mover: int = -1
    current_turn: int = 0
    raw: dict = field(default_factory=dict)

    def living_enemies(self) -> list[int]:
        return [index for index, fish in enumerate(self.enemy_fish) if fish.alive]

    def living_allies(self) -> list[int]:
        return [index for index, fish in enumerate(self.my_fish) if fish.alive]


def log(message: str) -> None:
    """诊断只能写 stderr —— stdout 是协议通道。"""

    print(message, file=sys.stderr, flush=True)


class AIClient:
    """选手基类：继承它并实现 ``pick`` / ``assert_fish`` / ``act``。"""

    def __init__(self) -> None:
        self.game = Game()
        self._stdin = sys.stdin.buffer
        self._stdout = sys.stdout.buffer

    # ---------------------------------------------------------------- 线协议
    def _read_frame(self) -> dict | None:
        """读一帧 JSON。

        判题器**不发长度前缀**，所以只能像官方那样逐字节累积、每次试着解析。
        流结束返回 ``None``（对局正常终止，不是错误）。
        """

        buffer = bytearray()
        while True:
            chunk = self._stdin.read(1)
            if not chunk:
                return None
            buffer += chunk
            # 只在可能闭合的位置尝试解析，省掉绝大部分无谓的 json.loads
            if chunk not in b"}]":
                continue
            try:
                return json.loads(buffer.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue

    def _send(self, operation: dict) -> None:
        """写 ``[4 字节大端长度][JSON]``；JSON 内不能有换行（官方约定）。"""

        text = json.dumps(operation, separators=(",", ":"), ensure_ascii=False)
        text = text.replace("\n", "").replace("\r", "")
        body = text.encode("utf-8")
        self._stdout.write(len(body).to_bytes(4, "big", signed=False))
        self._stdout.write(body)
        self._stdout.flush()

    # ---------------------------------------------------------------- 状态维护
    def _parse_game_info(self, info: object) -> None:
        if not isinstance(info, dict) or not info:
            return
        enemy_ids = info.get("EnemyFish") or []
        enemy_hp = info.get("EnemyHP") or []
        my_ids = info.get("MyFish") or []
        my_hp = info.get("MyHP") or []
        my_atk = info.get("MyATK") or []
        for index in range(FISH_COUNT):
            self.game.enemy_fish[index] = Fish(
                id=int(enemy_ids[index]) if index < len(enemy_ids) else -1,
                hp=int(enemy_hp[index]) if index < len(enemy_hp) else 0,
                atk=-1,
            )
            self.game.my_fish[index] = Fish(
                id=int(my_ids[index]) if index < len(my_ids) else -1,
                hp=int(my_hp[index]) if index < len(my_hp) else 0,
                atk=int(my_atk[index]) if index < len(my_atk) else 0,
            )

    # ---------------------------------------------------------------- 四个阶段
    def _do_pick(self, frame: dict) -> dict:
        self.game.current_turn += 1
        self.game.remain_fish = [int(x) for x in (frame.get("RemainFishs") or [])]
        self.game.first_mover = int(frame.get("FirstMover", -1))
        chosen = list(self.pick(self.game) or [])
        operation: dict = {"Action": "Pick", "ChooseFishs": []}
        for value in chosen:
            value = int(value)
            if value > IMITATE_BASE:
                # >12 表示拟态：上场的是 12 号，另外用 ImitateFish 指明模仿谁
                operation["ImitateFish"] = value - IMITATE_BASE
                operation["ChooseFishs"].append(IMITATE_BASE)
            else:
                operation["ChooseFishs"].append(value)
        return operation

    def _do_assert(self, frame: dict) -> dict:
        self._parse_game_info(frame.get("GameInfo"))
        position, fish_id = self.assert_fish(self.game)
        position = int(position)
        # 官方判据：位置越界、或该位置身份**已经**知道了，就必须发 Null。
        if position < 0 or position >= FISH_COUNT or self.game.enemy_fish[position].id != -1:
            return {"Action": "Null"}
        return {"Action": "Assert", "Pos": position, "ID": int(fish_id)}

    def _do_action(self, frame: dict) -> dict:
        self._parse_game_info(frame.get("GameInfo"))
        action = self.act(self.game)
        if not isinstance(action, Action):
            raise TypeError(f"AI.act() 必须返回 Action，实际返回 {type(action).__name__}")
        operation: dict = {
            "Action": "Action",
            "Type": int(action.type),
            "MyPos": int(action.my_pos),
        }
        if action.type == 0:
            operation["EnemyPos"] = int(action.enemy_target)
            return operation
        if action.enemy_target == -2:
            operation["EnemyList"] = self.game.living_enemies()
        elif action.enemy_target != -1:
            operation["EnemyList"] = [int(action.enemy_target)]
        else:
            operation["EnemyList"] = []
        operation["MyList"] = (
            [] if action.friend_target == -1 else [int(action.friend_target)]
        )
        return operation

    def _do_finish(self, frame: dict) -> dict:
        for key in ("GameInfo",):
            if key in frame:
                self._parse_game_info(frame[key])
        return {"Action": "Finish"}

    # ---------------------------------------------------------------- 主循环
    def run(self) -> None:
        """回合制循环（对齐官方 ``AIClient::run``）。"""

        handlers = {
            "Pick": self._do_pick,
            "Assert": self._do_assert,
            "Action": self._do_action,
            "Finish": self._do_finish,
        }
        while True:
            frame = self._read_frame()
            if frame is None:
                log("[aquawar_sdk] 对局结束（stdin 关闭）")
                return
            self.game.raw = frame
            kind = str(frame.get("Action", ""))
            handler = handlers.get(kind)
            if handler is None:
                log(f"[aquawar_sdk] 未知帧类型 {kind!r}，按 Finish 处理")
                handler = self._do_finish
            self._send(handler(frame))

    # ---------------------------------------------------------------- 填空点
    def pick(self, game: Game) -> list[int]:
        """选 4 条鱼上场；编号 >12 表示拟态（``12 + 被模仿者编号``）。"""

        raise NotImplementedError

    def assert_fish(self, game: Game) -> tuple[int, int]:
        """断言敌方某个位置是哪条鱼；返回 ``(-1, -1)`` 表示不断言。"""

        return (-1, -1)

    def act(self, game: Game) -> Action:
        """本回合行动。"""

        raise NotImplementedError

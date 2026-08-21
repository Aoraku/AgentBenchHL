"""Miracle（神迹）的候选侧决策探针。

它做什么
--------
从二进制回放的事件流**重建每回合的局面**，把局面喂给候选的决策逻辑，
记录它实际发出的操作、以及当时的合法操作集（支持集）。上层
``application/support_probe.py`` 用这些数据算精确的 behavioral IG。

与 antwar / antwar2 / rollman / generals 的同构关系
--------------------------------------------------
遵循同一个契约：``--candidate/--replay/--match-id/--role`` 进，stdout 上一行
``AGENTBENCH_POLICY_TRACE=<json>`` 出，json 含 ``decisions[]``，每项有
``state_id`` / ``actions`` / ``legal_supports`` / ``occupancy_id``。

Miracle 的特殊之处（决定了本文件的写法）
----------------------------------------
1. **回放是二进制 int32 事件流，不含状态快照**。前面几个游戏都有现成的完整
   局面可用（rollman 每帧带棋盘、generals 首帧全图 + 增量），miracle 只有
   ``[round, event, payload...]`` 每 7 个整数一条的事件序列。

   所以状态必须**从事件流重演**：Spawn 建单位、Move 改位置、Damage 扣血、
   Death 移除、Heal 回血、BuffAdd/Remove 改圣盾。这可行的前提是事件带足了
   信息，实测确实带足了（见后端 ``main.py::get_media_info``）：
     * ``Spawn  -> [type + 10*camp, level, posX, posY, id]``
     * ``Move   -> [id, destX, destY]``
     * ``Damage -> [targetId, sourceId, damage, damageType]``   ← 带**具体数值**
     * ``Heal   -> [targetId, sourceId, heal]``
   伤害/治疗都是显式数值而不是靠公式反推，所以重演是无损的。

2. **坐标是立方体坐标的前两维**。回放只写 (x, y)，第三维靠 ``z = -x - y``
   还原（六边形立方体坐标的恒等式）。SDK 的 ``reachable`` / ``cube_distance``
   都要三维元组。

3. **单位类型编码有两套顺序，必须用后端那一套**。
     * 后端 ``creature_names`` = ["", Swordsman, Archer, BlackBat, Priest,
       VolcanoDragon, FrostDragon, Inferno]
     * SDK ``UNIT_TYPE``       = [Archer, Swordsman, BlackBat, Priest,
       VolcanoDragon, Inferno]
   两者 Swordsman/Archer 互换，且 SDK 少了 FrostDragon。回放是后端写的，
   所以解码只能用后端顺序——用错会把剑士当弓箭手，属性全错。

4. **属性从 ``Data.json`` 按 (类型, 等级) 查表**。事件只给类型和等级，
   攻击/生命/射程/行动力要查表补齐，否则 ``reachable`` 算不出可达集。

5. **回合归属靠 TurnStart**。``[round, 1, camp]``：camp 就是该回合行动方。
   只在属于本方的回合上探测决策。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: 后端写回放时用的生物名顺序（logic/main.py::get_media_info 的 creature_names）。
#: ⚠️ 与 SDK 的 UNIT_TYPE 顺序**不同**，见模块头注释第 3 条。
BACKEND_CREATURES = (
    "",
    "Swordsman",
    "Archer",
    "BlackBat",
    "Priest",
    "VolcanoDragon",
    "FrostDragon",
    "Inferno",
)

#: 后端写回放时用的神器名顺序。
BACKEND_ARTIFACTS = ("", "HolyLight", "SalamanderShield", "InfernoFlame", "WindBlessing")

#: 事件码（EVENT_NAMES 的下标）。
EVENT_TURN_START = 1
EVENT_TURN_END = 2
EVENT_SPAWN = 3
EVENT_MOVE = 4
EVENT_ATTACK = 5
EVENT_DAMAGE = 6
EVENT_DEATH = 7
EVENT_HEAL = 8
EVENT_ACTIVATE_ARTIFACT = 9
EVENT_GAME_END = 10
EVENT_GAME_START = 11
EVENT_BUFF_ADD = 12
EVENT_BUFF_REMOVE = 13
EVENT_SUMMON = 18

#: 神迹（本体）初始血量，用于胜负与状态指纹。
MIRACLE_MAX_HP = 30


@dataclass
class ProbeUnit:
    """重演出来的单位。字段与 SDK ``gameunit.Unit`` 对齐，供 calculator 使用。"""

    id: int
    camp: int
    type: str
    level: int
    pos: tuple[int, int, int]
    atk: int = 0
    max_hp: int = 0
    hp: int = 0
    atk_range: tuple[int, int] = (1, 1)
    max_move: int = 0
    cool_down: int = 0
    cost: int = 0
    flying: bool = False
    atk_flying: bool = False
    agility: bool = False
    holy_shield: bool = False
    can_atk: bool = True
    can_move: bool = True


@dataclass
class ProbeState:
    """重演出来的局面。只含公开可见信息。"""

    round: int = 0
    units: dict[int, ProbeUnit] = field(default_factory=dict)
    miracle_hp: list[int] = field(default_factory=lambda: [MIRACLE_MAX_HP, MIRACLE_MAX_HP])
    #: camp -> 已知的手牌（GameStart 事件给出）。
    creatures: dict[int, tuple[str, ...]] = field(default_factory=dict)
    artifacts: dict[int, str] = field(default_factory=dict)


def _decode(path: Path) -> tuple[int, int, list[tuple[int, int, tuple[int, ...]]]]:
    """解码 int32 大端回放。返回 (map_type, day_time, [(round, event, payload)])。"""

    raw = path.read_bytes()
    if len(raw) < 28:
        raise ValueError("miracle replay too short to contain a header")
    count = len(raw) // 4
    ints = struct.unpack(">" + "i" * count, raw[: count * 4])
    header = ints[0:7]
    records: list[tuple[int, int, tuple[int, ...]]] = []
    for offset in range(7, len(ints) - 6, 7):
        chunk = ints[offset : offset + 7]
        records.append((int(chunk[0]), int(chunk[1]), tuple(int(x) for x in chunk[2:])))
    return int(header[3]), int(header[4]), records


def _cube(x: int, y: int) -> tuple[int, int, int]:
    """(x, y) → 立方体坐标。第三维由恒等式 x + y + z = 0 唯一确定。"""

    return (x, y, -x - y)


def _unit_data(data: Mapping[str, object], type_name: str, level: int) -> dict[str, object]:
    """从 Data.json 按 (类型, 等级) 取属性。

    等级在事件里是 1..3，表里是 0..2 下标；越界时夹到合法区间——
    回放与 Data.json 版本不一致时宁可用最近的等级，也不要让整局探针失败。
    """

    table = (data.get("UnitData") or {}).get(type_name)  # type: ignore[union-attr]
    if not isinstance(table, Mapping):
        return {}
    index = max(0, min(int(level) - 1, 2))

    def pick(key: str, default: object = 0) -> object:
        values = table.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if not values:
                return default
            return values[min(index, len(values) - 1)]
        return values if values is not None else default

    return {
        "cost": int(pick("cost", 0)),  # type: ignore[arg-type]
        "atk": int(pick("atk", 0)),  # type: ignore[arg-type]
        "hp": int(pick("hp", 1)),  # type: ignore[arg-type]
        "atk_range": tuple(int(v) for v in (pick("atk_range", [1, 1]) or [1, 1])),  # type: ignore[union-attr]
        "max_move": int(pick("max_move", 0)),  # type: ignore[arg-type]
        "cool_down": int(pick("cool_down", 0)),  # type: ignore[arg-type]
        "flying": bool(table.get("flying")),
        "atk_flying": bool(table.get("atk_flying")),
        "agility": bool(table.get("agility")),
        "holy_shield": bool(table.get("holy_shield")),
    }


def _apply(state: ProbeState, event: int, payload: Sequence[int], data: Mapping[str, object]) -> None:
    """把一条事件应用到局面上（就地修改）。

    只处理会改变公开局面的事件。Attack/Attacking/Attacked/Leave/Arrive 是
    过程性事件，其后果都由紧随的 Damage/Move 表达，无需单独处理。
    """

    if event == EVENT_SPAWN and len(payload) >= 5:
        encoded, level, pos_x, pos_y, unit_id = payload[:5]
        camp, type_index = divmod(int(encoded), 10)
        if not 0 <= type_index < len(BACKEND_CREATURES):
            return
        type_name = BACKEND_CREATURES[type_index]
        if not type_name:
            return
        attributes = _unit_data(data, type_name, int(level))
        state.units[int(unit_id)] = ProbeUnit(
            id=int(unit_id),
            camp=int(camp),
            type=type_name,
            level=int(level),
            pos=_cube(int(pos_x), int(pos_y)),
            atk=int(attributes.get("atk", 0)),
            max_hp=int(attributes.get("hp", 1)),
            hp=int(attributes.get("hp", 1)),
            atk_range=tuple(attributes.get("atk_range", (1, 1))),  # type: ignore[arg-type]
            max_move=int(attributes.get("max_move", 0)),
            cool_down=int(attributes.get("cool_down", 0)),
            cost=int(attributes.get("cost", 0)),
            flying=bool(attributes.get("flying")),
            atk_flying=bool(attributes.get("atk_flying")),
            agility=bool(attributes.get("agility")),
            holy_shield=bool(attributes.get("holy_shield")),
        )
    elif event == EVENT_MOVE and len(payload) >= 3:
        unit = state.units.get(int(payload[0]))
        if unit is not None:
            unit.pos = _cube(int(payload[1]), int(payload[2]))
    elif event == EVENT_DAMAGE and len(payload) >= 3:
        target_id, _source_id, damage = int(payload[0]), int(payload[1]), int(payload[2])
        unit = state.units.get(target_id)
        if unit is not None:
            unit.hp -= damage
        else:
            # 目标不是单位就是神迹本体：id 0/1 对应两个阵营。
            if target_id in (0, 1) and target_id < len(state.miracle_hp):
                state.miracle_hp[target_id] -= damage
    elif event == EVENT_HEAL and len(payload) >= 3:
        unit = state.units.get(int(payload[0]))
        if unit is not None:
            unit.hp = min(unit.max_hp, unit.hp + int(payload[2]))
    elif event == EVENT_DEATH and payload:
        state.units.pop(int(payload[0]), None)
    elif event == EVENT_BUFF_ADD and len(payload) >= 2:
        unit = state.units.get(int(payload[0]))
        if unit is not None and int(payload[1]) == 2:  # HolyShield
            unit.holy_shield = True
    elif event == EVENT_BUFF_REMOVE and len(payload) >= 2:
        unit = state.units.get(int(payload[0]))
        if unit is not None and int(payload[1]) == 2:
            unit.holy_shield = False
    elif event == EVENT_GAME_START and len(payload) >= 5:
        camp = int(payload[0])
        artifact_index = int(payload[1]) % 10
        if 0 <= artifact_index < len(BACKEND_ARTIFACTS):
            state.artifacts[camp] = BACKEND_ARTIFACTS[artifact_index]
        names: list[str] = []
        for encoded in payload[2:5]:
            index = int(encoded) % 10
            if 0 <= index < len(BACKEND_CREATURES) and BACKEND_CREATURES[index]:
                names.append(BACKEND_CREATURES[index])
        state.creatures[camp] = tuple(names)


def _to_sdk_map(state: ProbeState):
    """把重演状态包装成 SDK 的 ``Map``，供候选策略与 ``calculator`` 使用。

    ``Map()`` 的构造函数已经把**静态地形**建好了（兵营坐标、两个神迹的位置与
    召唤点、深渊/沼泽障碍），所以这里只需要覆盖**动态部分**：单位列表与
    神迹血量。

    ⚠️ 只填 ``units`` 是不够的。候选策略普遍要读 ``map.miracles``（自己/对方的
    本体位置，用来定方向）与 ``map.barracks``（占点目标）。漏掉它们会让候选在
    第一行几何计算就抛异常，而 ``_probe_actions`` 把异常吞掉记成 END——
    表现为"空动作占比 100%"，看起来像候选很消极，实际是探针没把环境搭全。

    直接构造 SDK 的 ``Unit`` 会踩到类型顺序问题（SDK 的 UNIT_TYPE 与后端不同），
    所以单位继续用鸭子类型的 ``ProbeUnit``：``reachable`` / ``cube_distance`` 与
    候选代码访问的都是 ``pos`` / ``camp`` / ``hp`` / ``type`` / ``flying`` /
    ``max_move`` 这些属性，``ProbeUnit`` 已全部提供。
    """

    from gameunit import Map

    sdk_map = Map()
    sdk_map.units = list(state.units.values())  # type: ignore[assignment]
    for camp, miracle in enumerate(getattr(sdk_map, "miracles", [])[:2]):
        if camp < len(state.miracle_hp):
            miracle.hp = int(state.miracle_hp[camp])
    return sdk_map


def _legal_support(state: ProbeState, camp: int, sdk_map: object) -> tuple[str, ...]:
    """枚举该阵营当前的全部合法操作。

    四类操作（与 decision_space.yaml 的 atomic_actions 对应）：
      * ``END``            —— 结束回合，永远合法（对应 ``HOLD``）；
      * ``M:<id>:<x>:<y>`` —— 移动，用 SDK 的 ``reachable`` 求可达集；
      * ``A:<id>:<tid>``   —— 攻击，用射程与对空规则筛选；
      * ``S:<type>``       —— 召唤己方手牌里的生物；
      * ``U:<x>:<y>``/``U:<id>`` —— 使用神器（按 target_type 分坐标/单位）。

    为什么不逐个向后端确认合法性：miracle 的 SDK 没有 ``is_operation_valid``
    这类 dry-run 入口（对比 antwar），而后端判定深埋在 StateSystem 里、
    需要完整对局上下文。所以这里用**官方 calculator 的几何判定**
    （``reachable`` / ``cube_distance`` 就是后端用的同一套函数）重建合法集。
    这比 generals 的"深拷贝试执行"弱一点：法力值约束无法从回放推出
    （事件流不记法力），所以召唤/神器按"手牌里有就算合法"计入。

    ⚠️ 这个口径差异必须记住：它让 |A(s)| 略微**偏大**（含少数因法力不足而
    实际不可用的召唤/神器）。仍然远优于常量 7 的字母表近似，而且偏差方向
    是一致的、可解释的。
    """

    from calculator import cube_distance, reachable

    legal = {"END"}

    for unit in state.units.values():
        if unit.camp != camp or unit.hp <= 0:
            continue
        # 移动：官方 reachable 已经处理了障碍、飞行、行动力。
        #
        # ⚠️ 它返回的是**按步数分层**的嵌套列表 —— [[起点], [走1步可达...],
        # [走2步可达...], ...]，不是平坦的坐标列表。当成平坦列表会把每一层
        # 当作一个"坐标"，结果一个合法移动都提取不出来（实测就踩了这个）。
        if unit.can_move and unit.max_move > 0:
            try:
                layers = reachable(unit, sdk_map)
            except Exception:  # noqa: BLE001 - 几何判定崩溃视为不可移动
                layers = []
            for step, layer in enumerate(layers):
                if step == 0:
                    # 第 0 层是原地，不是一次移动。
                    continue
                for target in layer:
                    if len(target) >= 2:
                        legal.add(f"M:{unit.id}:{target[0]}:{target[1]}")
        # 攻击：射程区间 + 对空规则。
        if unit.can_atk:
            low, high = (
                unit.atk_range if len(unit.atk_range) >= 2 else (1, unit.atk_range[0])
            )
            for other in state.units.values():
                if other.camp == camp or other.hp <= 0:
                    continue
                if other.flying and not unit.atk_flying:
                    continue
                try:
                    distance = cube_distance(unit.pos, other.pos)
                except Exception:  # noqa: BLE001
                    continue
                if low <= distance <= high:
                    legal.add(f"A:{unit.id}:{other.id}")

    # 召唤：手牌里的生物。法力约束推不出来，见函数注释。
    for type_name in state.creatures.get(camp, ()):
        legal.add(f"S:{type_name}")

    # 神器：按 target_type 决定参数形态。
    artifact = state.artifacts.get(camp)
    if artifact:
        legal.add(f"U:{artifact}")

    return tuple(sorted(legal))


def _occupancy_id(state: ProbeState) -> str:
    """状态指纹：同一局面无论怎么到达，指纹相同。"""

    payload = {
        "units": sorted(
            (unit.id, unit.camp, unit.type, unit.level, list(unit.pos), unit.hp)
            for unit in state.units.values()
        ),
        "miracle_hp": list(state.miracle_hp),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_entry_root(candidate: Path) -> Path:
    """含 main.py 的最浅目录（与 A 侧 runtime._find_python_entry 同规则）。"""

    candidates = sorted(
        candidate.rglob("main.py"),
        key=lambda path: (len(path.relative_to(candidate).parts), path.as_posix()),
    )
    if not candidates:
        raise RuntimeError(f"miracle candidate has no main.py: {candidate}")
    return candidates[0].parent


def _load_ai_class(entry_root: Path):
    """取出候选的 AI 类。

    Miracle 的官方模板是 ``class AI(AiClient)`` 且实现 ``play()``；
    池子里基本都遵守（这是 SDK 强制的继承关系）。找不到就明确报错，
    绝不静默返回假策略——那会让 IG 变成纯噪声。
    """

    import importlib.util

    ai_py = entry_root / "ai.py"
    if not ai_py.is_file():
        raise RuntimeError(f"miracle candidate has no ai.py: {entry_root}")
    spec = importlib.util.spec_from_file_location("candidate_ai", ai_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load miracle candidate ai.py: {ai_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("AI", "MyAI", "Ai"):
        attribute = getattr(module, name, None)
        if isinstance(attribute, type):
            return attribute
    raise RuntimeError("miracle candidate exposes no AI class")


def _probe_actions(ai_class: type, state: ProbeState, camp: int, sdk_map: object) -> list[str]:
    """让候选面对这个局面做决策，返回它发出的操作序列。

    要隔离候选的三种"越界"行为（都是实测遇到的，不是假想）：

    1. **操作方法会写 stdout**。``summon`` / ``move`` / ``attack`` / ``use`` /
       ``end_round`` 在 SDK 里靠 stdio 与后端通信。探针不能让它们真写出去
       （会污染我们自己的 ``AGENTBENCH_POLICY_TRACE`` 输出），所以全部替换成记录器。

    2. **``__init__`` 会读 stdin**。``AiClient.__init__`` 里有
       ``read_opt()['camp']``，没有后端喂数据就会阻塞或崩。所以用
       ``__new__`` 绕过构造函数，再手工把 ``play()`` 依赖的字段填好。

    3. **``play()`` 可能直接调 ``exit()``**。池子里确实有这种写法
       （例如 ``if self.round < 20: self.end_round() else: exit(0)``）。
       ``exit()`` 抛 ``SystemExit``，它继承 ``BaseException`` 而**不是**
       ``Exception``，所以必须显式接住——否则整个探针进程被这一个候选杀掉，
       表现为"没有任何输出、退出码 0"，极难排查。
    """

    recorded: list[str] = []

    instance = ai_class.__new__(ai_class)  # type: ignore[misc]
    # 绕过 __init__（它会 read_opt() 读 stdin）。手工补齐 play() 依赖的字段：
    # 字段名取自 ai_client.AiClient —— 阵营叫 my_camp，不是 camp。
    instance.map = sdk_map  # type: ignore[attr-defined]
    instance.round = state.round  # type: ignore[attr-defined]
    instance.my_camp = camp  # type: ignore[attr-defined]
    instance.camp = camp  # type: ignore[attr-defined]
    instance.artifacts = list(state.artifacts.get(camp, ()) or ["HolyLight"])  # type: ignore[attr-defined]
    instance.creatures = list(state.creatures.get(camp, ()))  # type: ignore[attr-defined]
    instance.players = [  # type: ignore[attr-defined]
        type("P", (), {"camp": 0, "mana": 99, "artifact": None, "creature_capacity": 99})(),
        type("P", (), {"camp": 1, "mana": 99, "artifact": None, "creature_capacity": 99})(),
    ]
    instance.player = instance.players[camp]  # type: ignore[attr-defined]
    instance.operations = []  # type: ignore[attr-defined]

    def _summon(type_name: object, *_args: object, **_kwargs: object) -> None:
        recorded.append(f"S:{type_name}")

    def _move(unit: object, dest: object, *_args: object, **_kwargs: object) -> None:
        unit_id = getattr(unit, "id", unit)
        position = tuple(dest) if isinstance(dest, Sequence) else (dest,)
        recorded.append(
            f"M:{unit_id}:{position[0]}:{position[1] if len(position) > 1 else 0}"
        )

    def _attack(unit: object, target: object, *_args: object, **_kwargs: object) -> None:
        recorded.append(f"A:{getattr(unit, 'id', unit)}:{getattr(target, 'id', target)}")

    def _use(*args: object, **_kwargs: object) -> None:
        recorded.append(f"U:{args[0] if args else 'artifact'}")

    def _end(*_args: object, **_kwargs: object) -> None:
        recorded.append("END")

    def _noop(*_args: object, **_kwargs: object) -> None:
        return None

    for name, handler in (
        ("summon", _summon),
        ("move", _move),
        ("attack", _attack),
        ("use", _use),
        ("use_artifact", _use),
        ("end_round", _end),
        ("end_turn", _end),
        # 这些是与后端通信的方法，探针里必须变成空操作。
        ("init", _noop),
        ("update_game_info", _noop),
        ("send_opt", _noop),
        ("read_opt", _noop),
    ):
        setattr(instance, name, handler)

    play = getattr(instance, "play", None)
    if callable(play):
        try:
            play()
        except SystemExit:
            # 候选调了 exit()。这是真实存在的写法，且 SystemExit 继承
            # BaseException 而非 Exception —— 不显式接住会让整个探针进程
            # 静默退出（无输出、退出码 0）。
            pass
        except Exception as error:  # noqa: BLE001 - 候选崩溃记为结束回合并继续
            # 把错误摘要写进 stderr。一个状态上的崩溃不该让整局探针作废，
            # 但**完全吞掉**会让"探针环境没搭全"伪装成"候选很消极"：
            # 实测漏填 map.miracles 时候选每回合都在几何计算处抛异常，
            # 表现为空动作占比 100%，极难归因。
            print(
                f"[miracle-probe] candidate play() raised "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
            )
    return recorded or ["END"]


def run(candidate: Path, replay_path: Path, match_id: str, role: str) -> dict[str, object]:
    entry_root = _find_entry_root(candidate)
    if str(entry_root) not in sys.path:
        sys.path.insert(0, str(entry_root))

    data_path = entry_root / "Data.json"
    data: Mapping[str, object] = {}
    if data_path.is_file():
        data = json.loads(data_path.read_text(encoding="utf-8"))

    _map_type, _day_time, records = _decode(replay_path)
    ai_class = _load_ai_class(entry_root)
    camp = 0 if role == "P0" else 1

    state = ProbeState()
    decisions: list[dict[str, object]] = []
    probed_rounds: set[int] = set()

    for round_index, event, payload in records:
        if event == EVENT_GAME_END:
            break
        if event == EVENT_TURN_START and payload:
            state.round = round_index
            actor = int(payload[0])
            # 只在属于本方的回合上探测，且每回合一次。
            if actor == camp and round_index not in probed_rounds:
                probed_rounds.add(round_index)
                sdk_map = _to_sdk_map(state)
                support = _legal_support(state, camp, sdk_map)
                actions = _probe_actions(ai_class, state, camp, sdk_map)
                occupancy = _occupancy_id(state)
                # 一个回合内可以发多条操作，而**线协议按每条操作计一个决策**。
                # 所以这里必须逐操作展开成独立的 decision，不能一个回合只记一条。
                #
                # 曾经每回合只记 1 条，于是探针产出 24 个决策而线协议有 187 个
                # （gap 163 远超容差 2），整局被判为对不齐、IG 静默退回
                # opcode_alphabet 近似 —— 又一个静默降级。
                #
                # 支持集在回合内会随已发操作收缩（单位动过就不能再动、法力被消耗），
                # 但事件流不记法力、也无法在候选侧无损重放"部分执行"的中间态，
                # 所以这里对同回合内的每条操作复用回合开始时的支持集。
                # 这让 |A(s)| 在回合后段略微偏大，方向一致且可解释——
                # 与 _legal_support 里"召唤/神器按手牌里有就算合法"是同一类保守近似。
                for offset, action in enumerate(actions):
                    decisions.append(
                        {
                            "state_id": (
                                f"{match_id}:r{round_index:04d}:c{camp}:s{offset:03d}"
                            ),
                            "actions": [action],
                            "legal_supports": [list(support)],
                            "occupancy_id": occupancy,
                        }
                    )
            continue
        _apply(state, event, payload, data)

    return {"match_id": match_id, "role": role, "decisions": decisions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--role", choices=("P0", "P1"), required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.candidate.resolve(),
        arguments.replay.resolve(),
        arguments.match_id,
        arguments.role,
    )
    print("AGENTBENCH_POLICY_TRACE=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

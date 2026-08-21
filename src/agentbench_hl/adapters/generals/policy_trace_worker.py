"""Generals（将军棋战）的候选侧决策探针。

它做什么
--------
在**候选自己的包**里重放一局公开回放：逐回合把状态喂给候选的决策函数，
记录它实际发出的命令序列、以及当时的合法命令集（支持集）。上层
``application/support_probe.py`` 用这些数据算精确的 behavioral IG。

与 antwar / antwar2 的同构关系
------------------------------
遵循同一个契约：``--candidate/--replay/--match-id/--role`` 进，stdout 上一行
``AGENTBENCH_POLICY_TRACE=<json>`` 出，json 含 ``decisions[]``，每项有
``state_id`` / ``actions`` / ``legal_supports`` / ``occupancy_id``。

Generals 的特殊之处（决定了本文件的写法）
----------------------------------------
1. **SDK 没有 dry-run**。antwar 有 ``is_operation_valid()``（内部
   ``apply_operation(dry_run=True)``），可以白问不改状态。generals 只有
   ``execute_single_command()``，它**返回 bool 且直接改状态**。
   所以枚举合法集必须"**深拷贝一份状态 → 试执行 → 看返回值 → 丢弃副本**"。
   这比 dry-run 慢，但是在没有 dry-run 时唯一正确的做法；用真实状态试执行
   会把状态改坏，后面所有决策都失真。

2. **参数域巨大，必须按"命令族"枚举而不是笛卡尔积**。
   一个 ``army_move`` 是 (225 格 × 4 方向 × 兵力)，全展开有十万量级，
   每个都深拷贝试执行是不可行的。所以支持集的粒度取
   **(命令码, 关键参数)** 这一层：
   - 移动类：枚举 (自己占据的格子 × 4 个方向)，兵力取"全部"这一代表值；
   - 将领类：枚举 (自己的将领 × 具体子动作)；
   - 全局类（科技/超武/召唤）：枚举各自的子类型。
   这与 ``decision_space.yaml`` 声明的 ``opcode_alphabet(8)`` 相比是**大幅细化**
   （从 8 个操作码细化到状态相关的数百个具体动作），仍然是**精确**的：
   每个被计入的动作都真的通过了官方判定。

3. **命令序列 = 一次决策**。generals 一个回合可以发多条命令（以单行 ``8``
   结束）。所以 ``actions`` 是这一回合的命令列表，``legal_supports`` 给出
   每一步的合法集（与 antwar2 的 pending 累积同构）——因为发完第一条命令后
   金币/兵力变了，第二条命令的合法集也就变了。

4. **回放是逐行 JSON 增量**。首行是全图快照，之后每行是变化的
   ``Cells``/``Generals`` 加上真实发生的 ``Action``。所以状态要靠首行建立、
   再用回放里的真实命令推进。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

#: 棋盘尺寸（logic/gamestate.py 的 row / col）。
BOARD_ROW = 15
BOARD_COL = 15

#: 四个移动方向的协议编码（1..4，见 decision_space.yaml 的 army_move）。
DIRECTIONS = (1, 2, 3, 4)

#: 将领升级的三个属性（1 产出 / 2 防御 / 3 机动）。
UPGRADE_ATTRIBUTES = (1, 2, 3)

#: 五个战法（1..5；1/2 需要目标坐标，3/4/5 不需要）。
SKILLS = (1, 2, 3, 4, 5)

#: 四项科技（1 行动力 / 2 攀岩 / 3 免疫沼泽 / 4 超级武器）。
TECHS = (1, 2, 3, 4)

#: 四种超级武器（1 核弹 / 2 强化 / 3 传送 / 4 时间停止）。
SUPER_WEAPONS = (1, 2, 3, 4)


def _load_frames(path: Path) -> list[dict]:
    """读逐行 JSON 回放。

    后端用 ``str(dict).replace("'", '"')`` 写出，所以每行都是合法 JSON；
    终局行/traceback 可能不是，按 replay_format.md 的约定跳过。
    """

    frames: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            frames.append(value)
    if not frames:
        raise ValueError("generals replay has no usable frames")
    return frames


def _build_state(initial: Mapping[str, object]):
    """用回放首行的全图快照建立 GameState。"""

    from generals_impact_game.gamestate import Cell, GameState, Generals

    state = GameState()
    state.round = int(initial.get("Round", 1))

    # 棋盘：初始行给出全部 225 格。
    board = [[Cell(position=[x, y]) for y in range(BOARD_COL)] for x in range(BOARD_ROW)]
    for entry in initial.get("Cells") or []:
        if not isinstance(entry, Sequence) or len(entry) < 3:
            continue
        position = entry[0]
        if not isinstance(position, Sequence) or len(position) < 2:
            continue
        x, y = int(position[0]), int(position[1])
        if not (0 <= x < BOARD_ROW and 0 <= y < BOARD_COL):
            continue
        board[x][y].player = int(entry[1])
        board[x][y].army = int(entry[2])
        if len(entry) >= 4:
            # 地形（若回放带出）：0 平原 / 1 沙漠 / 2 沼泽 等。
            board[x][y].type = int(entry[3])
    state.board = board

    generals: list = []
    for raw in initial.get("Generals") or []:
        if not isinstance(raw, Mapping):
            continue
        general = Generals()
        general.id = int(raw.get("Id", 0))
        general.player = int(raw.get("Player", -1))
        position = raw.get("Position") or [0, 0]
        general.position = [int(position[0]), int(position[1])]
        levels = raw.get("Level") or [1, 1, 1]
        general.produce_level = int(levels[0])
        general.defence_level = int(levels[1])
        general.mobility_level = int(levels[2])
        general.skill_duration = list(raw.get("Skill_rest") or [0, 0, 0])
        general.rest_move = int(raw.get("Rest_move", 0) or 0)
        general.alive = int(raw.get("Alive", 1))
        generals.append(general)
        # 把将领挂到棋盘格上：合法性判定要靠它认出"这里有我的将领"。
        x, y = general.position
        if 0 <= x < BOARD_ROW and 0 <= y < BOARD_COL:
            board[x][y].generals = general
    state.generals = generals
    state.next_generals_id = max((item.id for item in generals), default=-1) + 1

    state.coin = [int(item) for item in (initial.get("Coins") or [0, 0])[:2]]
    tech = initial.get("Tech_level") or [[2, 0, 0, 0], [2, 0, 0, 0]]
    state.tech_level = [[int(value) for value in row] for row in tech[:2]]
    state.super_weapon_cd = [
        int(item) for item in (initial.get("Weapon_cds") or [-1, -1])[:2]
    ]
    state.super_weapon_unlocked = [
        bool(row[3]) if len(row) > 3 else False for row in state.tech_level
    ]
    state.rest_move_step = [int(row[0]) for row in state.tech_level]
    return state


def _try_command(state: object, player: int, command: int, params: list[int]) -> bool:
    """在**状态副本**上试执行一条命令，返回它是否合法。

    为什么必须深拷贝：``execute_single_command`` 没有 dry-run 模式，它会真的
    改状态。若直接在真实状态上试，枚举一遍合法集就把局面改得面目全非，
    之后所有决策都是在错误状态上做的。

    判定自身抛异常时算非法——与 antwar 的处理一致：候选包自带的 SDK 在边界
    状态上会抛而不是返回 False（antwar 的例子是满级时 ``[200,250][2]`` 越界）。
    一个动作的崩溃不该掀翻整局探针，而且"连合法性都算不出来的操作"提交上去
    同样会被后端拒绝。
    """

    from generals_impact_game.execute import execute_single_command

    try:
        probe = copy.deepcopy(state)
        return bool(execute_single_command(player, probe, command, list(params)))
    except Exception:  # noqa: BLE001 - SDK 边界问题，视为非法
        return False


def _action_key(command: int, params: Sequence[int]) -> str:
    return ":".join(str(int(token)) for token in (command, *params))


def _my_cells(state: object, player: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    board = getattr(state, "board", [])
    for x in range(min(BOARD_ROW, len(board))):
        row = board[x]
        for y in range(min(BOARD_COL, len(row))):
            cell = row[y]
            if int(getattr(cell, "player", -1)) == player and int(getattr(cell, "army", 0)) > 1:
                cells.append((x, y))
    return cells


def _my_generals(state: object, player: int) -> list[object]:
    return [
        general
        for general in getattr(state, "generals", [])
        if int(getattr(general, "player", -1)) == player
        and int(getattr(general, "alive", 1)) == 1
    ]


def _legal_support(state: object, player: int) -> tuple[str, ...]:
    """枚举当前状态下该玩家的全部合法命令（按命令族粒度）。

    包含空动作 ``HOLD``（只发终止符 8）：一个回合不发任何命令永远合法，
    它是决策空间里真实存在的一个动作。漏掉会让 |A(s)| 系统性偏小 1。

    粒度说明见模块头注释第 2 条：移动类的兵力参数取"全部兵力"这一代表值，
    不做兵力维度的笛卡尔展开——否则每个状态要做十万次深拷贝，不可行。
    """

    legal = {"HOLD"}

    # 命令 1：army_move（自己占据且兵力 > 1 的格子 × 4 方向）。
    for x, y in _my_cells(state, player):
        army = int(state.board[x][y].army)  # type: ignore[index]
        for direction in DIRECTIONS:
            params = [x, y, direction, army - 1]
            if _try_command(state, player, 1, params):
                legal.add(_action_key(1, (x, y, direction, army - 1)))

    for general in _my_generals(state, player):
        general_id = int(getattr(general, "id", 0))
        position = list(getattr(general, "position", [0, 0]))

        # 命令 2：general_move（将领移动到相邻格）。
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            target = [position[0] + dx, position[1] + dy]
            if not (0 <= target[0] < BOARD_ROW and 0 <= target[1] < BOARD_COL):
                continue
            if _try_command(state, player, 2, [general_id, *target]):
                legal.add(_action_key(2, (general_id, *target)))

        # 命令 3：upgrade_general（三个属性各一档）。
        for attribute in UPGRADE_ATTRIBUTES:
            if _try_command(state, player, 3, [general_id, attribute]):
                legal.add(_action_key(3, (general_id, attribute)))

        # 命令 4：release_skill。1/2 需要目标坐标，3/4/5 不需要。
        for skill in SKILLS:
            if skill in (1, 2):
                # 目标坐标的全展开是 225 格；取将领自身位置作为代表值，
                # 与移动类同样的理由（避免十万级深拷贝）。
                params = [general_id, skill, position[0], position[1]]
            else:
                params = [general_id, skill, -1, -1]
            if _try_command(state, player, 4, params):
                legal.add(_action_key(4, params))

    # 命令 5：tech_update。
    for tech in TECHS:
        if _try_command(state, player, 5, [tech]):
            legal.add(_action_key(5, (tech,)))

    # 命令 6：super_weapon。传送(3)需要起点+终点，其余只要一个坐标。
    for weapon in SUPER_WEAPONS:
        cells = _my_cells(state, player)
        target = cells[0] if cells else (0, 0)
        params = (
            [weapon, target[0], target[1], target[0], target[1]]
            if weapon == 3
            else [weapon, target[0], target[1]]
        )
        if _try_command(state, player, 6, params):
            legal.add(_action_key(6, params))

    # 命令 7：call_subgeneral（在自己的格子上召唤副将）。
    for x, y in _my_cells(state, player):
        if _try_command(state, player, 7, [x, y]):
            legal.add(_action_key(7, (x, y)))

    return tuple(sorted(legal))


def _occupancy_id(state: object) -> str:
    """状态指纹：只取公开可见字段，同一局面无论怎么到达指纹相同。"""

    board = getattr(state, "board", [])
    cells = [
        (x, y, int(getattr(board[x][y], "player", -1)), int(getattr(board[x][y], "army", 0)))
        for x in range(min(BOARD_ROW, len(board)))
        for y in range(min(BOARD_COL, len(board[x])))
        if int(getattr(board[x][y], "army", 0)) != 0
        or int(getattr(board[x][y], "player", -1)) != -1
    ]
    generals = [
        (
            int(getattr(item, "id", 0)),
            int(getattr(item, "player", -1)),
            tuple(int(value) for value in getattr(item, "position", [0, 0])),
            int(getattr(item, "produce_level", 1)),
            int(getattr(item, "defence_level", 1)),
            int(getattr(item, "mobility_level", 1)),
            int(getattr(item, "alive", 1)),
        )
        for item in getattr(state, "generals", [])
    ]
    payload = {
        "round": int(getattr(state, "round", 0)),
        "cells": cells,
        "generals": sorted(generals),
        "coin": [int(item) for item in getattr(state, "coin", [])],
        "tech": [[int(v) for v in row] for row in getattr(state, "tech_level", [])],
        "weapon_cd": [int(item) for item in getattr(state, "super_weapon_cd", [])],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_strategy(entry_root: Path):
    """取出决策函数，统一成 ``f(round, seat, state) -> ops``。

    两类候选，优先级不同：

    **一、我们自己生成的候选（HL 迭代产物）** —— IG 真正要测的对象。
    它遵守框架注入的契约（见候选包里的 ``CANDIDATE_CONTRACT.md``）：
    ``ai.py`` 里有 ``class AI``，方法签名是
    ``decide(self, round: int, my_seat: int, state: GameState) -> list[list[int]]``。
    先按这条契约找，命中率 100%。

    ⚠️ 曾经只解析 ``main.py`` 顶层的 ``run_ai(...)``，结果一个都没命中：
    框架生成的 ``main.py`` 把 ``run_ai(decide)`` 写在 ``main()`` **函数体内**，
    而 ``decide`` 是从 ``ai.py`` 的 ``AI`` 实例上取的方法。表现为
    "exposes no recognisable ai function"，IG 静默退回 opcode_alphabet
    近似 —— 又一个静默降级（LESSONS_LEARNED 方法论第 2 条）。

    **二、人类池里的历史候选** —— 结构五花八门（实测 83 个 Python 候选里
    71 个用官方三参函数签名、12 个是无参函数 + 模块级全局）。这些只在离线
    重算时遇到，按 run_ai 捕获 / 签名匹配 / 常见命名去找。

    找不到就明确报错，绝不静默返回假策略——那会让 IG 变成纯噪声。
    """

    import ast
    import importlib.util
    import inspect

    # ---------------------------------------------------------------- 路径一
    ai_py = entry_root / "ai.py"
    if ai_py.is_file():
        ai_spec = importlib.util.spec_from_file_location("candidate_ai", ai_py)
        if ai_spec is not None and ai_spec.loader is not None:
            ai_module = importlib.util.module_from_spec(ai_spec)
            try:
                ai_spec.loader.exec_module(ai_module)
            except Exception:  # noqa: BLE001 - 退到路径二
                ai_module = None
            if ai_module is not None:
                for class_name in ("AI", "Agent", "MyAI"):
                    ai_class = getattr(ai_module, class_name, None)
                    if not isinstance(ai_class, type):
                        continue
                    try:
                        instance = ai_class()
                    except Exception:  # noqa: BLE001 - 构造失败就绕过 __init__
                        instance = ai_class.__new__(ai_class)  # type: ignore[misc]
                    for method_name in ("decide", "play", "act", "step"):
                        handler = getattr(instance, method_name, None)
                        if callable(handler):
                            return handler

    # ---------------------------------------------------------------- 路径二
    main_py = entry_root / "main.py"
    if not main_py.is_file():
        raise RuntimeError(f"generals candidate has no ai.py or main.py: {entry_root}")

    # 静态解析被 run_ai() 传入的函数名。要遍历**整棵树**（含函数体内的调用），
    # 不能只看模块顶层——框架生成的候选就是把它写在 main() 里的。
    preferred: list[str] = []
    try:
        tree = ast.parse(main_py.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_ai"
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                preferred.append(node.args[0].id)
    except SyntaxError:
        pass

    spec = importlib.util.spec_from_file_location("candidate_main", main_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generals candidate main.py: {main_py}")
    module = importlib.util.module_from_spec(spec)

    # 候选的 main.py 顶层就调 run_ai(...)，而 run_ai → GameController.init()
    # → load_map() 会**从 stdin 读**初始地图。探针没有对手进程给它喂数据，
    # 直接 exec 会在 json.loads("") 上崩掉。
    #
    # 所以要打两个桩：
    #   * run_ai      —— 只捕获传进来的决策函数，不进通信循环；
    #   * load_map    —— 返回一个占位状态，避免读 stdin。
    # 两者都在 exec 之后还原，不污染后续调用。
    captured: list[object] = []

    import generals_impact_game.controller as controller
    from generals_impact_game.gamestate import GameState

    original_run_ai = getattr(controller, "run_ai", None)
    original_load_map = getattr(controller, "load_map", None)
    controller.run_ai = lambda func: captured.append(func)  # type: ignore[assignment]
    controller.load_map = lambda: (0, GameState())  # type: ignore[assignment]
    try:
        spec.loader.exec_module(module)
        # run_ai 可能写在 main() 里，顶层 exec 不会触发它。主动调一次 main()，
        # 桩会把决策函数捕获下来（load_map 已被替换，不会读 stdin）。
        entry = getattr(module, "main", None)
        if not captured and callable(entry):
            try:
                entry()
            except SystemExit:
                pass
            except Exception:  # noqa: BLE001
                pass
    except SystemExit:
        pass
    except Exception:  # noqa: BLE001 - 顶层还有别的副作用，退回静态查找
        pass
    finally:
        if original_run_ai is not None:
            controller.run_ai = original_run_ai  # type: ignore[assignment]
        if original_load_map is not None:
            controller.load_map = original_load_map  # type: ignore[assignment]

    def _adapt(handler: object):
        """把候选的实际签名适配成 ``f(round, seat, state)``。"""

        try:
            arity = len(inspect.signature(handler).parameters)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            arity = 3

        if arity >= 3:
            return handler
        if arity == 2:
            return lambda round_index, seat, state: handler(seat, state)  # type: ignore[operator]
        if arity == 1:
            return lambda round_index, seat, state: handler(state)  # type: ignore[operator]

        # 无参函数：从模块全局读状态。调用前把 main.py 里常见的全局名写好，
        # 让它看到当前局面。哪个名字都没有就说明这个候选不可探测。
        def _global_style(round_index: int, seat: int, state: object):
            for name in ("state", "game_state", "gamestate", "STATE"):
                if hasattr(module, name):
                    setattr(module, name, state)
            for name in ("my_seat", "seat", "MY_SEAT", "player"):
                if hasattr(module, name):
                    setattr(module, name, seat)
            for name in ("round", "round_num", "ROUND"):
                if hasattr(module, name):
                    setattr(module, name, round_index)
            return handler()  # type: ignore[operator]

        return _global_style

    if captured and callable(captured[0]):
        return _adapt(captured[0])

    # 按签名找：形参里带 my_seat / state 的最可能是决策函数。
    for name, handler in vars(module).items():
        if not callable(handler) or name.startswith("_"):
            continue
        try:
            parameters = set(inspect.signature(handler).parameters)
        except (TypeError, ValueError):
            continue
        if {"my_seat", "state"} <= parameters or {"seat", "state"} <= parameters:
            return _adapt(handler)

    for name in (*preferred, "decide", "ai", "example_ai", "silly_ai", "my_ai"):
        handler = getattr(module, name, None)
        if callable(handler):
            return _adapt(handler)
    raise RuntimeError(
        "generals candidate exposes no recognisable ai function; tried ai.py AI.decide, "
        f"run_ai capture, signature match, and names {[*preferred, 'decide', 'ai']}"
    )


def _find_entry_root(candidate: Path) -> Path:
    """含 main.py 的最浅目录（与 A 侧 runtime._find_python_entry 同规则）。"""

    candidates = sorted(
        candidate.rglob("main.py"),
        key=lambda path: (len(path.relative_to(candidate).parts), path.as_posix()),
    )
    if not candidates:
        raise RuntimeError(f"generals candidate has no main.py: {candidate}")
    return candidates[0].parent


def _commands(frame: Mapping[str, object]) -> list[list[int]]:
    """从回放帧取出真实发生的命令。"""

    action = frame.get("Action")
    if not isinstance(action, Sequence) or not action:
        return []
    try:
        tokens = [int(item) for item in action]
    except (TypeError, ValueError):
        return []
    # 命令 8 是回合终止符、9 是终局说明行，都不是可执行动作。
    if tokens[0] in (8, 9):
        return []
    return [tokens]


def run(candidate: Path, replay_path: Path, match_id: str, role: str) -> dict[str, object]:
    import sys

    entry_root = _find_entry_root(candidate)
    if str(entry_root) not in sys.path:
        sys.path.insert(0, str(entry_root))

    frames = _load_frames(replay_path)
    strategy = _load_strategy(entry_root)
    player = 0 if role == "P0" else 1

    state = _build_state(frames[0])
    decisions: list[dict[str, object]] = []
    seen_round: int | None = None

    from generals_impact_game.execute import execute_single_command
    from generals_impact_game.gamestate import update_round

    for index in range(1, len(frames)):
        frame = frames[index]
        actor = int(frame.get("Player", -1))
        round_index = int(frame.get("Round", index))

        # 每个回合、每个玩家只探测一次（回放里同一回合会有多行增量）。
        if actor == player and seen_round != round_index:
            seen_round = round_index
            state_id = f"{match_id}:r{round_index:04d}:p{player}"
            supports = [list(_legal_support(state, player))]
            try:
                proposed = strategy(round_index, player, state)
            except Exception:  # noqa: BLE001 - 候选崩溃记空动作并继续
                proposed = []
            actions: list[str] = []
            if isinstance(proposed, Sequence):
                # 逐条累积：发完一条命令后金币/兵力变了，下一条的合法集也变了，
                # 与 antwar2 的 pending 累积同构。
                probe = copy.deepcopy(state)
                for command in proposed:
                    if not isinstance(command, Sequence) or not command:
                        continue
                    tokens = [int(item) for item in command]
                    if tokens[0] == 8:
                        break
                    try:
                        accepted = bool(
                            execute_single_command(player, probe, tokens[0], tokens[1:])
                        )
                    except Exception:  # noqa: BLE001
                        accepted = False
                    if accepted:
                        actions.append(_action_key(tokens[0], tokens[1:]))
                        supports.append(list(_legal_support(probe, player)))
            decisions.append(
                {
                    "state_id": state_id,
                    "actions": actions or ["HOLD"],
                    "legal_supports": supports,
                    "occupancy_id": _occupancy_id(state),
                }
            )

        # 用回放里真实发生的命令推进状态。
        for command in _commands(frame):
            if actor in (0, 1):
                try:
                    execute_single_command(actor, state, command[0], command[1:])
                except Exception:  # noqa: BLE001 - 回放与后端版本差异，跳过该步
                    continue
        if actor == -1:
            # 系统行 = 回合结算。
            try:
                update_round(state)
            except Exception:  # noqa: BLE001
                pass

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

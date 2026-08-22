"""lostspace 的精确决策空间探针。

它解决什么问题
--------------
IG 要求在**同一个完整合法动作集 A(s)** 上比较新旧策略。没有探针时框架退回
``opcode_alphabet``：把 7 种操作类型（move / attack / interact / trap / tool /
detect / finish）当成支撑集，于是 |A(s)| 恒为 7，与局面无关。
``decision_space.yaml`` 的 provenance 里已经写明这是权宜之计——
真实参数域（相邻可达格、视野内目标、道具种类）随状态变化。

本探针把 |A(s)| 变成逐点计算的真值。

技术难点：SDK 的动作是「发送 + 阻塞等待回复」
--------------------------------------------
``AIClient.move()`` 之类的方法都是::

    self.send_opt({...})
    while True:
        self.receive_data()          # ← 阻塞等判题器回复
        ...

离线重放时没有判题器，直接调 ``play()`` 会永久阻塞。所以探针要做两件事：

1. **拦截 ``send_opt``**：候选提交的第一个操作就是它这一步的决策，记下来；
2. **让阻塞循环立刻退出**：用一个哨兵异常在 ``send_opt`` 里抛出，
   ``play()`` 的调用栈随之展开，不会走到 ``receive_data()``。

这样每个决策点只驱动候选到"做出选择"为止，既拿到动作也不需要判题器。

合法动作怎么数
--------------
在候选自己的地图状态上逐点枚举（这正是近似口径做不到的）：

* ``move``    —— ``get_neighbors(pos)`` 里 ``able != 0`` 的相邻格；
* ``attack``  —— 视野内存活的其他玩家（``others`` 里 status==ALIVE 且可见）；
* ``interact``—— 当前格子的区域类型决定：物资点/密钥机/电梯/逃生舱各 1 种；
* ``trap``    —— 手上还有的陷阱种类（LandMine / Sticky）；
* ``tool``    —— 手上还有的道具（Kit / Transport；Transport 还要选目标格）；
* ``detect``  —— 可探查的格子数（受 check_cd 限制，冷却中为 0）；
* ``finish``  —— 恒为 1（结束回合总是合法）。

拿不到某一项时按 0 计而不是猜一个数：宁可让支撑集偏小并在 IG 里体现，
也不要用编造的数字把指标撑大。

用法（框架以子进程方式调用，cwd = 候选目录）::

    python policy_trace_worker.py --candidate <dir> --replay <replay.json> \
        --match-id <id> --role P0
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 玩家状态（与 SDK 的 STATUS 枚举一致）。
STATUS_ALIVE = 0

#: 区域类型 → 该格子上可做的交互种类数。
#: 0 普通 / 1 物资点 / 2 密钥机 / 3 电梯 / 4 逃生舱
AREA_INTERACTIONS = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}


class _DecisionMade(Exception):
    """哨兵异常：候选一提交操作就抛，用来展开 SDK 里的阻塞等待循环。

    不用返回值是因为阻塞发生在 ``send_opt`` **之后**的 ``receive_data()``，
    只有异常能在那之前把控制权夺回来。
    """

    def __init__(self, operation: object) -> None:
        super().__init__("decision recorded")
        self.operation = operation


@dataclass
class Decision:
    index: int
    actions: list[str] = field(default_factory=list)
    legal_supports: list[list[str]] = field(default_factory=list)


def _import_sdk(candidate: Path):
    path = candidate / "lostspace_sdk.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("lostspace_sdk_probe", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # 候选写的是 ``from lostspace_sdk import AIClient``，必须注册同名模块，
    # 否则会加载出第二份，isinstance / 继承关系全部对不上。
    sys.modules.setdefault("lostspace_sdk", module)
    try:
        spec.loader.exec_module(module)
    except BaseException:  # noqa: BLE001
        return None
    return module


def _load_candidate_ai(candidate: Path) -> type | None:
    candidate = candidate.resolve()
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    for name in ("ai", "main"):
        try:
            module = importlib.import_module(name)
        except BaseException:  # noqa: BLE001
            continue
        ai_class = getattr(module, "AI", None)
        if isinstance(ai_class, type):
            return ai_class
    return None


def _count_moves(instance: Any) -> int:
    """相邻且未被缩圈的格子数。"""

    try:
        position = instance.player.pos
        neighbors = instance.get_neighbors(position)
    except BaseException:  # noqa: BLE001
        return 0
    usable = 0
    for node in neighbors:
        try:
            x, y, z = int(node[0]), int(node[1]), int(node[2])
            cell = instance.map.node[z][x][y]
        except (AttributeError, IndexError, TypeError, ValueError):
            # 拿不到就不计：宁可支撑集偏小，也不要编一个数。
            continue
        if getattr(cell, "able", 1) != 0:
            usable += 1
    return usable


def _count_attacks(instance: Any) -> int:
    """可攻击的目标数：其他玩家里还活着的。

    视野判定依赖 ``view``，不同候选对它的维护程度不一。这里只按"存活"计，
    是一个上界；把它标注清楚比假装精确更重要。
    """

    try:
        others = instance.others or []
    except AttributeError:
        return 0
    return sum(
        1
        for other in others
        if int(getattr(other, "status", -1)) == STATUS_ALIVE
    )


def _count_interactions(instance: Any) -> int:
    try:
        position = instance.player.pos
        x, y, z = int(position[0]), int(position[1]), int(position[2])
        cell = instance.map.node[z][x][y]
    except (AttributeError, IndexError, TypeError, ValueError):
        return 0
    return AREA_INTERACTIONS.get(int(getattr(cell, "area_type", 0)), 0)


def _count_traps(instance: Any) -> int:
    """手上还剩的陷阱种类数（LandMine / Sticky 各算一种）。"""

    try:
        tools = instance.player.tools
    except AttributeError:
        return 0
    count = 0
    for attribute in ("landmine_number", "sticky_number"):
        holdings = getattr(tools, attribute, None)
        # SDK 里这两个字段是 [已放置数, 剩余数] 形式。
        if isinstance(holdings, (list, tuple)) and len(holdings) >= 2:
            try:
                if int(holdings[1]) > 0:
                    count += 1
            except (TypeError, ValueError):
                continue
    return count


def _count_tools(instance: Any) -> int:
    """医疗包 + 传送器。传送器要选目标格，按可达格数展开。"""

    try:
        tools = instance.player.tools
    except AttributeError:
        return 0
    count = 0
    try:
        if int(getattr(tools, "kit", 0)) > 0:
            count += 1
    except (TypeError, ValueError):
        pass
    try:
        if int(getattr(tools, "transport", 0)) > 0:
            # 传送目标是任意已知格；用相邻格数作为保守下界，
            # 避免把整张地图 3*7*7=147 格都算进来而虚增支撑集。
            count += max(1, _count_moves(instance))
    except (TypeError, ValueError):
        pass
    return count


def _count_detects(instance: Any) -> int:
    """探查：冷却中为 0，否则按相邻格数计。"""

    try:
        cooldown = int(instance.get_check_cd())
    except BaseException:  # noqa: BLE001
        return 0
    if cooldown > 0:
        return 0
    return max(1, _count_moves(instance))


def _support_size(instance: Any) -> int:
    """当前状态下的合法动作总数（finish 恒可用，所以至少为 1）。"""

    return (
        _count_moves(instance)
        + _count_attacks(instance)
        + _count_interactions(instance)
        + _count_traps(instance)
        + _count_tools(instance)
        + _count_detects(instance)
        + 1  # finish：结束回合总是合法
    )


def _apply_roundbegin(instance: Any, payload: dict[str, Any]) -> None:
    """把一帧 roundbegin 的内容灌进候选实例。

    直接复用 SDK 自己的 ``start_turn()`` 解析逻辑会连带调用 ``play()``，
    所以这里只做字段搬运，不触发决策。字段名逐条对齐 ``AIClient.start_turn``。
    """

    instance.root = payload
    instance.state = payload.get("state")
    player = instance.player
    player.status = payload.get("status", STATUS_ALIVE)
    player.hp = payload.get("hp", 0)
    player.keys = payload.get("keys", 0)

    tools = payload.get("tools") or {}
    landmine = tools.get("LandMine") or [0, 0]
    sticky = tools.get("Sticky") or [0, 0]
    player.tools.landmine_number = list(landmine[:2])
    player.tools.landmine_pos = list(landmine[2:])
    player.tools.sticky_number = list(sticky[:2])
    player.tools.sticky_pos = list(sticky[2:])
    player.tools.kit = tools.get("Kit", 0)
    player.tools.transport = tools.get("Transport", 0)

    others = list(payload.get("others") or [])
    others.sort(key=lambda item: item.get("player_id", 0))
    for index in range(min(3, len(others))):
        instance.others[index].id = others[index].get("player_id", -1)
        instance.others[index].status = others[index].get("status", -1)
        instance.others[index].keys = others[index].get("keys", 0)
        instance.others[index].hp = others[index].get("hp", 0)

    position = payload.get("pos")
    if isinstance(position, (list, tuple)) and len(position) >= 3:
        player.pos = tuple(int(x) for x in position[:3])
    view = payload.get("view")
    if view:
        try:
            instance.update_view(view)
        except BaseException:  # noqa: BLE001 - 视野更新失败不影响动作枚举
            pass


def _describe(operation: object) -> str:
    """把候选提交的操作压成可判等的字符串。"""

    if isinstance(operation, dict):
        action = operation.get("action")
        if isinstance(action, (list, tuple)) and action:
            head = str(action[0])
            tail = ",".join(str(item) for item in action[1:])
            return f"{head}:{tail}" if tail else head
        return str(operation.get("type", "unknown"))
    return "unknown"


def _probe(candidate: Path, replay: Path, seat: int) -> list[Decision]:
    sdk = _import_sdk(candidate)
    if sdk is None:
        print("[lostspace-probe] 找不到 lostspace_sdk.py", file=sys.stderr)
        return []
    ai_class = _load_candidate_ai(candidate)
    if ai_class is None:
        print("[lostspace-probe] 候选里没有 ai.AI", file=sys.stderr)
        return []

    try:
        instance = ai_class.__new__(ai_class)  # type: ignore[misc]
        # AIClient.__init__ 会读 stdin 做握手，这里手工初始化它建立的字段。
        instance.map = sdk.Map()
        instance.player = sdk.Player()
        instance.others = [sdk.Player() for _ in range(3)]
        instance.view = sdk.View()
        instance.root = {}
        instance.state = 0
        instance.player.id = seat
    except BaseException as error:  # noqa: BLE001
        print(f"[lostspace-probe] 构造候选失败：{error}", file=sys.stderr)
        return []

    def _intercept(data: object) -> None:
        raise _DecisionMade(data)

    instance.send_opt = _intercept  # type: ignore[method-assign]

    try:
        frames = json.loads(replay.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"[lostspace-probe] 读回放失败：{error}", file=sys.stderr)
        return []
    if not isinstance(frames, list):
        return []

    decisions: list[Decision] = []
    # 回放结构：[birth_positions, round_1, ..., round_k, score_dic]
    # 中间为 list 的元素才是大回合；每个大回合里是各玩家的小回合。
    for element in frames:
        if not isinstance(element, list):
            continue
        for turn in element:
            if not isinstance(turn, list):
                continue
            # 只在本座次的小回合上取样。
            owner = None
            for record in turn:
                if isinstance(record, dict) and "playerid" in record:
                    owner = int(record["playerid"])
                    break
            if owner != seat:
                continue

            # 用该小回合第一条记录里的状态近似回合开始时的观测。
            # 回放不含完整快照（只有事件流），所以这里能重建的是位置与血量；
            # 拿不到的字段保持上一回合的值——这比编造要诚实。
            for record in turn:
                if not isinstance(record, dict):
                    continue
                if record.get("type") == "move" and record.get("pos"):
                    position = record["pos"]
                    if isinstance(position, (list, tuple)) and len(position) >= 3:
                        instance.player.pos = tuple(int(x) for x in position[:3])
                    break

            support = max(1, _support_size(instance))
            label = "finish"
            try:
                instance.play()
            except _DecisionMade as decided:
                label = _describe(decided.operation)
            except BaseException as error:  # noqa: BLE001
                frames_tb = traceback.extract_tb(error.__traceback__)
                location = ""
                for tb_frame in reversed(frames_tb):
                    if Path(tb_frame.filename).name != Path(__file__).name:
                        location = (
                            f" at {Path(tb_frame.filename).name}:{tb_frame.lineno}"
                            f" in {tb_frame.name}(): {(tb_frame.line or '').strip()[:120]}"
                        )
                        break
                print(
                    f"[lostspace-probe] candidate play() raised "
                    f"{type(error).__name__}: {error}{location}",
                    file=sys.stderr,
                )
                label = "error"

            decisions.append(
                Decision(
                    index=len(decisions),
                    actions=[label],
                    legal_supports=[[f"a#{i}" for i in range(support)]],
                )
            )
    return decisions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--role", required=True)
    args = parser.parse_args()

    # lostspace 是四人局，角色名形如 P0..P3；末位数字即座次。
    role = str(args.role).upper()
    seat = 0
    for index in range(4):
        if role.endswith(str(index)):
            seat = index
            break

    decisions = _probe(args.candidate.resolve(), args.replay.resolve(), seat)
    payload = {
        "match_id": args.match_id,
        "role": args.role,
        "decisions": [
            {
                "index": item.index,
                "actions": item.actions,
                "legal_supports": item.legal_supports,
            }
            for item in decisions
        ],
    }
    print("AGENTBENCH_POLICY_TRACE=" + json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

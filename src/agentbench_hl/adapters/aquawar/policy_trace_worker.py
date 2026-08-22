"""aquawar 的精确决策空间探针。

它解决什么问题
--------------
IG 的定义要求在**同一个完整合法动作集 A(s)** 上比较新旧策略。没有探针时
框架只能退回 ``opcode_alphabet`` 近似——把操作码字母表（Pick/Assert/Action/Null
这几个字符串）当成支撑集，于是 |A(s)| 变成一个与局面无关的常量。
那样算出的 IG 仍然有值，但已经不是它声称度量的那个量（实测系统性偏低约 25%）。

aquawar 比其他游戏好办的地方在于**回放是逐帧完整快照**：每一帧都带
``players[].fight_fish``（双方 4 条鱼的 hp/atk/id/state），不需要像 miracle
那样推演状态。难点只在于把这份快照装回候选能读的 ``Game`` 对象，
并且逐点枚举出真实的合法动作数。

合法动作怎么数
--------------
三个决策阶段的动作空间互不相同（``gamestate`` 见回放）：

* ``2`` Pick —— 从 ``RemainFishs`` 里挑 4 条上场。真实空间是 C(n,4) 再加上
  拟态选项，但候选契约里 pick 只调用一次、且对局早期定型，
  这里按"可选鱼的组合数"计。
* ``3`` Assert —— 断言敌方某个位置是哪条鱼。官方判据：位置越界、或该位置
  身份**已经**暴露（``id != -1``）时必须发 Null。所以合法集是
  ``{(pos, fish_id) | pos 身份未知}`` 再加上一个"不断言"。
* ``4`` Action —— 主体决策。``Action(type, my_pos, enemy_target, friend_target)``：
  - ``type=0`` 普通攻击：出手鱼 × 存活敌方；
  - ``type=1`` 主动技能：出手鱼 × (存活敌方 ∪ {AOE}) ，部分技能还要选友方目标。
  技能是否需要友方目标依赖鱼的 id，这里取**并集上界**：
  对每条存活己方鱼，普攻 |living_enemies| 种，技能 (|living_enemies| + 1) 种
  （+1 是 AOE），需要友方目标的技能再乘 |living_allies|。

这个口径与 ``decision_space.yaml`` 的声明保持一致；宁可给一个**可复现的上界**，
也不要给一个看起来精确、实则依赖未公开技能表的数字。

用法（框架会以子进程方式调用，cwd 设为候选目录）::

    python policy_trace_worker.py --candidate <dir> --replay <replay.json> \
        --match-id <id> --role P0
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import io
import json
import math
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: 每方上场鱼数（官方固定 4）。
FISH_COUNT = 4

#: 鱼的种类编号上界：0..11 是普通鱼，12 是拟态。
FISH_KINDS = 13

#: 回放里的 gamestate 取值。
STATE_PICK = 2
STATE_ASSERT = 3
STATE_ACTION = 4


@dataclass
class Decision:
    """一个决策点：候选做了什么，以及当时的合法动作集有多大。"""

    index: int
    phase: str
    actions: list[str] = field(default_factory=list)
    legal_supports: list[list[str]] = field(default_factory=list)


def _load_candidate_ai(candidate: Path) -> type | None:
    """加载候选的 ``ai.AI``。

    候选契约（CANDIDATE_CONTRACT.md）规定：候选只写 ``ai.py``，
    里面定义 ``class AI(AIClient)``。这里必须把候选目录放到 ``sys.path``
    **首位**，否则会 import 到框架自己的同名模块。
    """

    candidate = candidate.resolve()
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    for name in ("ai", "main"):
        try:
            module = importlib.import_module(name)
        except BaseException:  # noqa: BLE001 - 候选可能在导入期就崩
            continue
        ai_class = getattr(module, "AI", None)
        if isinstance(ai_class, type):
            return ai_class
    return None


def _new_instance(ai_class: type) -> object | None:
    """造一个候选实例，尽量让它自己的 ``__init__`` 跑一遍。

    为什么不直接 ``__new__``：候选常在 ``__init__`` 里设自定义属性
    （``self.turn_count = 0`` 之类），``__new__`` 路径下这些属性不存在，
    候选第一行就 ``AttributeError``，于是每个决策点都退化成空动作。
    miracle 探针正是栽在这里（决策数 1 vs 线协议 80）。

    ``AIClient.__init__`` 只是初始化几个字段、不读 stdin（读帧发生在
    ``run()`` 里），所以通常能直接构造成功。仍然兜底：万一候选自己在
    ``__init__`` 里读输入，就换成一个带 ``.buffer`` 的假 stdin 再试，
    最后才退回 ``__new__``。
    """

    try:
        return ai_class()  # type: ignore[call-arg]
    except BaseException:  # noqa: BLE001
        pass

    # aquawar 的线协议是 4 字节大端长度前缀 + JSON。给一段合法帧，
    # 让可能存在的读取逻辑不至于抛异常。
    payload = json.dumps({"gamestate": STATE_PICK, "RemainFishs": list(range(FISH_KINDS))}).encode()
    frame = len(payload).to_bytes(4, "big") + payload
    original = sys.stdin
    try:
        sys.stdin = io.TextIOWrapper(io.BytesIO(frame * 32), encoding="utf-8", errors="replace")
        try:
            return ai_class()  # type: ignore[call-arg]
        except BaseException:  # noqa: BLE001
            try:
                return ai_class.__new__(ai_class)  # type: ignore[misc]
            except BaseException:  # noqa: BLE001
                return None
    finally:
        sys.stdin = original


def _import_sdk(candidate: Path):
    """加载候选包里的 ``aquawar_sdk``（Fish / Action / Game 都在里面）。"""

    path = candidate / "aquawar_sdk.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("aquawar_sdk_probe", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    # 注册进 sys.modules：候选的 ai.py 里写的是 ``from aquawar_sdk import ...``，
    # 不注册会让它再 import 一份，导致 isinstance(action, Action) 判定失败。
    sys.modules.setdefault("aquawar_sdk", module)
    try:
        spec.loader.exec_module(module)
    except BaseException:  # noqa: BLE001
        return None
    return module


def _build_game(sdk, frame: dict[str, Any], seat: int):
    """把回放的一帧快照装回候选能读的 ``Game``。

    回放里 ``players[seat].fight_fish`` 是**己方**上场鱼的完整信息，
    对手那一侧要按官方视角做遮蔽：敌方鱼只有在 ``is_expose`` 为真时
    ``id`` 才可见，否则是 -1（候选正是靠这个决定要不要断言）。
    """

    game = sdk.Game()
    players = frame.get("players") or []
    if len(players) <= seat:
        return game

    def _fish_list(entry: Any) -> list[dict[str, Any]]:
        rows = (entry or {}).get("fight_fish") or []
        return [row for row in rows if isinstance(row, dict)]

    mine = _fish_list(players[seat])
    theirs = _fish_list(players[1 - seat] if len(players) > 1 else {})

    for index in range(FISH_COUNT):
        if index < len(mine):
            row = mine[index]
            game.my_fish[index] = sdk.Fish(
                id=int(row.get("id", -1)),
                hp=int(row.get("hp", 0)),
                atk=int(row.get("atk", 0)),
            )
        if index < len(theirs):
            row = theirs[index]
            exposed = bool(row.get("is_expose"))
            game.enemy_fish[index] = sdk.Fish(
                # 未暴露的敌方鱼身份对候选不可见，必须遮成 -1，
                # 否则候选会"看见"它本不该知道的信息，决策与真实对局不一致。
                id=int(row.get("id", -1)) if exposed else -1,
                hp=int(row.get("hp", 0)),
                atk=-1,
            )

    game.remain_fish = [int(x) for x in (frame.get("RemainFishs") or [])]
    game.first_mover = int(frame.get("FirstMover", -1))
    game.current_turn = int(frame.get("cur_turn", 0))
    game.raw = frame
    return game


def _support_pick(game) -> int:
    """Pick 阶段的合法动作数：从可选鱼里挑 4 条。

    ``remain_fish`` 为空时退回全部 12 种普通鱼（对局第一帧常见）。
    拟态（12 + 被模仿者）额外提供 |remain| 种变体，这里一并计入。
    """

    pool = list(game.remain_fish) or list(range(FISH_KINDS - 1))
    size = len(pool)
    if size < FISH_COUNT:
        return max(size, 1)
    combinations = math.comb(size, FISH_COUNT)
    # 拟态：上场 12 号并指定模仿对象，等价于在任一组合里替换一条鱼。
    return combinations + size


def _support_assert(game) -> int:
    """Assert 阶段：只能断言**身份未知**的位置，外加"不断言"这一个选项。"""

    unknown = [index for index, fish in enumerate(game.enemy_fish) if fish.id == -1]
    # 每个未知位置可以猜 FISH_KINDS 种身份；再加上 Null（不断言）。
    return len(unknown) * FISH_KINDS + 1


def _support_action(game) -> int:
    """Action 阶段：出手鱼 × 目标的组合数。

    取并集上界（见模块文档）：
      普攻      |living_allies| × |living_enemies|
      技能      |living_allies| × (|living_enemies| + 1)   # +1 是 AOE
      带友方目标 |living_allies| × |living_allies|          # 治疗/护盾类技能
    """

    allies = game.living_allies()
    enemies = game.living_enemies()
    if not allies or not enemies:
        return 1
    normal = len(allies) * len(enemies)
    skill = len(allies) * (len(enemies) + 1)
    friendly = len(allies) * len(allies)
    return normal + skill + friendly


def _describe_action(sdk, action: Any, phase: str) -> str:
    """把候选返回值压成一个可比较的字符串。

    IG 只关心"两个策略在同一状态下是否选了同一个动作"，所以要的是
    **可判等的紧凑表示**，而不是完整对象。
    """

    if phase == "pick":
        try:
            return "pick:" + ",".join(str(int(x)) for x in (action or []))
        except (TypeError, ValueError):
            return "pick:invalid"
    if phase == "assert":
        try:
            position, fish_id = action
            return f"assert:{int(position)}:{int(fish_id)}"
        except (TypeError, ValueError):
            return "assert:null"
    if isinstance(action, sdk.Action):
        return (
            f"act:{int(action.type)}:{int(action.my_pos)}"
            f":{int(action.enemy_target)}:{int(action.friend_target)}"
        )
    return "act:invalid"


def _probe(candidate: Path, replay: Path, seat: int) -> list[Decision]:
    sdk = _import_sdk(candidate)
    if sdk is None:
        print("[aquawar-probe] 找不到 aquawar_sdk.py", file=sys.stderr)
        return []
    ai_class = _load_candidate_ai(candidate)
    if ai_class is None:
        print("[aquawar-probe] 候选里没有 ai.AI", file=sys.stderr)
        return []
    instance = _new_instance(ai_class)
    if instance is None:
        print("[aquawar-probe] 无法构造候选实例", file=sys.stderr)
        return []

    try:
        frames = json.loads(replay.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"[aquawar-probe] 读回放失败：{error}", file=sys.stderr)
        return []
    if not isinstance(frames, list):
        return []

    decisions: list[Decision] = []
    for frame in frames:
        # 契约里写明：帧数组**末元素是 null 哨兵**，必须过滤。
        if not isinstance(frame, dict):
            continue
        # 只在轮到本方决策的帧上取样：cur_turn 标明这一帧是谁在动。
        if int(frame.get("cur_turn", -1)) != seat:
            continue
        state = int(frame.get("gamestate", -1))
        if state == STATE_PICK:
            phase, method, support = "pick", "pick", _support_pick
        elif state == STATE_ASSERT:
            phase, method, support = "assert", "assert_fish", _support_assert
        elif state == STATE_ACTION:
            phase, method, support = "act", "act", _support_action
        else:
            continue

        game = _build_game(sdk, frame, seat)
        # 候选可能把状态挂在实例上（SDK 的 run() 会维护 self.game），补上以防万一。
        try:
            instance.game = game  # type: ignore[attr-defined]
        except AttributeError:
            pass

        try:
            handler = getattr(instance, method)
            action = handler(game)
            label = _describe_action(sdk, action, phase)
        except BaseException as error:  # noqa: BLE001 - 候选崩溃不该让整局作废
            # 报出候选代码里的**精确位置**：只打异常类型完全看不出是哪个字段
            # 出的问题，而这类信息几乎总是直接指向状态重建的缺口。
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
                f"[aquawar-probe] candidate {method}() raised "
                f"{type(error).__name__}: {error}{location}",
                file=sys.stderr,
            )
            label = f"{phase}:error"

        size = max(1, int(support(game)))
        decisions.append(
            Decision(
                index=len(decisions),
                phase=phase,
                actions=[label],
                # 支撑集只需要基数：用占位符表达大小，避免把上万个组合真的列出来。
                legal_supports=[[f"{phase}#{i}" for i in range(size)]],
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

    # 角色名以 game.yaml 为准（P0 / P1）；座次决定读回放的哪一侧。
    seat = 0 if str(args.role).upper().endswith("0") else 1
    decisions = _probe(args.candidate.resolve(), args.replay.resolve(), seat)

    payload = {
        "match_id": args.match_id,
        "role": args.role,
        "decisions": [
            {
                "index": item.index,
                "phase": item.phase,
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

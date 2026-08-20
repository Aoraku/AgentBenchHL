"""antwar：在候选目录内、对一局公开回放做确定性决策复现的 worker。

这份 worker 存在的唯一目的：拿到**规范化、完整的合法动作支持集 A(s)**。
IG 规范要求严格 KL 必须建立在真 A(s) 上；用「操作码字母表」（|A|=10）近似会直接
改变 KL 的尺度，因为闭式解是 (m−u)·ln(m/u)、u = ε/|A|。
实测真实 |A(s)| 的均值只有 7.1（min 2 / max 34），近似在开局系统性**高估**支持集。

为什么状态必须逐回合从回放同步，而**不能**靠 SDK 自己往前推
--------------------------------------------------------
antwar 的对局循环表面上是纯客户端模拟（后端只转发操作，``GameState`` 由选手侧从
``seed + 双方操作`` 自推）。第一版 worker 就是照抄这个循环的，结果保真度自检报出：

    coins 第 149 回合 回放=[14, 104] 重建=[14, 55]

追查到根因（详见 ``games/antwar/known_issues.md``）：**官方 SDK 的经济结算与真后端不一致**。
``downgrade_tower_income()`` 在 ``downgrade_tower()`` 已经改动/移除塔之后才被调用，
拆除等级 1 塔时塔已被 pop，函数走到 ``t is None`` 分支返回 ``-1``（倒扣 1 金），
而后端返还 ``12×2^(N−1)``；P1 在持 3 塔时拆一座 BASIC，后端 +48、SDK −1，差值正是 49。

这个 bug 在官方公开 SDK 里逐字节存在，94 名人类选手用的都是它，所以**不能修**
（修了就改变了竞赛条件）。但它意味着客户端模拟器**不是**后端的忠实复制品，
误差会逐回合累积。因此这里：

* **状态**：每回合从回放的 ``round_state`` 同步 —— 回放由真后端写出，是唯一可信来源；
* **SDK**：只用来做**合法性判定**（``is_operation_valid``，纯 dry-run）与动作枚举。

⚠️ ``round_state.towers`` 是**增量**（官方回放指南原文：「只包含回合内新建/等级状态
发生变化的塔」），所以必须按 id 跨回合累积，不能当成全量快照。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

#: 超级武器的操作码（21–24），枚举时对全地图格子展开。
_WEAPON_OPCODES = (21, 22, 23, 24)


def _action_key(operation: object) -> str:
    """动作的规范化标识。用官方 ``Operation.dump()``，与评测机交换的格式一致。"""

    return operation.dump().replace(" ", ":")  # type: ignore[attr-defined]


def _map_cells() -> tuple[tuple[int, int], ...]:
    """地图内所有格子（``distance(coord, (9,9)) <= 9``）。"""

    from antwar.coord import Coord, is_in_map

    return tuple(
        (x, y) for x in range(19) for y in range(19) if is_in_map(Coord(x, y))
    )


def _buildable_cells(player: int) -> tuple[tuple[int, int], ...]:
    """该玩家可建塔的格子：**只有自己的高台**（官方硬约束，非近似）。"""

    from antwar.coord import Coord, is_player_highland

    return tuple(
        (x, y)
        for x in range(19)
        for y in range(19)
        if is_player_highland(Coord(x, y), player)
    )


def _is_legal(state: object, player: int, operation: object) -> bool:
    """官方合法性判定，且**判定自身崩溃时算非法**。

    为什么需要这层保护：候选包自带的 SDK 里 ``is_operation_valid`` 在少数边界
    状态上会抛异常而不是返回 False。实测最典型的一例是等级已满时再试升级：

        antwar/gamedata.py  upgrade_cost():  return [200, 250][level]
        # level == 2（已满级）⇒ IndexError

    那本该是"这个动作非法"，而 SDK 让它变成了崩溃。枚举支持集必须把所有候选
    动作都问一遍，于是一个格子的崩溃会掀翻整局探针——实测 antwar 有 11 轮
    （it18-20、22-28）因此完全拿不到精确 |A(s)|，只能回落到字母表近似。

    把异常判为非法在语义上也是对的：一个连合法性都算不出来的操作，选手提交上去
    同样会被后端拒绝。计数由调用方汇总上报，绝不静默。
    """

    try:
        return bool(state.is_operation_valid(player, operation))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - SDK 边界 bug，视为非法并由上层计数
        return False


def _legal_support(
    state: object,
    player: int,
    *,
    map_cells: Sequence[tuple[int, int]],
    buildable: Sequence[tuple[int, int]],
) -> tuple[str, ...]:
    """枚举当前状态下该玩家的**全部**合法操作，外加 ``HOLD``（结束本回合）。

    ``HOLD`` 必须在支持集里：一个回合内提交 0 个操作永远合法，它是决策空间里
    真实存在的一个动作。漏掉它会让 |A(s)| 系统性偏小 1。

    合法性一律交给官方 ``is_operation_valid``（内部就是 ``apply_operation(dry_run=True)``，
    纯检查、不改状态），我们只负责把候选动作枚举完整。判定崩溃按非法处理，
    理由见 ``_is_legal``。
    """

    from antwar.gamedata import TowerType, can_tower_upgrade_to
    from antwar.protocol import Operation, OperationType

    candidates: list[object] = [
        Operation(OperationType.BUILD_TOWER, x, y) for x, y in buildable
    ]
    for tower in state.towers:  # type: ignore[attr-defined]
        if tower.player != player:
            continue
        candidates.extend(
            Operation(OperationType.UPGRADE_TOWER, tower.id, int(target))
            for target in TowerType
            if can_tower_upgrade_to(tower.type, target)
        )
        candidates.append(Operation(OperationType.DOWNGRADE_TOWER, tower.id))
    for opcode in _WEAPON_OPCODES:
        operation_type = OperationType(opcode)
        candidates.extend(Operation(operation_type, x, y) for x, y in map_cells)
    candidates.extend(
        (
            Operation(OperationType.UPGRADE_GENERATE_SPEED),
            Operation(OperationType.UPGRADE_ANT_MAXHP),
        )
    )
    legal = {"HOLD"}
    legal.update(
        _action_key(item) for item in candidates if _is_legal(state, player, item)
    )
    return tuple(sorted(legal))


def _occupancy_id(state: object, round_index: int) -> str:
    """状态访问分布用的规范状态指纹。

    只用**公开可观测**的量，且顺序固定（塔/蚂蚁按 id 排序），
    这样同一局面在两个版本之间必然得到同一个 id —— occupancy_shift 才有意义。
    """

    value = {
        "round": round_index,
        "towers": sorted(
            (t.id, t.player, t.coord.x, t.coord.y, int(t.type))
            for t in state.towers  # type: ignore[attr-defined]
        ),
        "ants": sorted(
            (a.id, a.player, a.coord.x, a.coord.y, a.hp, int(a.state))
            for a in state.ants  # type: ignore[attr-defined]
        ),
        # ⚠️ 官方字段名是 `coin` / `hp`（不是 coins / bases）。
        "coin": list(state.coin),  # type: ignore[attr-defined]
        "hp": list(state.hp),  # type: ignore[attr-defined]
        "gen_speed_lv": list(state.gen_speed_lv),  # type: ignore[attr-defined]
        "ant_maxhp_lv": list(state.ant_maxhp_lv),  # type: ignore[attr-defined]
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decision(
    agent: object,
    state: object,
    player: int,
    state_id: str,
    *,
    map_cells: Sequence[tuple[int, int]],
    buildable: Sequence[tuple[int, int]],
) -> dict[str, object]:
    """在一个决策点上记录：候选选了什么、当时的合法集是什么。

    **候选的决策只被观测，绝不落子。**
    这是「冻结重放」的核心，也是一个踩过的坑：最初的实现把候选决定的操作真的应用到了
    状态上，结果轨迹立刻偏离录像——保真度自检在第 149 回合报出 P1 金币 回放=104 /
    重建=55。原因是候选的反事实操作改变了蚂蚁流，进而改变击杀与金币收入，误差逐回合放大。
    更要紧的是：两个版本一旦各自跑到不同轨迹上，KL 就不再建立在**共享的**决策上下文
    z 上，IG 规范要求的配对比较直接失效。

    所以状态只由**录像里记录的双方操作**推进（见 ``run``），这里对状态的任何改动都发生在
    深拷贝上、随即丢弃。一个回合可提交多个操作，因此支持集**逐个原子动作**记录：
    第 i 个原子动作面对的合法集，是在前 i−1 个已被接受之后的（拷贝）状态上枚举的，
    与官方 ``try_apply_our_op`` 逐个应用的语义一致；最后再记一次，
    对应「决定不再提交」这个 HOLD 决策所面对的合法集。
    """

    import copy

    occupancy = _occupancy_id(state, int(getattr(state, "round", 0)))
    proposed = agent.decide(player, state)  # type: ignore[attr-defined]
    if not isinstance(proposed, list):
        raise TypeError("AI.decide 必须返回 list[Operation]")
    # 在副本上推进，真状态一个字节都不动。
    scratch = copy.deepcopy(state)
    accepted: list[str] = []
    supports: list[tuple[str, ...]] = []
    for operation in proposed:
        support = _legal_support(scratch, player, map_cells=map_cells, buildable=buildable)
        # 官方语义：非法操作被丢弃，后续操作继续尝试。
        if not scratch.apply_operation(player, operation):  # type: ignore[attr-defined]
            continue
        supports.append(support)
        accepted.append(_action_key(operation))
    supports.append(_legal_support(scratch, player, map_cells=map_cells, buildable=buildable))
    return {
        "state_id": state_id,
        "actions": accepted,
        "legal_supports": [list(item) for item in supports],
        "occupancy_id": occupancy,
    }


def _sync_from_replay(state: object, record: Mapping[str, object], towers: dict[int, dict]) -> None:
    """把回放记录的权威 ``round_state`` 写回客户端状态。

    回放由真后端产出，是唯一可信的状态来源。这一步把 SDK 经济 bug 造成的累积误差
    每回合清零，让后续的合法性判定建立在**裁判看到的**状态上而不是选手侧的错账上。

    ⚠️ ``towers`` 是增量字段，必须按 id 跨回合累积（官方回放指南原文：
    「只包含回合内新建/等级状态发生变化的塔」）。把单回合的 towers 当全量快照，
    会得出「对手只有 1~2 座塔」的荒谬结论，进而让合法集里凭空多出一堆建塔位。
    """

    from antwar.coord import Coord
    from antwar.gamedata import Ant, AntState, Tower, TowerType

    raw = record.get("round_state")
    if not isinstance(raw, Mapping):
        return

    for key, target in (("coins", "coin"), ("camps", "hp"), ("speedLv", "gen_speed_lv"),
                        ("anthpLv", "ant_maxhp_lv")):
        value = raw.get(key)
        if isinstance(value, list) and len(value) >= 2:
            setattr(state, target, [int(value[0]), int(value[1])])

    # 塔：增量累积。``type == -1`` 是**拆除哨兵**（实测确认：一局里 3 条 -1 记录，
    # 对应 P1 的 3 次拆塔，且没有任何 id 在哨兵之后重新出现；塔数对账也吻合：
    # P0 建 3 + P1 建 6 − 拆 3 = 存活 6）。等级变化则按 id 覆盖。
    # 不认这个哨兵会导致 TowerType(-1) 直接抛错，或（若容错跳过）让已拆的塔永久留在
    # 合法集里 —— 后者更糟：它会凭空造出不存在的升级/降级动作。
    for entry in raw.get("towers", []) or []:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), int):
            continue
        tower_id = int(entry["id"])
        if int(entry.get("type", 0)) == -1:
            towers.pop(tower_id, None)
            continue
        towers[tower_id] = dict(entry)
    rebuilt_towers = []
    for entry in towers.values():
        position = entry.get("pos")
        if not isinstance(position, Mapping):
            continue
        # 字段顺序照抄官方 dataclass：Tower(id, player, coord, type, cd)
        rebuilt_towers.append(
            Tower(
                int(entry["id"]),
                int(entry["player"]),
                Coord(int(position["x"]), int(position["y"])),
                TowerType(int(entry["type"])),
                int(entry.get("cd", 0)),
            )
        )
    state.towers = rebuilt_towers  # type: ignore[attr-defined]

    # 蚂蚁：回放是全量记录，只保留存活的（status==0）。
    rebuilt_ants = []
    for entry in raw.get("ants", []) or []:
        if not isinstance(entry, Mapping) or entry.get("status") != 0:
            continue
        position = entry.get("pos")
        if not isinstance(position, Mapping):
            continue
        hp = int(entry["hp"])
        # 字段顺序照抄官方 dataclass：
        # Ant(id, player, hp, maxhp, coord, level, age, evasion_count, state, path)
        # ⚠️ 回放不记录 maxhp / evasion_count：maxhp 用当前 hp 兜底（只影响治疗类判断，
        # 不影响本 worker 关心的合法性判定），evasion_count 记 0。
        # 这两个是**已知的近似**，写在这里而不是悄悄填 0 就当没事。
        rebuilt_ants.append(
            Ant(
                int(entry["id"]),
                int(entry["player"]),
                hp,
                max(hp, int(entry.get("maxhp", hp))),
                Coord(int(position["x"]), int(position["y"])),
                int(entry.get("level", 0)),
                int(entry.get("age", 0)),
                int(entry.get("evasion_count", 0)),
                AntState(int(entry.get("status", 0))),
            )
        )
    state.ants = rebuilt_ants  # type: ignore[attr-defined]

    phero = raw.get("pheromone")
    if isinstance(phero, list) and len(phero) == 2:
        state.phero = phero  # type: ignore[attr-defined]


def run(
    candidate: Path,
    replay_path: Path,
    match_id: str,
    role: str,
    *,
    max_rounds: int | None = None,
) -> dict[str, object]:
    """在一局回放上复现候选的决策与合法集。

    ``max_rounds`` 只用于**成本量测**：合法集枚举是这里唯一的重活
    （每个决策点要对全地图×4 超武 + 自己高台建塔 + 每塔升级树 做 dry-run），
    正式测量必须跑完整局，否则决策点集合被截断、KL 的分母就变了。
    截断时会在结果里写明 ``truncated``，不允许静默当成完整轨迹。
    """

    import sys
    import time

    sys.path.insert(0, str(candidate))
    from ai import AI
    from antwar.gamestate import GameState

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if not isinstance(replay, list) or not replay:
        raise ValueError("antwar 回放应当是非空 JSON 数组")
    first = replay[0]
    if not isinstance(first, Mapping):
        raise ValueError("回放首帧格式非法")
    seed = int(first.get("seed", 0))
    player = 0 if role == "P0" else 1

    agent = AI()
    state = GameState()
    state.init_with_seed(seed)
    map_cells = _map_cells()
    buildable = _buildable_cells(player)

    total = len(replay)
    limit = total if max_rounds is None else min(total, max_rounds)
    decisions: list[dict[str, object]] = []
    towers: dict[int, dict] = {}
    started = time.monotonic()
    for index, record in enumerate(replay[:limit]):
        if not isinstance(record, Mapping):
            raise ValueError("回放记录格式非法")
        # 关键：先把权威状态同步进来，再让候选在**裁判看到的**状态上决策。
        # 不再调用 simulate_next_round()——官方 SDK 的经济结算与后端不一致
        # （见 games/antwar/known_issues.md），自推会逐回合累积误差。
        _sync_from_replay(state, record, towers)
        state.round = index  # type: ignore[attr-defined]
        if player == 0:
            decisions.append(
                _decision(
                    agent, state, 0, f"{match_id}:r{index:04d}:p0",
                    map_cells=map_cells, buildable=buildable,
                )
            )
        else:
            decisions.append(
                _decision(
                    agent, state, 1, f"{match_id}:r{index:04d}:p1",
                    map_cells=map_cells, buildable=buildable,
                )
            )
    elapsed = time.monotonic() - started
    supports = [len(s) for d in decisions for s in d["legal_supports"]]  # type: ignore[union-attr]
    return {
        "match_id": match_id,
        "role": role,
        "decisions": decisions,
        # 量测与诚实标注：截断的轨迹绝不能被当成完整轨迹使用。
        "truncated": limit < total,
        "rounds_replayed": limit,
        "rounds_total": total,
        "elapsed_s": round(elapsed, 3),
        "support_size_min": min(supports) if supports else None,
        "support_size_max": max(supports) if supports else None,
        "support_size_mean": round(sum(supports) / len(supports), 1) if supports else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--role", choices=("P0", "P1"), required=True)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="只跑前 N 回合（**仅用于成本量测**，正式测量必须跑完整局）",
    )
    arguments = parser.parse_args()
    result = run(
        arguments.candidate.resolve(),
        arguments.replay.resolve(),
        arguments.match_id,
        arguments.role,
        max_rounds=arguments.max_rounds,
    )
    print("AGENTBENCH_POLICY_TRACE=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

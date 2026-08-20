"""Candidate-local worker for deterministic decisions on one public replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def _operation(raw: Mapping[str, object]):
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    operation_type = OperationType(int(raw["type"]))
    if int(operation_type) in {11, 21, 22, 23, 24}:
        position = raw.get("pos")
        if not isinstance(position, Mapping):
            raise ValueError("position operation has no pos object")
        return Operation(operation_type, int(position["x"]), int(position["y"]))
    if int(operation_type) == 12:
        return Operation(operation_type, int(raw["id"]), int(raw["args"]))
    if int(operation_type) == 13:
        return Operation(operation_type, int(raw["id"]))
    return Operation(operation_type)


def _operations(record: Mapping[str, object], player: int) -> list[object]:
    value = record.get(f"op{player}", [])
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"replay op{player} is malformed")
    return [_operation(item) for item in value]


def _round_number(record: Mapping[str, object], fallback: int) -> int:
    raw = record.get("round_state")
    if isinstance(raw, Mapping):
        value = raw.get("round")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return fallback


def _public_state(
    record: Mapping[str, object],
    tower_map: dict[int, Mapping[str, object]],
    *,
    fallback_round: int,
):
    from SDK.backend.engine import PublicRoundState

    raw = record.get("round_state")
    if not isinstance(raw, Mapping):
        raise ValueError("replay record has no round_state")
    towers = raw.get("towers", [])
    if not isinstance(towers, list):
        raise ValueError("round_state towers must be a list")
    for item in towers:
        if not isinstance(item, Mapping):
            raise ValueError("round_state tower is malformed")
        tower_id = int(item["id"])
        if int(item["type"]) == -1:
            tower_map.pop(tower_id, None)
        else:
            tower_map[tower_id] = item
    tower_rows = []
    for tower_id, item in sorted(tower_map.items()):
        position = item["pos"]
        if not isinstance(position, Mapping):
            raise ValueError("tower position is malformed")
        tower_rows.append(
            (
                tower_id,
                int(item["player"]),
                int(position["x"]),
                int(position["y"]),
                int(item["type"]),
                int(item.get("cd", 0)),
                int(item.get("hp", -1)),
            )
        )
    ants = raw.get("ants", [])
    if not isinstance(ants, list):
        raise ValueError("round_state ants must be a list")
    ant_rows = []
    for item in ants:
        if not isinstance(item, Mapping):
            raise ValueError("round_state ant is malformed")
        position = item["pos"]
        if not isinstance(position, Mapping):
            raise ValueError("ant position is malformed")
        ant_rows.append(
            (
                int(item["id"]),
                int(item["player"]),
                int(position["x"]),
                int(position["y"]),
                int(item["hp"]),
                int(item["level"]),
                int(item.get("age", 0)),
                int(item["status"]),
                int(item.get("behavior", 0)),
                int(item.get("kind", 0)),
            )
        )
    effects = raw.get("activeEffects", [])
    if not isinstance(effects, list):
        raise ValueError("round_state activeEffects must be a list")
    effect_rows = []
    for item in effects:
        if not isinstance(item, Mapping):
            raise ValueError("active effect is malformed")
        effect_rows.append(
            (
                int(item["type"]),
                int(item["player"]),
                int(item.get("x", item.get("pos", {}).get("x", -1))),
                int(item.get("y", item.get("pos", {}).get("y", -1))),
                int(item.get("duration", 0)),
            )
        )
    return PublicRoundState(
        round_index=_round_number(record, fallback_round),
        towers=tower_rows,
        ants=ant_rows,
        coins=tuple(int(item) for item in raw.get("coins", [0, 0])[:2]),
        camps_hp=tuple(int(item) for item in raw.get("camps", [0, 0])[:2]),
        speed_lv=tuple(int(item) for item in raw.get("speedLv", [0, 0])[:2]),
        anthp_lv=tuple(int(item) for item in raw.get("anthpLv", [0, 0])[:2]),
        weapon_cooldowns=tuple(
            tuple(int(item) for item in row) for row in raw.get("weaponCooldowns", [])
        ),
        active_effects=effect_rows,
    )


def _action_key(operation: object) -> str:
    return ":".join(str(int(token)) for token in operation.to_protocol_tokens())


def _is_legal(state: object, player: int, operation: object, pending: list[object]) -> bool:
    """官方合法性判定，且**判定自身崩溃时算非法**。

    候选包自带的 SDK 在少数边界状态上会让 ``can_apply_operation`` 抛异常而不是
    返回 False（antwar 侧的同类问题是满级时 ``[200,250][2]`` 越界）。枚举支持集
    必须把每个候选动作都问一遍，所以一个格子的崩溃会掀翻整局探针，让这一轮
    退回字母表近似。把异常判为非法在语义上也是对的：连合法性都算不出来的操作，
    提交上去同样会被后端拒绝。
    """

    try:
        return bool(state.can_apply_operation(player, operation, pending))
    except Exception:  # noqa: BLE001 - SDK 边界 bug，视为非法
        return False


def _legal_support(state: object, player: int, pending: list[object]) -> tuple[str, ...]:
    from SDK.backend.model import Operation
    from SDK.utils.constants import TOWER_UPGRADE_TREE, VALID_CELLS, OperationType

    candidates = [Operation(OperationType.BUILD_TOWER, x, y) for x, y in VALID_CELLS]
    for tower in state.towers_of(player):
        candidates.extend(
            Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
            for target in TOWER_UPGRADE_TREE.get(tower.tower_type, ())
        )
        candidates.append(Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id))
    for operation_type in (
        OperationType.USE_LIGHTNING_STORM,
        OperationType.USE_EMP_BLASTER,
        OperationType.USE_DEFLECTOR,
        OperationType.USE_EMERGENCY_EVASION,
    ):
        candidates.extend(Operation(operation_type, x, y) for x, y in VALID_CELLS)
    candidates.extend(
        (
            Operation(OperationType.UPGRADE_GENERATION_SPEED),
            Operation(OperationType.UPGRADE_GENERATED_ANT),
        )
    )
    legal = {"HOLD"}
    legal.update(
        _action_key(item) for item in candidates if _is_legal(state, player, item, pending)
    )
    return tuple(sorted(legal))


def _occupancy_id(state: object) -> str:
    public = state.to_public_round_state()
    value = {
        "round": public.round_index,
        "towers": public.towers,
        "ants": public.ants,
        "coins": public.coins,
        "camps_hp": public.camps_hp,
        "speed_lv": public.speed_lv,
        "anthp_lv": public.anthp_lv,
        "weapon_cooldowns": public.weapon_cooldowns,
        "active_effects": public.active_effects,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decision(agent: object, state: object, player: int, state_id: str) -> dict[str, object]:
    proposed = agent.choose_operations(state, player)
    if not isinstance(proposed, list):
        raise TypeError("AI.choose_operations must return a list")
    accepted: list[object] = []
    supports: list[tuple[str, ...]] = []
    for operation in proposed:
        if state.can_apply_operation(player, operation, accepted):
            supports.append(_legal_support(state, player, accepted))
            accepted.append(operation)
    supports.append(_legal_support(state, player, accepted))
    agent.on_self_operations(accepted)
    return {
        "state_id": state_id,
        "actions": [_action_key(item) for item in accepted],
        "legal_supports": [list(item) for item in supports],
        "occupancy_id": _occupancy_id(state),
    }


def run(candidate: Path, replay_path: Path, match_id: str, role: str) -> dict[str, object]:
    import sys

    sys.path.insert(0, str(candidate))
    from ai import AI
    from SDK.backend.state import create_python_backend_state

    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if not isinstance(replay, list) or len(replay) < 2:
        raise ValueError("policy probe replay must contain at least one transition")
    first = replay[0]
    if not isinstance(first, Mapping):
        raise ValueError("policy probe replay has malformed initial record")
    seed = int(first.get("seed", 0))
    player = 0 if role == "P0" else 1
    agent = AI()
    agent.on_match_start(player, seed)
    state = create_python_backend_state(seed=seed)
    tower_map: dict[int, Mapping[str, object]] = {}
    state.sync_public_round_state(_public_state(first, tower_map, fallback_round=0))
    decisions: list[dict[str, object]] = []
    for index in range(1, len(replay)):
        record = replay[index]
        if not isinstance(record, Mapping):
            raise ValueError("policy probe replay record is malformed")
        round_index = _round_number(record, index)
        operations0 = _operations(record, 0)
        operations1 = _operations(record, 1)
        if player == 0:
            decisions.append(_decision(agent, state, 0, f"{match_id}:r{round_index:04d}:p0"))
        state.apply_operation_list(0, operations0)
        if player == 1:
            agent.on_opponent_operations(operations0)
            decisions.append(_decision(agent, state, 1, f"{match_id}:r{round_index:04d}:p1"))
        state.apply_operation_list(1, operations1)
        if player == 0:
            agent.on_opponent_operations(operations1)
        state.advance_round()
        public = _public_state(record, tower_map, fallback_round=index)
        state.sync_public_round_state(public)
        agent.on_round_state(public)
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

"""Ground official AntWar2 replay numbers in public-state semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

_ACTION_NAMES = {
    11: "BUILD_TOWER",
    12: "UPGRADE_TOWER",
    13: "DOWNGRADE_TOWER",
    21: "USE_LIGHTNING_STORM",
    22: "USE_EMP_BLASTER",
    23: "USE_DEFLECTOR",
    24: "USE_EMERGENCY_EVASION",
    31: "UPGRADE_GENERATION_SPEED",
    32: "UPGRADE_GENERATED_ANT",
}
_WEAPONS = frozenset({21, 22, 23, 24})


def _pair(value: object, default: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        return default
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return default
    return float(value[0]), float(value[1])


@dataclass(frozen=True)
class CanonicalFrame:
    state_id: str
    round_index: int
    base_hp: tuple[float, float]
    generation_level: tuple[float, float]
    ant_level: tuple[float, float]
    coins: tuple[float, float]
    weapon_cooldowns: tuple[tuple[int, ...], tuple[int, ...]]
    towers: tuple[Mapping[str, object], ...]
    ants: tuple[Mapping[str, object], ...]
    active_effects: tuple[Mapping[str, object], ...]
    winner: int | None


@dataclass(frozen=True)
class AtomicEvent:
    state_id: str
    round_index: int
    player: str
    operation_index: int
    action: str
    code: int | None
    arguments: tuple[int, ...]
    before_frame_id: str
    after_frame_id: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "after_frame_id": self.after_frame_id,
            "arguments": list(self.arguments),
            "before_frame_id": self.before_frame_id,
            "code": self.code,
            "operation_index": self.operation_index,
            "player": self.player,
            "round": self.round_index,
            "state_id": self.state_id,
            "text": self.text,
        }


@dataclass(frozen=True)
class CriticalWindow:
    start_state_id: str
    end_state_id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "start_state_id": self.start_state_id,
            "end_state_id": self.end_state_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class StrategicClaim:
    text: str
    evidence_state_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "evidence_state_ids": list(self.evidence_state_ids)}


@dataclass(frozen=True)
class ReplayReport:
    match_id: str
    frames: tuple[CanonicalFrame, ...]
    timeline: tuple[AtomicEvent, ...]
    critical_windows: tuple[CriticalWindow, ...]
    strategic_claims: tuple[StrategicClaim, ...]
    metrics: Mapping[str, object]
    winner: str | None
    narrative: str
    frame_by_id: Mapping[str, CanonicalFrame]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "frame_by_id", MappingProxyType(dict(self.frame_by_id)))

    def window(self, start_state_id: str, end_state_id: str) -> tuple[AtomicEvent, ...]:
        state_ids = [event.state_id for event in self.timeline]
        try:
            start = state_ids.index(start_state_id)
            end = len(state_ids) - 1 - state_ids[::-1].index(end_state_id)
        except ValueError as exc:
            raise ValueError("replay window references an unknown state_id") from exc
        if end < start:
            raise ValueError("replay window end precedes start")
        return self.timeline[start : end + 1]


def _round_number(record: Mapping[str, object], fallback: int) -> int:
    state = record.get("round_state")
    if isinstance(state, Mapping):
        value = state.get("round")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return fallback


def _normalize_cooldowns(value: object) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not isinstance(value, list) or len(value) != 2:
        return (), ()
    result: list[tuple[int, ...]] = []
    for player in value:
        if not isinstance(player, list):
            result.append(())
        else:
            result.append(tuple(int(item) for item in player))
    return result[0], result[1]


def _normalize_frames(
    replay: Sequence[Mapping[str, object]],
    match_id: str,
) -> tuple[CanonicalFrame, ...]:
    tower_state: dict[int, dict[str, object]] = {}
    frames: list[CanonicalFrame] = []
    for index, record in enumerate(replay):
        state = record.get("round_state")
        if not isinstance(state, Mapping):
            raise ValueError(f"replay record {index} has no public round_state")
        round_index = _round_number(record, index)
        raw_towers = state.get("towers", [])
        if not isinstance(raw_towers, list):
            raise ValueError(f"replay round {round_index} has invalid tower delta")
        for raw_tower in raw_towers:
            if not isinstance(raw_tower, Mapping) or not isinstance(raw_tower.get("id"), int):
                raise ValueError(f"replay round {round_index} has malformed tower")
            tower_id = int(raw_tower["id"])
            if raw_tower.get("type") == -1:
                tower_state.pop(tower_id, None)
            else:
                tower_state[tower_id] = dict(raw_tower)
        ants = state.get("ants", [])
        effects = state.get("activeEffects", [])
        if not isinstance(ants, list) or not isinstance(effects, list):
            raise ValueError(f"replay round {round_index} has invalid public entities")
        winner_value = state.get("winner")
        winner = winner_value if winner_value in {0, 1} else None
        frames.append(
            CanonicalFrame(
                state_id=f"{match_id}:r{round_index:04d}:state",
                round_index=round_index,
                base_hp=_pair(state.get("camps"), (50.0, 50.0)),
                generation_level=_pair(state.get("speedLv"), (0.0, 0.0)),
                ant_level=_pair(state.get("anthpLv"), (0.0, 0.0)),
                coins=_pair(state.get("coins"), (0.0, 0.0)),
                weapon_cooldowns=_normalize_cooldowns(state.get("weaponCooldowns")),
                towers=tuple(dict(item) for _, item in sorted(tower_state.items())),
                ants=tuple(dict(item) for item in ants if isinstance(item, Mapping)),
                active_effects=tuple(dict(item) for item in effects if isinstance(item, Mapping)),
                winner=winner,
            )
        )
    return tuple(frames)


def _operation_arguments(operation: Mapping[str, object], code: int) -> tuple[int, ...]:
    if code in {11, 21, 22, 23, 24}:
        position = operation.get("pos")
        if isinstance(position, Mapping):
            return int(position.get("x", -1)), int(position.get("y", -1))
        return -1, -1
    if code == 12:
        return int(operation.get("id", -1)), int(operation.get("args", -1))
    if code == 13:
        return (int(operation.get("id", -1)),)
    return ()


def _event_text(
    *,
    player: str,
    action: str,
    arguments: tuple[int, ...],
    before: CanonicalFrame,
) -> str:
    coins = int(before.coins[int(player[1])])
    base_hp = tuple(int(value) for value in before.base_hp)
    if action == "HOLD":
        fact = "未提交操作（HOLD）"
    elif action == "BUILD_TOWER":
        fact = f"在 ({arguments[0]},{arguments[1]}) 建造基础塔"
    elif action == "UPGRADE_TOWER":
        fact = f"将塔 id={arguments[0]} 升级为类型 {arguments[1]}"
    elif action == "DOWNGRADE_TOWER":
        fact = f"降级或拆除塔 id={arguments[0]}"
    elif action == "USE_LIGHTNING_STORM":
        fact = f"在 ({arguments[0]},{arguments[1]}) 使用闪电风暴"
    elif action == "USE_EMP_BLASTER":
        fact = f"在 ({arguments[0]},{arguments[1]}) 使用 EMP"
    elif action == "USE_DEFLECTOR":
        fact = f"在 ({arguments[0]},{arguments[1]}) 使用偏转器"
    elif action == "USE_EMERGENCY_EVASION":
        fact = f"在 ({arguments[0]},{arguments[1]}) 使用紧急规避"
    elif action == "UPGRADE_GENERATION_SPEED":
        fact = "升级出兵速度"
    elif action == "UPGRADE_GENERATED_ANT":
        fact = "升级生成蚂蚁生命"
    else:
        fact = f"提交未识别操作 {action}，参数={list(arguments)}"
    return f"{player} {fact}；动作前金币={coins}，双方基地HP={base_hp}"


def _decode_timeline(
    replay: Sequence[Mapping[str, object]],
    frames: tuple[CanonicalFrame, ...],
    match_id: str,
) -> tuple[AtomicEvent, ...]:
    events: list[AtomicEvent] = []
    for index in range(1, len(replay)):
        record = replay[index]
        round_index = frames[index].round_index
        before = frames[index - 1]
        after = frames[index]
        for player_number in (0, 1):
            player = f"P{player_number}"
            state_id = f"{match_id}:r{round_index:04d}:p{player_number}"
            raw_operations = record.get(f"op{player_number}", [])
            if not isinstance(raw_operations, list):
                raise ValueError(f"{state_id} operations are not a list")
            operations: list[Mapping[str, object] | None] = (
                list(raw_operations) if raw_operations else [None]
            )
            for operation_index, operation in enumerate(operations):
                if operation is None:
                    code = None
                    action = "HOLD"
                    arguments: tuple[int, ...] = ()
                else:
                    if not isinstance(operation, Mapping):
                        raise ValueError(f"{state_id} contains malformed operation")
                    raw_code = operation.get("type")
                    if isinstance(raw_code, bool) or not isinstance(raw_code, int):
                        raise ValueError(f"{state_id} operation has invalid type")
                    code = raw_code
                    action = _ACTION_NAMES.get(code, f"UNKNOWN_{code}")
                    arguments = _operation_arguments(operation, code)
                events.append(
                    AtomicEvent(
                        state_id=state_id,
                        round_index=round_index,
                        player=player,
                        operation_index=operation_index,
                        action=action,
                        code=code,
                        arguments=arguments,
                        before_frame_id=before.state_id,
                        after_frame_id=after.state_id,
                        text=_event_text(
                            player=player,
                            action=action,
                            arguments=arguments,
                            before=before,
                        ),
                    )
                )
    return tuple(events)


def _derive(
    timeline: tuple[AtomicEvent, ...],
    frames: tuple[CanonicalFrame, ...],
) -> tuple[dict[str, object], tuple[StrategicClaim, ...], tuple[CriticalWindow, ...]]:
    first_action: dict[str, int | None] = {"P0": None, "P1": None}
    first_weapon: dict[str, int | None] = {"P0": None, "P1": None}
    build_counts = {"P0": 0, "P1": 0}
    downgrade_counts = {"P0": 0, "P1": 0}
    last_downgrade: dict[str, int | None] = {"P0": None, "P1": None}
    downgrade_to_weapon_delay: dict[str, int | None] = {"P0": None, "P1": None}
    weapon_targets: dict[str, set[tuple[int, ...]]] = {"P0": set(), "P1": set()}
    idle_resource_rounds: dict[str, list[int]] = {"P0": [], "P1": []}
    claims: list[StrategicClaim] = []
    windows: list[CriticalWindow] = []
    for event in timeline:
        if event.action != "HOLD" and first_action[event.player] is None:
            first_action[event.player] = event.round_index
            claims.append(
                StrategicClaim(
                    f"{event.player} 的首次公开操作发生在第 "
                    f"{event.round_index} 轮：{event.action}。",
                    (event.state_id,),
                )
            )
        if event.code in _WEAPONS and first_weapon[event.player] is None:
            first_weapon[event.player] = event.round_index
            previous_downgrade = last_downgrade[event.player]
            if previous_downgrade is not None:
                downgrade_to_weapon_delay[event.player] = event.round_index - previous_downgrade
            claims.append(
                StrategicClaim(
                    f"{event.player} 的首次超级武器发生在第 "
                    f"{event.round_index} 轮：{event.action}。",
                    (event.state_id,),
                )
            )
            windows.append(CriticalWindow(event.state_id, event.state_id, "首次超级武器"))
        if event.code in _WEAPONS:
            weapon_targets[event.player].add(event.arguments)
        if event.action == "BUILD_TOWER":
            build_counts[event.player] += 1
        if event.action == "DOWNGRADE_TOWER":
            downgrade_counts[event.player] += 1
            last_downgrade[event.player] = event.round_index
        if event.action == "HOLD":
            before = next(frame for frame in frames if frame.state_id == event.before_frame_id)
            if before.coins[int(event.player[1])] >= 90:
                idle_resource_rounds[event.player].append(event.round_index)
    base_breaches: list[dict[str, object]] = []
    for before, after in zip(frames, frames[1:], strict=False):
        for player_index in (0, 1):
            damage = before.base_hp[player_index] - after.base_hp[player_index]
            if damage <= 0:
                continue
            player = f"P{player_index}"
            base_breaches.append({"round": after.round_index, "player": player, "damage": damage})
            evidence = next(
                (event.state_id for event in timeline if event.round_index == after.round_index),
                None,
            )
            if evidence is not None:
                claims.append(
                    StrategicClaim(
                        f"第 {after.round_index} 轮结算后 {player} 基地损失 "
                        f"{damage:g} HP，剩余 {after.base_hp[player_index]:g} HP。",
                        (evidence,),
                    )
                )
    if frames and frames[-1].winner in {0, 1}:
        winner = f"P{frames[-1].winner}"
        terminal_events = [
            event for event in timeline if event.round_index == frames[-1].round_index
        ]
        if terminal_events:
            evidence = terminal_events[0].state_id
            claims.append(
                StrategicClaim(
                    f"终局胜者为 {winner}，基地HP={frames[-1].base_hp}。",
                    (evidence,),
                )
            )
            windows.append(CriticalWindow(evidence, evidence, "终局基地突破"))
    metrics = {
        "first_action_round": first_action,
        "first_weapon_round": first_weapon,
        "build_count": build_counts,
        "downgrade_count": downgrade_counts,
        "downgrade_to_weapon_delay": downgrade_to_weapon_delay,
        "weapon_target_coverage": {player: len(weapon_targets[player]) for player in ("P0", "P1")},
        "base_breaches": base_breaches,
        "idle_resource_rounds": idle_resource_rounds,
        "build_downgrade_churn": {
            player: min(build_counts[player], downgrade_counts[player]) for player in ("P0", "P1")
        },
    }
    return metrics, tuple(claims), tuple(windows)


def _render_narrative(
    match_id: str,
    winner: str | None,
    timeline: tuple[AtomicEvent, ...],
    claims: tuple[StrategicClaim, ...],
) -> str:
    lines = [
        f"# AntWar2 公开回放：{match_id}",
        "",
        f"有效终局胜者：{winner or '无'}。",
        "",
        "## 原子时间线",
        "",
    ]
    lines.extend(f"- state_id={event.state_id} | {event.text}。" for event in timeline)
    lines.extend(("", "## 有证据的汇总", ""))
    lines.extend(
        f"- state_id={','.join(claim.evidence_state_ids)} | {claim.text}" for claim in claims
    )
    return "\n".join(lines) + "\n"


def decode_replay(
    replay: Sequence[Mapping[str, object]],
    *,
    match_id: str,
) -> ReplayReport:
    if not match_id.strip() or "/" in match_id or ".." in match_id:
        raise ValueError("match_id must be a safe non-empty identifier")
    if not replay:
        raise ValueError("replay cannot be empty")
    frames = _normalize_frames(replay, match_id)
    timeline = _decode_timeline(replay, frames, match_id)
    metrics, claims, windows = _derive(timeline, frames)
    winner = None if frames[-1].winner is None else f"P{frames[-1].winner}"
    frame_by_id: dict[str, CanonicalFrame] = {frame.state_id: frame for frame in frames}
    canonical_frames = {frame.state_id: frame for frame in frames}
    for event in timeline:
        frame_by_id[event.state_id] = canonical_frames[event.before_frame_id]
    narrative = _render_narrative(match_id, winner, timeline, claims)
    return ReplayReport(
        match_id=match_id,
        frames=frames,
        timeline=timeline,
        critical_windows=windows,
        strategic_claims=claims,
        metrics=metrics,
        winner=winner,
        narrative=narrative,
        frame_by_id=frame_by_id,
    )

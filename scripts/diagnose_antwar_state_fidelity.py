"""诊断：antwar 的客户端状态重建从哪一回合、哪个字段开始与官方回放分歧。

为什么需要这支脚本
----------------
`policy_trace_worker` 的保真度自检报出「第 149 回合 coins 回放=[14,104] 重建=[14,55]」，
但金币是**派生量**（收入来自击杀敌蚁），它出现分歧往往意味着更早就有别的东西错了 ——
塔、蚂蚁或信息素。只比 coins/camps 会把"最早的病灶"藏起来，让人对着 149 回合白找。

所以这里对**每一个可比字段**独立记录首次分歧回合，一次性定位病灶。
只读、不写任何文件，只往 stdout 打摘要。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path


def _operation(raw: Mapping[str, object]):
    from antwar.protocol import Operation, OperationType

    opcode = int(raw["type"])  # type: ignore[arg-type]
    operation_type = OperationType(opcode)
    if opcode == 11 or opcode in (21, 22, 23, 24):
        position = raw["pos"]
        assert isinstance(position, Mapping)
        return Operation(operation_type, int(position["x"]), int(position["y"]))
    if opcode == 12:
        return Operation(operation_type, int(raw["id"]), int(raw["args"]))
    if opcode == 13:
        return Operation(operation_type, int(raw["id"]))
    return Operation(operation_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--trace",
        default="",
        help="逐回合追踪区间，形如 143:152（用于定位首次分歧的成因）",
    )
    arguments = parser.parse_args()
    trace_from, trace_to = -1, -1
    if arguments.trace:
        head, _, tail = arguments.trace.partition(":")
        trace_from, trace_to = int(head), int(tail or head)

    import sys

    sys.path.insert(0, str(arguments.candidate.resolve()))
    from antwar.gamestate import GameState

    replay = json.loads(arguments.replay.read_text(encoding="utf-8"))
    seed = int(replay[0]["seed"])
    state = GameState()
    state.init_with_seed(seed)

    first: dict[str, tuple[int, object, object]] = {}
    rejected: list[tuple[int, int, str]] = []

    def note(field: str, index: int, expected: object, actual: object) -> None:
        if field not in first and expected != actual:
            first[field] = (index, expected, actual)

    for index, record in enumerate(replay):
        tracing = trace_from <= index <= trace_to
        operations = {
            player: [_operation(raw) for raw in record.get(f"op{player}", [])]
            for player in (0, 1)
        }
        if tracing:
            print(
                f"r{index}: 入口 重建coin={list(state.coin)} "
                f"op0={[o.dump() for o in operations[0]]} op1={[o.dump() for o in operations[1]]}"
            )
        for player in (0, 1):
            for operation in operations[player]:
                # 官方录像里的操作**理应全部合法**。若我们的重建判它非法，
                # 那正是病灶：说明状态已经偏了（或 Operation 还原方式不对）。
                if not state.apply_operation(player, operation):
                    rejected.append((index, player, operation.dump()))
                    if tracing:
                        print(f"      !! P{player} {operation.dump()} 被判非法")
        state.simulate_next_round()

        expected = record.get("round_state")
        if not isinstance(expected, Mapping):
            continue
        if tracing:
            print(
                f"      出口 重建coin={list(state.coin)} 回放coin={expected['coins'][:2]} "
                f"| 重建hp={list(state.hp)} 回放camps={expected['camps'][:2]}"
            )
        note("coins", index, [int(v) for v in expected["coins"][:2]], [int(v) for v in state.coin])
        note("camps", index, [int(v) for v in expected["camps"][:2]], [int(v) for v in state.hp])
        note(
            "speedLv", index,
            [int(v) for v in expected["speedLv"][:2]], [int(v) for v in state.gen_speed_lv],
        )
        note(
            "anthpLv", index,
            [int(v) for v in expected["anthpLv"][:2]], [int(v) for v in state.ant_maxhp_lv],
        )
        # 蚂蚁：回放里 ants 是全量记录，数存活的（status==0）即可对比。
        alive_expected = sum(
            1 for a in expected.get("ants", []) if isinstance(a, Mapping) and a.get("status") == 0
        )
        note("ants_alive", index, alive_expected, len(state.ants))

    print(f"回合总数 {len(replay)}")
    print(f"被我们判为非法而丢弃的录像操作: {len(rejected)} 条")
    for item in rejected[:10]:
        print(f"  第 {item[0]} 回合 P{item[1]}: {item[2]}")
    print("各字段首次分歧：")
    if not first:
        print("  无 —— 全程逐字段一致 ✅")
    for field, (index, expected_value, actual_value) in sorted(first.items(), key=lambda kv: kv[1][0]):
        print(f"  {field:12s} 第 {index:4d} 回合  回放={expected_value}  重建={actual_value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""探查 antwar 回放里 `round_state.towers` 增量字段的语义。

要回答的问题：``type = -1`` 是什么意思？（推测是"塔被拆除"的哨兵）
验证办法：找出所有 type=-1 的记录，看该 id 在之后的回合里是否还以正常 type 出现；
并统计按"-1 表示移除"累积出的塔数，与该玩家的建塔/拆塔操作数对账。
只读，不写文件。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    replay = json.loads(parser.parse_args().replay.read_text(encoding="utf-8"))

    types: Counter[int] = Counter()
    sentinel_rounds: list[tuple[int, int, int]] = []
    reappeared: list[tuple[int, int]] = []
    alive: dict[int, int] = {}
    builds = Counter()
    downgrades = Counter()

    for index, record in enumerate(replay):
        for player in (0, 1):
            for raw in record.get(f"op{player}", []):
                if raw.get("type") == 11:
                    builds[player] += 1
                elif raw.get("type") == 13:
                    downgrades[player] += 1
        for entry in record.get("round_state", {}).get("towers", []) or []:
            tower_id = int(entry["id"])
            tower_type = int(entry["type"])
            types[tower_type] += 1
            if tower_type == -1:
                sentinel_rounds.append((index, tower_id, int(entry.get("player", -1))))
                alive.pop(tower_id, None)
            else:
                if tower_id in {t for t, _ in sentinel_rounds and []}:
                    pass
                alive[tower_id] = tower_type

    # 哨兵之后又出现的 id（若存在，说明 -1 不是"永久移除"）
    seen_after: dict[int, int] = {}
    sentinel_ids = {tower_id: index for index, tower_id, _ in sentinel_rounds}
    for index, record in enumerate(replay):
        for entry in record.get("round_state", {}).get("towers", []) or []:
            tower_id, tower_type = int(entry["id"]), int(entry["type"])
            if tower_type != -1 and tower_id in sentinel_ids and index > sentinel_ids[tower_id]:
                seen_after.setdefault(tower_id, index)
    reappeared = sorted(seen_after.items())

    print(f"回合数 {len(replay)}")
    print(f"towers 增量记录里的 type 取值分布：{dict(sorted(types.items()))}")
    print(f"type=-1 的记录 {len(sentinel_rounds)} 条，前 8 条 (回合, id, player)：{sentinel_rounds[:8]}")
    print(f"哨兵之后又以正常 type 出现的 id：{reappeared[:8]}（空 = -1 是永久移除）")
    print(f"按 -1=移除 累积出的存活塔 {len(alive)} 座：{dict(sorted(alive.items()))}")
    print(f"录像里的建塔操作数 {dict(builds)}，降级/拆除操作数 {dict(downgrades)}")


if __name__ == "__main__":
    main()

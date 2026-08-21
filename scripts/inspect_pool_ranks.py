"""检查某个游戏的人类池锚点分布，用来给烟测挑一个合适的对手名次。

为什么需要它
------------
烟测的目的是"验证全链路能连续跑完 4 轮"，不是"验证能取胜"。对手挑太强会让
4 轮全 0 胜率，而"全败"和"对局根本没跑起来"在指标上不易区分；挑太弱则
掩盖不了候选包本身的问题。所以要看一眼池子的 Elo 分布，挑一个中游名次。

顺便报告 runnable 数量：烟测的对手必须是能跑起来的选手，否则会全部 incomplete。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(agentbench_root: Path, game: str) -> dict[str, object]:
    players = agentbench_root / "games" / game / "players"
    measured_path = players / "measured_elo.json"
    if not measured_path.is_file():
        return {"game": game, "error": f"missing {measured_path}"}

    document = json.loads(measured_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in document.get("ratings") or []
        if isinstance(row, dict) and row.get("measured_elo") is not None
    ]
    rows.sort(key=lambda row: -float(row["measured_elo"]))

    runnable_count = None
    runnable_path = players / "runnable.json"
    if runnable_path.is_file():
        payload = json.loads(runnable_path.read_text(encoding="utf-8"))
        entries = payload.get("players") or payload.get("results") or []
        runnable_count = sum(
            1
            for item in entries
            if isinstance(item, dict)
            and (item.get("runnable") or item.get("status") == "ok")
        )

    def at(rank: int) -> dict[str, object] | None:
        if rank <= 0 or rank > len(rows):
            return None
        row = rows[rank - 1]
        return {
            "rank": rank,
            "player_id": str(row["player_id"]),
            "elo": round(float(row["measured_elo"]), 1),
            "winrate": round(float(row.get("winrate") or 0.0), 3),
        }

    middle = max(1, len(rows) // 2)
    return {
        "game": game,
        "degree": document.get("degree"),
        "rated_players": len(rows),
        "runnable_players": runnable_count,
        "top": at(1),
        "rank10": at(10),
        "rank30": at(30),
        "middle": at(middle),
        "bottom": at(len(rows)),
        # 烟测推荐名次：取中位。它一定存在，且强度适中。
        "smoke_rank": middle,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agentbench-root", required=True, type=Path)
    parser.add_argument("--games", nargs="+", required=True)
    arguments = parser.parse_args(argv)

    for game in arguments.games:
        report = inspect(arguments.agentbench_root.resolve(), game)
        if report.get("error"):
            print(f"===== {game}: {report['error']}")
            continue
        print(
            f"===== {game}  degree={report['degree']} "
            f"rated={report['rated_players']} runnable={report['runnable_players']}"
        )
        for label in ("top", "rank10", "rank30", "middle", "bottom"):
            row = report.get(label)
            if row:
                print(
                    f"  {label:<8} #{row['rank']:<4} elo={row['elo']:<9} "
                    f"win={row['winrate']:<6} {row['player_id'][:52]}"
                )
        print(f"  → 烟测建议 --opponent-rank {report['smoke_rank']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

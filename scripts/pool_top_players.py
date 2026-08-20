"""人类池榜首实况：两个游戏的实测 Elo 前 N 名。

为什么单独一个脚本
------------------
"我们排第几"这句话只有在**说清楚尺子**之后才有意义。这里报的是
``players/measured_elo.json`` 里的 ``measured_elo``——我们自己在本机后端上
重新打出来的分（``abhl ladder eval``），不是 A 的 manifest 里那套历史分。
两者刻度不同，混用会得出错误结论，所以每次报榜都把口径与样本量一起打出来。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def leaderboard(measured_path: Path, top: int) -> dict[str, object]:
    document = json.loads(measured_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in document.get("ratings") or []
        if isinstance(row, dict) and row.get("measured_elo") is not None
    ]
    rows.sort(key=lambda row: -float(row["measured_elo"]))
    return {
        "game": document.get("game"),
        "scope": document.get("scope"),
        "scope_note": document.get("scope_note"),
        "degree": document.get("degree"),
        "played_matches": document.get("played_matches"),
        "rated_players": len(rows),
        "elo_max": round(float(rows[0]["measured_elo"]), 2) if rows else None,
        "elo_min": round(float(rows[-1]["measured_elo"]), 2) if rows else None,
        "top": [
            {
                "rank": index,
                "player_id": str(row["player_id"]),
                "measured_elo": round(float(row["measured_elo"]), 2),
                "matches": float(row.get("matches") or 0.0),
                "winrate": float(row.get("winrate") or 0.0),
            }
            for index, row in enumerate(rows[:top], start=1)
        ],
    }


def render(report: dict[str, object]) -> str:
    lines = [
        f"===== {report['game']}  "
        f"（口径：{report['scope']} / {report['rated_players']} 人有分 / "
        f"实测 {report['played_matches']} 局 / 每人约 {report['degree']} 个对手）",
        f"      Elo 跨度 {report['elo_min']} ~ {report['elo_max']}",
        f"{'名次':>4} {'measured_elo':>13} {'局数':>6} {'胜率':>7}  player_id",
    ]
    for row in report["top"]:
        lines.append(
            f"{'#' + str(row['rank']):>4} {row['measured_elo']:>13.2f} "
            f"{row['matches']:>6.0f} {row['winrate']:>7.3f}  {row['player_id']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", nargs="+", required=True, type=Path)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)

    reports = [leaderboard(path, arguments.top) for path in arguments.measured]
    for report in reports:
        print(render(report))
        print()
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

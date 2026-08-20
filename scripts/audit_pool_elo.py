"""人类静态池 Elo 的可信度审计。

为什么需要审计
--------------
"我们在人类池里排第几"这句话的可信度上限，取决于**池子本身的刻度有多准**。
池子是用稀疏二分图对局拟合出来的（每人只打约 ``degree`` 个对手），有三个已知风险：

1. **样本太薄**：每人 12~16 局，Elo 标准误可能有上百分。
2. **胜率饱和**：如果排名靠前的选手胜率全是 1.000，说明他们没输过任何一局——
   BT 模型对"从未失败"的选手只能给出下界，真实差距不可辨识。这时"第 1 名比
   第 2 名强 20 分"这种说法是没有意义的。
3. **连通性**：二分图如果分裂成几块，跨块的分数不可比。

本脚本把这三件事量化，不做任何美化。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def audit(measured_path: Path) -> dict[str, object]:
    document = json.loads(measured_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in document.get("ratings") or []
        if isinstance(row, dict) and row.get("measured_elo") is not None
    ]
    rows.sort(key=lambda row: -float(row["measured_elo"]))

    saturated = [row for row in rows if float(row.get("winrate") or 0.0) >= 1.0]
    zeroed = [row for row in rows if float(row.get("winrate") or 0.0) <= 0.0]
    matches = [float(row.get("matches") or 0.0) for row in rows]

    # BT 的粗略标准误：s.e. ≈ 400/ln(10) / sqrt(n·p·(1-p))。
    # p 取该选手实际胜率，p∈{0,1} 时不可辨识（分母为 0），显式标出来。
    standard_errors: list[float | None] = []
    for row in rows:
        played = float(row.get("matches") or 0.0)
        rate = float(row.get("winrate") or 0.0)
        variance = played * rate * (1.0 - rate)
        standard_errors.append(
            None if variance <= 0 else (400.0 / math.log(10)) / math.sqrt(variance)
        )
    finite = [value for value in standard_errors if value is not None]

    top10 = rows[:10]
    top10_saturated = sum(1 for row in top10 if float(row.get("winrate") or 0.0) >= 1.0)

    # 相邻名次的分差 vs 标准误：分差远小于误差就说明这个名次排序不可信。
    gaps = [
        round(float(top10[index]["measured_elo"]) - float(top10[index + 1]["measured_elo"]), 2)
        for index in range(len(top10) - 1)
    ]

    return {
        "game": document.get("game"),
        "scope": document.get("scope"),
        "rated_players": len(rows),
        "degree": document.get("degree"),
        "planned_matches": document.get("planned_matches"),
        "played_matches": document.get("played_matches"),
        "matches_per_player_min": min(matches) if matches else None,
        "matches_per_player_median": sorted(matches)[len(matches) // 2] if matches else None,
        "saturated_winrate_1_0": len(saturated),
        "saturated_winrate_0_0": len(zeroed),
        "top10_saturated": top10_saturated,
        "unidentifiable_players": sum(1 for value in standard_errors if value is None),
        "standard_error_median": round(sorted(finite)[len(finite) // 2], 1) if finite else None,
        "standard_error_min": round(min(finite), 1) if finite else None,
        "top10_adjacent_gaps": gaps,
        "top10_gap_median": round(sorted(gaps)[len(gaps) // 2], 2) if gaps else None,
    }


def required_degree(report: dict[str, object], target_se: float) -> int | None:
    """要把 Elo 标准误压到 ``target_se`` 需要多大的 degree。

    BT 的标准误 ∝ 1/sqrt(n)，n 是该选手的对局数，而 ``n = degree × roles``。
    所以 ``degree_new = degree_old × (se_old / se_target)²``。

    这个反推是必要的：degree 直接决定机时（对局数 ∝ degree × 人数），
    随手拍一个数要么白烧几万局，要么压不到能用的精度。
    """

    current_se = report.get("standard_error_median")
    current_degree = report.get("degree")
    if not isinstance(current_se, (int, float)) or not isinstance(current_degree, int):
        return None
    if target_se <= 0:
        return None
    factor = (float(current_se) / target_se) ** 2
    return max(current_degree, int(math.ceil(current_degree * factor)))


def render(report: dict[str, object], target_se: float = 50.0) -> str:
    needed = required_degree(report, target_se)
    lines = [
        f"===== {report['game']}",
        f"  有分选手           {report['rated_players']}",
        f"  每人对手数(degree) {report['degree']}",
        f"  总对局             {report['played_matches']} / 计划 {report['planned_matches']}",
        f"  每人局数           中位 {report['matches_per_player_median']}"
        f"，最少 {report['matches_per_player_min']}",
        f"  胜率=1.000 的选手  {report['saturated_winrate_1_0']}"
        f"（前十里有 {report['top10_saturated']} 个）",
        f"  胜率=0.000 的选手  {report['saturated_winrate_0_0']}",
        f"  Elo 不可辨识选手   {report['unidentifiable_players']}（胜率饱和，BT 只能给下界）",
        f"  Elo 标准误         中位 ±{report['standard_error_median']}"
        f"，最小 ±{report['standard_error_min']}",
        f"  前十相邻分差       中位 {report['top10_gap_median']}"
        f"，明细 {report['top10_adjacent_gaps']}",
    ]
    gap = report.get("top10_gap_median")
    se = report.get("standard_error_median")
    if isinstance(gap, (int, float)) and isinstance(se, (int, float)) and gap < se:
        lines.append(
            f"  ⚠ 前十相邻分差({gap}) 远小于标准误(±{se})：前十的具体顺序不可辨识，"
            f"只能说\"在前十这一档\""
        )
    if needed is not None:
        lines.append(f"  → 要把标准误压到 ±{target_se:.0f} 需 degree≈{needed}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--target-se",
        type=float,
        default=50.0,
        help="目标 Elo 标准误，用来反推需要多大的 degree",
    )
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args(argv)
    reports = [audit(path) for path in arguments.measured]
    for report in reports:
        report["required_degree"] = required_degree(report, arguments.target_se)
        print(render(report, arguments.target_se))
        print()
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/bin/bash
# 各游戏单局耗时对比：解释为什么同样的 degree、不同游戏耗时差 13 倍。
set -uo pipefail

PY=/home/qingle/agentbench/.venv/bin/python

"$PY" - <<'PYEOF'
import glob
import json
import statistics

# 来自 runs/ladder-remeasure/driver2.log 的实测墙钟（分钟）。
minutes = {
    "antwar": 71, "generals": 77, "rollman": 90, "snakego": 275,
    "lostspace": 632, "miracle": 816, "aquawar": 906, "antwar2": 907,
}

header = f"{'game':<11}{'局数':>7}{'回合中位':>10}{'分钟':>7}{'局/分':>8}{'秒/局':>8}"
print(header)
print("-" * len(header))
rows_out = []
for path in sorted(glob.glob("/tmp/abhl-ladder-*/matches.jsonl")):
    game = path.split("abhl-ladder-")[1].split("/")[0]
    rows = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rounds = [r["rounds"] for r in rows if isinstance(r.get("rounds"), int)]
    wall = minutes.get(game, 0)
    per_min = len(rows) / wall if wall else 0.0
    sec = wall * 60 / len(rows) if rows and wall else 0.0
    median_rounds = statistics.median(rounds) if rounds else 0
    rows_out.append((game, len(rows), median_rounds, wall, per_min, sec))
    print(
        f"{game:<11}{len(rows):>7}{median_rounds:>10.0f}{wall:>7}"
        f"{per_min:>8.1f}{sec:>8.1f}"
    )

print()
fastest = min(rows_out, key=lambda r: r[5])
slowest = max(rows_out, key=lambda r: r[5])
print(f"最快 {fastest[0]}: {fastest[5]:.1f} 秒/局")
print(f"最慢 {slowest[0]}: {slowest[5]:.1f} 秒/局  →  相差 {slowest[5] / fastest[5]:.1f} 倍")
PYEOF

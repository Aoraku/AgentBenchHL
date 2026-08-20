"""后台固定池评测队列的进度速览。

用法::

    python3 scripts/pool_elo_status.py <run_root> [<run_root> ...]

报的是每个版本在**冻结人类池**里的实测成绩：总胜率、Elo、插入名次。
这些数字与 ``IterationMetricsFinalized`` 里的 win_rate / pool_elo 不是同一个量——
后者是"对当轮那一个对手"的 2 局，这里是"对全池 94 人各两个座次"的 188 局。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(run_root: Path) -> list[dict]:
    out: list[dict] = []
    queue = run_root / "pool-elo"
    if not queue.is_dir():
        return out
    for directory in sorted(queue.iterdir()):
        summary = directory / "challenger-elo.json"
        if not summary.is_file():
            continue
        try:
            out.append(json.loads(summary.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for name in argv:
        run_root = Path(name).resolve()
        data = rows(run_root)
        status_path = run_root / "pool-elo" / "worker-status.json"
        status = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {}
        )
        print(f"===== {run_root.name}")
        if status:
            print(
                f"  队列 {status.get('pending')}/{status.get('discovered')} 待测，"
                f"load {status.get('load_average')}/{status.get('cpu_count')}，"
                f"池指纹 {status.get('pool_fingerprint')}"
            )
        header = (
            f"{'iter':>4} {'candidate':<28} {'elo':>8} {'rank':>5} "
            f"{'win_rate':>9} {'W-D-L':>12} {'局':>8}"
        )
        print(header)
        print("-" * len(header))
        data.sort(key=lambda row: (row.get("iteration") is None, row.get("iteration") or 0))
        for row in data:
            played = row.get("complete_matches") or 0
            planned = row.get("planned_matches") or 0
            wins = row.get("wins") or 0
            draws = row.get("draws") or 0
            losses = row.get("losses") or 0
            win_rate = (wins + 0.5 * draws) / played if played else 0.0
            partial = "*" if row.get("partial") else ""
            iteration = row.get("iteration")
            elo = row.get("elo")
            print(
                f"{iteration if iteration is not None else '?':>4} "
                f"{str(row.get('challenger_id'))[:28]:<28} "
                f"{elo if elo is not None else '-':>8} "
                f"{'#' + str(row.get('pool_rank')):>5} "
                f"{win_rate:>9.4f} "
                f"{f'{wins}-{draws}-{losses}':>12} "
                f"{f'{played}/{planned}{partial}':>8}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

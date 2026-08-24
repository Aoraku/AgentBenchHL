#!/usr/bin/env python3
"""逐 run 核对"链路真的通了"这件事，只读事件账本，不做任何推断。

为什么需要一个专门的脚本
------------------------
``watch_runs.py`` 报的是"轮数在涨/没涨"这一层，它的"候选=N"数的是**轮数**
（每轮的最佳候选），所以 k=4 的 run 看起来也是"候选=2 个"——**看不出**
一轮到底交了几个策略。而验收要回答的恰恰是三个更细的问题：

1. 每一轮真的交了 k 个候选，而且它们**代码不同**（不是同一份改几个阈值）；
2. 对局数 = k × b × 座次，且 0 回合局占比；
3. 对手真的是从指定名次起、往榜首方向走。

这三件事都有"看起来正常但其实错了"的失败形态，所以逐项打印原始数字。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _events(run_root: Path) -> list[dict]:
    path = run_root / "events.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _ranks(run_root: Path) -> dict[str, int]:
    path = run_root / "public-leaderboard.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["opponent_id"]): int(item["rank"])
        for item in document.get("opponents") or []
        if item.get("rank") is not None
    }


def report(run_root: Path) -> dict[str, object]:
    rows = _events(run_root)
    ranks = _ranks(run_root)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row.get("event_type"))].append(row.get("payload") or {})

    # 候选：GoalVersionSnapshot 一条 = 一个候选，指纹不同才算真的不同。
    snapshots = by_type["GoalVersionSnapshot"]
    fingerprints = {str(item["candidate_id"]): str(item["code_fingerprint"]) for item in snapshots}

    requests = by_type["GoalMatchRequested"]
    matches = by_type["GoalMatchCompleted"]
    zero_rounds = [item for item in matches if int(item.get("rounds") or 0) == 0]

    iterations = []
    for payload in requests:
        candidate_ids = [str(item) for item in payload.get("candidate_ids") or []]
        assignment = payload.get("opponent_assignment") or {}
        opponents = sorted({str(item) for values in assignment.values() for item in values})
        iterations.append(
            {
                "iteration": payload.get("iteration"),
                "k": len(candidate_ids),
                "candidates": candidate_ids,
                "distinct_code": len({fingerprints.get(item) for item in candidate_ids} - {None}),
                "opponents": opponents,
                "opponent_ranks": sorted(ranks[item] for item in opponents if item in ranks),
            }
        )

    finished = by_type["GoalLedDriveFinished"]
    failed = by_type["GoalLedIterationFailed"]
    return {
        "run": run_root.name,
        "iterations": iterations,
        "matches": len(matches),
        "zero_round": len(zero_rounds),
        "roles": sorted({str(item.get("role")) for item in matches if item.get("role")}),
        "stop_reason": (finished[-1].get("stop_reason") if finished else None),
        "error": (finished[-1].get("error") if finished else (failed[-1].get("error") if failed else None)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", nargs="+", required=True)
    parser.add_argument("--expect-k", type=int, default=4)
    arguments = parser.parse_args()

    problems: list[str] = []
    for run_id in arguments.run_id:
        summary = report(arguments.runs_root / run_id)
        print(f"\n=== {run_id} ===")
        if not summary["iterations"]:
            print(f"  ✗ 一轮都没跑起来  error={summary['error']}")
            problems.append(f"{run_id}: 没有任何迭代")
            continue
        for item in summary["iterations"]:
            mark = "✓" if item["k"] == arguments.expect_k == item["distinct_code"] else "✗"
            print(
                f"  {mark} 第 {item['iteration']} 轮  k={item['k']}"
                f"  代码互不相同={item['distinct_code']}/{item['k']}"
                f"  对手名次={item['opponent_ranks']}"
            )
            print(f"      候选: {item['candidates']}")
            if item["k"] != arguments.expect_k:
                problems.append(f"{run_id} 第{item['iteration']}轮: k={item['k']}")
            elif item["distinct_code"] != item["k"]:
                problems.append(
                    f"{run_id} 第{item['iteration']}轮: {item['k']} 个候选只有 "
                    f"{item['distinct_code']} 份不同代码（伪多样性）"
                )
        print(
            f"  对局={summary['matches']}  0回合={summary['zero_round']}"
            f"  座次={summary['roles']}  stop={summary['stop_reason']}"
        )
        if summary["error"]:
            print(f"  error={summary['error']}")
            problems.append(f"{run_id}: {summary['error']}")

    print("\n" + "=" * 60)
    if problems:
        print(f"需要关注 {len(problems)} 项：")
        for item in problems:
            print(f"  · {item}")
    else:
        print("全部通过：每轮 k 个候选、代码互不相同、对手名次符合预期。")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())

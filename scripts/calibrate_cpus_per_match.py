"""校准 ``cpus_per_match``：找出既不误判超时、又不浪费核的最小值。

为什么需要校准而不是拍一个数
----------------------------
A 的对战器用**墙钟**做每步超时判定。如果多局对局共享核，计算重的选手会因为
邻居抢占而被判"超时"，于是"慢但合法"的策略被系统性误杀——跑分一旦受并行度
影响，整个基准就失去可比性（见 ``cpu_leases.py`` 的实测记录：rank05 vs rank01
串行 150 s 正常完成 308 回合，4 局并行时全部判超时）。

所以 ``cpus_per_match`` 不能随便降。但实测又发现它设得过大：一局其实是
**两个选手进程轮流走子 + 一个判题器**，合计只吃约 1.05 核，而租约占 3 核，
利用率仅 35%。30 核的租约池因此只能跑 10 路并发，白扔 2/3 算力。

本脚本用**同一批对局**在不同 ``cpus_per_match`` 下各跑一遍，比较：

* ``result`` 序列是否逐局一致（游戏在固定 seed 下是确定性的，
  不一致就说明有对局被环境干扰了）；
* ``incomplete`` 比例（超时/崩溃）；
* ``rounds`` 分布（被误判超时的局会提前结束，回合数显著偏低）。

判据：**结果完全一致 + incomplete 率不升**，才允许把该值用于全量评测。
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from agentbench_hl.application.challenger_eval import (  # noqa: E402
    evaluate_many,
    load_frozen_pool,
)


def _run_once(
    *,
    game: str,
    agentbench_root: Path,
    challenger_id: str,
    snapshot: Path,
    work_root: Path,
    cpus_per_match: int,
    parallel: int,
    opponent_limit: int,
    lease_root: Path,
) -> dict[str, object]:
    """跑一遍并返回逐局结果摘要。"""

    import os

    # 用独立的租约目录，避免和正在跑的全量评测互相抢核。
    os.environ["ABHL_CPU_LEASE_ROOT"] = str(lease_root)

    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    pool = load_frozen_pool(agentbench_root, game)
    # 只取前 N 个对手，控制校准成本；对手固定，两次跑的是同一批。
    trimmed = dict(sorted(pool.anchors.items())[:opponent_limit])
    pool.anchors = trimmed
    pool._sorted = sorted(trimmed.values(), reverse=True)  # noqa: SLF001

    started = time.time()
    evaluate_many(
        game,
        agentbench_root,
        [(challenger_id, snapshot)],
        queue_root=work_root,
        pool=pool,
        parallel=parallel,
        cpus_per_match=cpus_per_match,
    )
    elapsed = time.time() - started

    rows = [
        json.loads(line)
        for line in (work_root / challenger_id / "matches.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: (row["opponent"], row["role"], row["seed"]))
    complete = [row for row in rows if row["status"] == "complete"]
    rounds = [row["rounds"] for row in complete if row.get("rounds")]
    return {
        "cpus_per_match": cpus_per_match,
        "parallel": parallel,
        "elapsed_s": round(elapsed, 1),
        "matches": len(rows),
        "complete": len(complete),
        "incomplete": len(rows) - len(complete),
        "incomplete_rate": round(1 - len(complete) / len(rows), 4) if rows else None,
        "rounds_median": statistics.median(rounds) if rounds else None,
        "rounds_mean": round(statistics.mean(rounds), 1) if rounds else None,
        "throughput_per_min": round(len(rows) / (elapsed / 60), 1) if elapsed else None,
        # 逐局结果指纹，用来判断两次跑的结论是否完全一致。
        "signature": [
            f"{row['opponent']}|{row['role']}|{row['seed']}|{row['result']}" for row in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--agentbench-root", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path, help="任意一个候选包目录")
    parser.add_argument(
        "--cpus",
        default="3,2,1",
        help="要比较的 cpus_per_match，逗号分隔。基线放第一个",
    )
    parser.add_argument("--total-cpus", type=int, default=30, help="租约池可用核数")
    parser.add_argument("--opponents", type=int, default=12, help="用多少个对手做校准")
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    arguments = parser.parse_args(argv)

    agentbench_root = arguments.agentbench_root.resolve()
    snapshot = arguments.snapshot.resolve()
    work_root = arguments.work_root.resolve()
    lease_root = work_root / "leases"
    values = [int(item) for item in str(arguments.cpus).split(",") if item.strip()]

    results: list[dict[str, object]] = []
    for cpus in values:
        parallel = max(1, arguments.total_cpus // cpus)
        print(f"[calib] cpus_per_match={cpus} parallel={parallel} 开始", flush=True)
        summary = _run_once(
            game=arguments.game,
            agentbench_root=agentbench_root,
            challenger_id=f"calib-c{cpus}",
            snapshot=snapshot,
            work_root=work_root / f"c{cpus}",
            cpus_per_match=cpus,
            parallel=parallel,
            opponent_limit=arguments.opponents,
            lease_root=lease_root,
        )
        results.append(summary)
        print(
            f"[calib] cpus={cpus} parallel={parallel} "
            f"{summary['matches']} 局 / {summary['elapsed_s']}s "
            f"吞吐 {summary['throughput_per_min']} 局/分 "
            f"incomplete={summary['incomplete']} "
            f"回合中位={summary['rounds_median']}",
            flush=True,
        )

    baseline = results[0]
    print(f"\n[calib] 基线 cpus_per_match={baseline['cpus_per_match']}")
    verdict: list[dict[str, object]] = []
    for summary in results[1:]:
        same = summary["signature"] == baseline["signature"]
        no_worse = float(summary["incomplete_rate"] or 0) <= float(
            baseline["incomplete_rate"] or 0
        )
        speedup = (
            round(float(summary["throughput_per_min"]) / float(baseline["throughput_per_min"]), 2)
            if baseline["throughput_per_min"]
            else None
        )
        ok = same and no_worse
        verdict.append(
            {
                "cpus_per_match": summary["cpus_per_match"],
                "parallel": summary["parallel"],
                "results_identical": same,
                "incomplete_not_worse": no_worse,
                "speedup": speedup,
                "safe": ok,
            }
        )
        print(
            f"  cpus={summary['cpus_per_match']:>2} parallel={summary['parallel']:>2} "
            f"结果一致={'是' if same else '否'} "
            f"incomplete 不升={'是' if no_worse else '否'} "
            f"吞吐 ×{speedup} → {'可用' if ok else '不可用'}"
        )
        if not same:
            differences = [
                (expected, actual)
                for expected, actual in zip(
                    baseline["signature"], summary["signature"], strict=False
                )
                if expected != actual
            ]
            print(f"     不一致 {len(differences)} 局，前 3 例：{differences[:3]}")

    safe = [row for row in verdict if row["safe"]]
    if safe:
        best = min(safe, key=lambda row: int(row["cpus_per_match"]))
        print(
            f"\n[calib] 推荐 cpus_per_match={best['cpus_per_match']}"
            f"（parallel={best['parallel']}，相对基线吞吐 ×{best['speedup']}）"
        )
    else:
        print(f"\n[calib] 没有更小的安全值，保持 cpus_per_match={baseline['cpus_per_match']}")

    if arguments.report is not None:
        arguments.report.write_text(
            json.dumps(
                {"runs": results, "verdict": verdict}, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""把 run 里**每个中间版本**放到固定人类池刻度上算 Elo（离线，不烧机时）。

为什么单独一个脚本
------------------
逐轮指标里的 ``pool_elo`` 是"该 run 迄今全部对局"的累积估计——它回答的是
"这条研究轨迹整体处在池子的什么位置"，把早期弱版本和探索性失败也算进去了，
所以它天然低于当前最佳版本的真实水平。

而"我们的某个版本到底有多强"需要按 **candidate_id 分组** 单独估计：
每个版本只用它自己打过的局，锚点仍是对手在人类池的 measured Elo，
尺子不变，所以版本之间、以及版本与人类池之间都可比。

``fixed_pool_elo``（真慢通道：把版本重新塞进全池打一遍）需要额外机时，
本脚本给的是**基于已有对局**的锚定 MLE，字段里如实标注 method 与样本量。

用法
----
    python scripts/pool_elo_versions.py --run-root runs/sota-antwar
    python scripts/pool_elo_versions.py --run-root runs/sota-antwar --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from agentbench_hl.domain.pool_elo import estimate_pool_elo  # noqa: E402

MATCH_EVENTS = ("GoalMatchCompleted",)
METRICS_EVENT = "IterationMetricsFinalized"


@dataclass(frozen=True)
class Anchor:
    """人类池里一位选手的固定刻度。"""

    player_id: str
    elo: float
    rank: int | None


def _read_events(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "events.jsonl"
    if not path.is_file():
        raise SystemExit(f"没有 events.jsonl：{path}")
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # 崩溃时最后一行可能被截断，跳过而不是让整份分析失败。
            continue
    return events


def _load_anchors(run_root: Path) -> dict[str, Anchor]:
    """优先用 run 自己冻结的 public-leaderboard.json（口径与当时对局一致）。"""

    candidates = [
        run_root / "public-leaderboard.json",
        run_root / "workspace" / "leaderboard.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("opponents") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        anchors: dict[str, Anchor] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            player = row.get("opponent_id") or row.get("player_id")
            score = row.get("score")
            if player is None or score is None:
                continue
            rank = row.get("rank")
            anchors[str(player)] = Anchor(
                player_id=str(player),
                elo=float(score),
                rank=int(rank) if rank is not None else None,
            )
        if anchors:
            return anchors
    raise SystemExit(f"找不到可用的人类池锚点（public-leaderboard.json）：{run_root}")


def _match_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") not in MATCH_EVENTS:
            continue
        payload = event.get("payload")
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _iteration_of_request(events: list[dict[str, Any]]) -> dict[str, int]:
    """request_id -> research_iteration，用来把候选归到它出生的那一轮。"""

    mapping: dict[str, int] = {}
    for event in events:
        if event.get("event_type") != METRICS_EVENT:
            continue
        payload = event.get("payload") or {}
        request_id = payload.get("request_id")
        iteration = payload.get("research_iteration")
        if request_id is not None and iteration is not None:
            mapping[str(request_id)] = int(iteration)
    return mapping


def _best_of_iteration(events: list[dict[str, Any]]) -> dict[str, str]:
    best: dict[str, str] = {}
    for event in events:
        if event.get("event_type") != METRICS_EVENT:
            continue
        payload = event.get("payload") or {}
        request_id = payload.get("request_id")
        chosen = payload.get("best_candidate_id")
        if request_id is not None and chosen is not None:
            best[str(request_id)] = str(chosen)
    return best


def _rank_for_elo(anchors: dict[str, Anchor], elo: float) -> int:
    """这个 Elo 插进人类池后会排第几（1 = 榜首）。"""

    stronger = sum(1 for anchor in anchors.values() if anchor.elo > elo)
    return stronger + 1


def analyse(run_root: Path) -> dict[str, Any]:
    events = _read_events(run_root)
    anchors = _load_anchors(run_root)
    anchor_elo = {key: value.elo for key, value in anchors.items()}
    rows = _match_rows(events)
    iteration_of = _iteration_of_request(events)
    best_of = _best_of_iteration(events)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate = row.get("candidate_id")
        if candidate is None:
            continue
        candidate = str(candidate)
        grouped[candidate].append(row)
        request_id = str(row.get("request_id"))
        entry = meta.setdefault(
            candidate,
            {"request_id": request_id, "iteration": iteration_of.get(request_id)},
        )
        if entry.get("iteration") is None:
            entry["iteration"] = iteration_of.get(request_id)

    versions: list[dict[str, Any]] = []
    for candidate, candidate_rows in grouped.items():
        scored = [row for row in candidate_rows if row.get("status") == "complete"]
        estimate = estimate_pool_elo(scored, anchor_elo)
        request_id = meta[candidate]["request_id"]
        wins = sum(1 for row in scored if row.get("result") == "win")
        draws = sum(1 for row in scored if row.get("result") == "draw")
        losses = sum(1 for row in scored if row.get("result") == "loss")
        opponents = sorted({str(row.get("opponent_id")) for row in scored})
        beaten = sorted(
            {str(row.get("opponent_id")) for row in scored if row.get("result") == "win"}
        )
        record = {
            "candidate_id": candidate,
            "iteration": meta[candidate]["iteration"],
            "request_id": request_id,
            "selected_as_best": best_of.get(request_id) == candidate,
            "matches_total": len(candidate_rows),
            "matches_scored": len(scored),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "opponents": opponents,
            "beaten_opponents": beaten,
            "beaten_best_rank": min(
                (anchors[item].rank for item in beaten if anchors.get(item) and anchors[item].rank),
                default=None,
            ),
            "pool_elo": None if estimate is None else round(estimate.elo, 2),
            "pool_elo_detail": None if estimate is None else estimate.as_dict(),
            "would_rank": None if estimate is None else _rank_for_elo(anchors, estimate.elo),
        }
        versions.append(record)

    versions.sort(key=lambda item: (item["iteration"] is None, item["iteration"], item["candidate_id"]))

    pool_sorted = sorted(anchors.values(), key=lambda item: -item.elo)
    return {
        "run_root": str(run_root),
        "pool_size": len(anchors),
        "pool_top": [
            {"rank": item.rank, "player_id": item.player_id, "elo": round(item.elo, 2)}
            for item in pool_sorted[:5]
        ],
        "pool_elo_range": [round(pool_sorted[-1].elo, 2), round(pool_sorted[0].elo, 2)],
        "versions": versions,
    }


def _render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"run: {report['run_root']}")
    lines.append(
        "固定人类池: {size} 人, Elo 区间 {low} ~ {high}".format(
            size=report["pool_size"],
            low=report["pool_elo_range"][0],
            high=report["pool_elo_range"][1],
        )
    )
    lines.append("池内前 5:")
    for row in report["pool_top"]:
        lines.append(f"  #{row['rank']:<3} {row['elo']:>8.2f}  {row['player_id']}")
    lines.append("")
    header = f"{'iter':>4} {'candidate_id':<34} {'elo':>8} {'插入名次':>8} {'局':>4} {'W-D-L':>9} {'最强战胜':>8} best"
    lines.append(header)
    lines.append("-" * len(header))
    for row in report["versions"]:
        elo = "n/a" if row["pool_elo"] is None else f"{row['pool_elo']:.2f}"
        rank = "n/a" if row["would_rank"] is None else f"#{row['would_rank']}"
        beaten = "-" if row["beaten_best_rank"] is None else f"#{row['beaten_best_rank']}"
        lines.append(
            "{iteration:>4} {candidate:<34} {elo:>8} {rank:>8} {played:>4} "
            "{wdl:>9} {beaten:>8} {best}".format(
                iteration="?" if row["iteration"] is None else row["iteration"],
                candidate=row["candidate_id"][:34],
                elo=elo,
                rank=rank,
                played=row["matches_scored"],
                wdl=f"{row['wins']}-{row['draws']}-{row['losses']}",
                beaten=beaten,
                best="*" if row["selected_as_best"] else "",
            )
        )

    rated = [row for row in report["versions"] if row["pool_elo"] is not None]
    if rated:
        peak = max(rated, key=lambda row: row["pool_elo"])
        lines.append("")
        lines.append(
            "峰值版本: {candidate} (iter {iteration}) elo {elo:.2f} → 插入人类池第 #{rank}".format(
                candidate=peak["candidate_id"],
                iteration=peak["iteration"],
                elo=peak["pool_elo"],
                rank=peak["would_rank"],
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--json", type=Path, default=None, help="同时把结构化结果写到文件")
    parser.add_argument("--min-matches", type=int, default=0, help="少于这么多局的版本不展示")
    args = parser.parse_args(argv)

    report = analyse(args.run_root.resolve())
    if args.min_matches > 0:
        report["versions"] = [
            row for row in report["versions"] if row["matches_scored"] >= args.min_matches
        ]
    print(_render(report))
    if args.json is not None:
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\n结构化结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

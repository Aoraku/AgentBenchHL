#!/usr/bin/env python3
"""诊断：为什么"已经打穿 rank2 了"，报告的 pool_elo 还这么低。

两个叠加的原因
--------------
``pool_elo`` 的定义是：拿该 run **迄今为止全部** complete 官方对局，以每个对手的
人类池 elo 为固定锚点，对候选强度 θ 做一维 BT 极大似然估计。这个定义换来了
"换对手不跳变、跨轮可比"，但代价是两点：

1. **它不会遗忘**。第 1~4 轮对 rank10 的 32 连败永远留在样本里，和现在打穿 rank2
   的胜局一起参与同一个 θ 的估计。所以它衡量的是"这个 run 从头到尾的平均实力"，
   不是"现在这版策略有多强"。
2. **它混合了全部 k 个候选**。每轮 8 局分给 4 个候选，其中包含故意冒险、
   最后被证伪的探索候选。它们的败局同样计入 θ。

所以"当前最佳策略的实力"必须换口径看。这个脚本把几种口径并排算出来，
公式和锚点都复用线上那一份 ``estimate_pool_elo``，不另写一套。

用法::

    python3 scripts/diag_elo_windows.py <run_root> --agentbench-root <A 仓>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from agentbench_hl.domain.pool_elo import estimate_pool_elo

POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def _events(run_root: Path) -> list[dict]:
    out: list[dict] = []
    for line in (run_root / "events.jsonl").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _anchors(agentbench_root: Path, game: str) -> dict[str, float]:
    path = agentbench_root / "games" / game / "players" / "measured_elo.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for item in document.get("ratings") or []:
        value = item.get("measured_elo")
        if value is not None:
            out[str(item.get("player_id"))] = float(value)
    return out


def _matches(run_root: Path) -> tuple[list[dict], dict[int, str], str]:
    """逐局记录（附所属轮次），以及每轮的最佳候选、游戏名。"""

    rows: list[dict] = []
    best: dict[int, str] = {}
    game = "?"
    iteration = 0
    for event in _events(run_root):
        kind = event.get("event_type")
        payload = event.get("payload") or {}
        if kind == "GoalMatchCompleted":
            if payload.get("status") != "complete":
                continue
            result = payload.get("result")
            if result not in POINTS:
                continue
            rows.append(
                {
                    "iteration": iteration + 1,
                    "opponent_id": payload.get("opponent_id"),
                    "candidate_id": payload.get("candidate_id"),
                    "points": POINTS[str(result)],
                }
            )
        elif kind == "IterationMetricsFinalized":
            iteration = int(payload.get("research_iteration") or (iteration + 1))
            game = str(payload.get("game") or game)
            if payload.get("best_candidate_id"):
                best[iteration] = str(payload["best_candidate_id"])
            # 该轮定稿后，后续对局属于下一轮。
            for row in rows:
                if row["iteration"] > iteration:
                    row["iteration"] = iteration
    return rows, best, game


def _estimate(rows: list[dict], anchors: dict[str, float]) -> tuple[float | None, int, float]:
    estimate = estimate_pool_elo(rows, anchors)
    if estimate is None:
        return None, 0, 0.0
    return estimate.elo, estimate.anchored_matches, estimate.score_rate


def main() -> int:
    parser = argparse.ArgumentParser(description="pool_elo 的口径对比")
    parser.add_argument("run_root")
    parser.add_argument(
        "--agentbench-root", default=os.environ.get("AGENTBENCH_ROOT", "../AgentBench")
    )
    parser.add_argument("--windows", default="5,10,20", help="滑动窗口（最近 N 轮）")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    rows, best, game = _matches(run_root)
    anchors = _anchors(Path(args.agentbench_root).expanduser().resolve(), game)
    if not rows or not anchors:
        print("缺少对局或锚点")
        return 1
    last_iteration = max(row["iteration"] for row in rows)

    views: OrderedDict[str, list[dict]] = OrderedDict()
    views["全历史（= 报告的 pool_elo）"] = rows
    for window in [int(item) for item in args.windows.split(",") if item.strip()]:
        cutoff = last_iteration - window
        views[f"最近 {window} 轮"] = [row for row in rows if row["iteration"] > cutoff]
    # 只看每轮的最佳候选：把"故意冒险的探索候选"从样本里剔掉。
    views["仅每轮最佳候选"] = [
        row for row in rows if best.get(row["iteration"]) == row["candidate_id"]
    ]
    views["仅最近 10 轮的最佳候选"] = [
        row
        for row in rows
        if row["iteration"] > last_iteration - 10 and best.get(row["iteration"]) == row["candidate_id"]
    ]

    print(f"run={run_root.name} game={game} 共 {len(rows)} 局 / {last_iteration} 轮\n")
    header = f"{'口径':<28} {'elo':>9} {'样本局':>7} {'得分率':>8}"
    print(header)
    print("-" * len(header))
    for label, subset in views.items():
        elo, matches, rate = _estimate(subset, anchors)
        print(
            f"{label:<28} {(f'{elo:.1f}' if elo is not None else '-'):>9} "
            f"{matches:>7} {f'{rate * 100:.0f}%':>8}"
        )

    strongest = max(
        (anchors[row["opponent_id"]] for row in rows if row["opponent_id"] in anchors),
        default=None,
    )
    if strongest is not None:
        print(f"\n最强交手对手的池内 elo: {strongest:.0f}")
    print(
        "口径差 = 「这个 run 的历史平均实力」与「当前这版策略的实力」之差。"
        "报告 SOTA 用后者，画学习曲线用前者（前者才跨轮可比）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

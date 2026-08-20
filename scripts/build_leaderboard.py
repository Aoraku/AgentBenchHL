#!/usr/bin/env python3
"""实验 1 的主表：把一批 run 聚合成 leaderboard。

回答的问题
----------
"同样跑 N 轮，哪个（模型 × harness）在 8 个游戏上进化得最好？"

所以主表有两层：

1. **单元格**（模型 × harness × 游戏）：多个 seed 的 run 聚合成一行；
2. **总表**（模型 × harness）：跨游戏汇总。

为什么不能只看胜率
------------------
胜率是"相对当轮对手"的。有序课程下越强的候选打的对手越强，胜率反而可能更低——
直接按胜率排名会把强模型排到后面。所以主指标用 ``pool_elo``
（相对固定人类池的累积锚定 Elo，见 ``domain/pool_elo.py``），它跨轮、跨游戏同尺。

跨游戏怎么合并
--------------
各游戏的 Elo 刻度虽然都是 400 分制，但**绝对值不可比**（antwar 的池是 -287~1183，
antwar2 是 599~1982）。所以跨游戏汇总用两个**无量纲**口径：

* ``elo_percentile``：候选 Elo 在该游戏人类池里的百分位（0~1），
  "打进人类前 5%" 这种说法就来自这里；
* ``conquest_cleared``：有序课程下已稳定击败的对手数（本身就是可比的整数）。

绝对 Elo 仍然逐游戏列出，供单游戏对比。

用法::

    python3 scripts/build_leaderboard.py --runs-root runs --out reports/exp1
    python3 scripts/build_leaderboard.py --runs-root runs --grid exp1-main --out reports/exp1
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_curves import read_metric_rows  # noqa: E402 - 复用事件读取，避免两份实现


@dataclass
class RunOutcome:
    """一个 run 的末态（主表的最小单位）。"""

    run_id: str
    game: str
    model: str
    harness: str
    opponent_policy: str
    history_mode: str
    iterations: int
    trajectories_seen: int | None
    pool_elo: float | None
    pool_elo_matches: int | None
    best_pool_elo: float | None
    win_rate: float | None
    outcome_ig: float | None
    tokens: int | None
    wall_time_s: float | None
    infra_errors: int
    conquest_cleared: int | None
    ladder_size: int | None
    elo_percentile: float | None = None


def _last_non_null(rows: Sequence[dict[str, object]], key: str) -> object | None:
    for row in reversed(rows):
        value = row.get(key)
        if value is not None:
            return value
    return None


def _numeric_series(rows: Sequence[dict[str, object]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def summarise_run(run_root: Path) -> RunOutcome | None:
    rows = read_metric_rows(run_root)
    if not rows:
        return None
    last = rows[-1]
    detail = last.get("pool_elo_detail")
    matches = detail.get("anchored_matches") if isinstance(detail, dict) else None
    pool_series = _numeric_series(rows, "pool_elo")
    conquest = _last_non_null(rows, "conquest")
    cleared = conquest.get("cleared") if isinstance(conquest, dict) else None

    def as_float(key: str) -> float | None:
        value = _last_non_null(rows, key)
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def as_int(key: str) -> int | None:
        value = as_float(key)
        return None if value is None else int(value)

    return RunOutcome(
        run_id=run_root.name,
        game=str(last.get("game") or "?"),
        model=str(last.get("model") or "?"),
        harness=str(last.get("harness") or "?"),
        opponent_policy=str(last.get("opponent_policy") or "?"),
        history_mode=str(last.get("history_mode") or "full"),
        iterations=len(rows),
        trajectories_seen=as_int("trajectories_seen"),
        pool_elo=as_float("pool_elo"),
        pool_elo_matches=int(matches) if isinstance(matches, (int, float)) else None,
        best_pool_elo=max(pool_series) if pool_series else None,
        win_rate=as_float("win_rate"),
        outcome_ig=as_float("outcome_ig_nats"),
        tokens=as_int("total_tokens"),
        wall_time_s=as_float("total_wall_time_s"),
        infra_errors=sum(int(row.get("infra_errors") or 0) for row in rows),
        conquest_cleared=int(cleared) if isinstance(cleared, (int, float)) else None,
        ladder_size=as_int("ladder_size"),
    )


def load_pool_elos(agentbench_root: Path, game: str) -> list[float]:
    """该游戏人类池的实测 Elo（升序），用于算百分位。"""

    path = agentbench_root / "games" / game / "players" / "measured_elo.json"
    if not path.is_file():
        return []
    document = json.loads(path.read_text(encoding="utf-8"))
    values = [
        float(row["measured_elo"])
        for row in document.get("ratings") or []
        if row.get("measured_elo") is not None
    ]
    return sorted(values)


def attach_percentiles(outcomes: Sequence[RunOutcome], agentbench_root: Path) -> None:
    """把绝对 Elo 换成"在人类池里的百分位"——这才是能跨游戏加总的量。"""

    cache: dict[str, list[float]] = {}
    for outcome in outcomes:
        if outcome.pool_elo is None:
            continue
        pool = cache.setdefault(outcome.game, load_pool_elos(agentbench_root, outcome.game))
        if not pool:
            continue
        outcome.elo_percentile = bisect.bisect_left(pool, outcome.pool_elo) / len(pool)


@dataclass
class Cell:
    """(模型, harness, 游戏) 单元格：多个 seed 的 run 聚合。"""

    model: str
    harness: str
    game: str
    runs: list[RunOutcome] = field(default_factory=list)

    def _median(self, attribute: str) -> float | None:
        values = [
            getattr(run, attribute) for run in self.runs if getattr(run, attribute) is not None
        ]
        return statistics.median(values) if values else None

    def as_row(self) -> dict[str, object]:
        # 用**中位数**而不是均值：seed 之间偶尔有一个 run 因基建问题几乎没跑动，
        # 均值会被它拽下去，中位数不会。样本量（seeds）同时列出以便判断。
        return {
            "model": self.model,
            "harness": self.harness,
            "game": self.game,
            "seeds": len(self.runs),
            "iterations_median": self._median("iterations"),
            "pool_elo_median": self._median("pool_elo"),
            "pool_elo_best": max(
                (run.best_pool_elo for run in self.runs if run.best_pool_elo is not None),
                default=None,
            ),
            "elo_percentile_median": self._median("elo_percentile"),
            "conquest_cleared_median": self._median("conquest_cleared"),
            "win_rate_median": self._median("win_rate"),
            "outcome_ig_median": self._median("outcome_ig"),
            "tokens_median": self._median("tokens"),
            "infra_errors_total": sum(run.infra_errors for run in self.runs),
            "run_ids": [run.run_id for run in self.runs],
        }


Row = dict[str, object]


def build(outcomes: Sequence[RunOutcome]) -> tuple[list[Row], list[Row]]:
    cells: dict[tuple[str, str, str], Cell] = {}
    for outcome in outcomes:
        key = (outcome.model, outcome.harness, outcome.game)
        cells.setdefault(key, Cell(*key)).runs.append(outcome)
    cell_rows = [cell.as_row() for cell in cells.values()]
    cell_rows.sort(key=lambda row: (row["model"], row["harness"], row["game"]))

    totals: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in cell_rows:
        totals[(str(row["model"]), str(row["harness"]))].append(row)

    total_rows: list[dict[str, object]] = []
    for (model, harness), rows in totals.items():
        percentiles = [
            float(row["elo_percentile_median"])
            for row in rows
            if row["elo_percentile_median"] is not None
        ]
        cleared = [
            float(row["conquest_cleared_median"])
            for row in rows
            if row["conquest_cleared_median"] is not None
        ]
        tokens = [float(row["tokens_median"]) for row in rows if row["tokens_median"] is not None]
        total_rows.append(
            {
                "model": model,
                "harness": harness,
                "games": len(rows),
                # 跨游戏主指标：人类池百分位的均值（无量纲，可加总）。
                "elo_percentile_mean": (
                    sum(percentiles) / len(percentiles) if percentiles else None
                ),
                "conquest_cleared_mean": (sum(cleared) / len(cleared) if cleared else None),
                "tokens_mean": (sum(tokens) / len(tokens) if tokens else None),
                "infra_errors_total": sum(int(row["infra_errors_total"]) for row in rows),
            }
        )
    total_rows.sort(
        key=lambda row: (
            row["elo_percentile_mean"] is None,
            -(row["elo_percentile_mean"] or 0.0),
        )
    )
    return cell_rows, total_rows


def _fmt(value: object, spec: str) -> str:
    """表格里的数字格式化；``None`` 一律显示 ``—``。

    刻意不把 ``None`` 显示成 ``0``：``pool_elo`` 为空意味着"这个 run 还没有任何
    可锚定的完整对局"，跟"实力为 0"是两件完全不同的事。
    """

    if value is None:
        return "—"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def write_outputs(
    cell_rows: Sequence[dict[str, object]],
    total_rows: Sequence[dict[str, object]],
    outcomes: Sequence[RunOutcome],
    out: Path,
) -> None:
    out.mkdir(parents=True, exist_ok=True)

    with (out / "leaderboard_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        if cell_rows:
            writer = csv.DictWriter(handle, fieldnames=list(cell_rows[0].keys()))
            writer.writeheader()
            for row in cell_rows:
                writer.writerow({**row, "run_ids": ";".join(row["run_ids"])})  # type: ignore[arg-type]

    with (out / "leaderboard_totals.csv").open("w", encoding="utf-8", newline="") as handle:
        if total_rows:
            writer = csv.DictWriter(handle, fieldnames=list(total_rows[0].keys()))
            writer.writeheader()
            writer.writerows(total_rows)

    (out / "leaderboard.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "runs": [outcome.__dict__ for outcome in outcomes],
                "cells": cell_rows,
                "totals": total_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# 实验 1 主表",
        "",
        "主指标 = `elo_percentile`（候选 Elo 在该游戏**人类池**里的百分位）。",
        "为什么不用绝对 Elo：各游戏刻度不可比（antwar 池 -287~1183，antwar2 池 599~1982）。",
        "为什么不用胜率：有序课程下越强的候选打越强的对手，胜率反而可能更低。",
        "",
        "## 总表（模型 × harness）",
        "",
        "| 模型 | harness | 游戏数 | 人类池百分位 | 平均征服数 | 平均 token | 基建失败 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in total_rows:
        lines.append(
            f"| {row['model']} | {row['harness']} | {row['games']} | "
            f"{_fmt(row['elo_percentile_mean'], '.1%')} | "
            f"{_fmt(row['conquest_cleared_mean'], '.1f')} | "
            f"{_fmt(row['tokens_mean'], '.0f')} | "
            f"{row['infra_errors_total']} |"
        )
    lines += [
        "",
        "## 单元格（模型 × harness × 游戏）",
        "",
        "| 模型 | harness | 游戏 | seeds | 轮数 | pool_elo 中位/最好 | 百分位 | 征服数 | 胜率 |",
        "|---|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in cell_rows:
        cleared = row["conquest_cleared_median"]
        lines.append(
            f"| {row['model']} | {row['harness']} | {row['game']} | {row['seeds']} | "
            f"{row['iterations_median']} | "
            f"{_fmt(row['pool_elo_median'], '.0f')} / {_fmt(row['pool_elo_best'], '.0f')} | "
            f"{_fmt(row['elo_percentile_median'], '.1%')} | "
            f"{cleared if cleared is not None else '—'} | "
            f"{_fmt(row['win_rate_median'], '.2f')} |"
        )
    (out / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def discover_runs(runs_root: Path, grid: str | None) -> list[Path]:
    roots = [item for item in sorted(runs_root.iterdir()) if (item / "events.jsonl").is_file()]
    if grid:
        roots = [item for item in roots if item.name.startswith(grid)]
    return roots


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-root", type=Path, required=True, help="runs 根目录")
    parser.add_argument("--grid", default=None, help="只聚合 run_id 以此开头的（网格前缀）")
    parser.add_argument(
        "--agentbench-root",
        type=Path,
        default=None,
        help="A 仓路径（算人类池百分位用；默认取 ../AgentBench）",
    )
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)

    agentbench_root = (
        arguments.agentbench_root.resolve()
        if arguments.agentbench_root
        else (REPO_ROOT.parent / "AgentBench")
    )

    roots = discover_runs(arguments.runs_root.resolve(), arguments.grid)
    if not roots:
        raise SystemExit(f"{arguments.runs_root}: 没找到任何 run（需要 events.jsonl）")

    outcomes: list[RunOutcome] = []
    skipped: list[str] = []
    for root in roots:
        outcome = summarise_run(root)
        if outcome is None:
            skipped.append(root.name)
            continue
        outcomes.append(outcome)
    if not outcomes:
        raise SystemExit("所有 run 都还没有 IterationMetricsFinalized 事件，无法聚合")

    attach_percentiles(outcomes, agentbench_root)
    cell_rows, total_rows = build(outcomes)
    write_outputs(cell_rows, total_rows, outcomes, arguments.out.resolve())

    print(f"=== 聚合 {len(outcomes)} 个 run → {arguments.out}")
    if skipped:
        print(f"  跳过（还没有指标事件）：{skipped}")
    missing_elo = [outcome.run_id for outcome in outcomes if outcome.pool_elo is None]
    if missing_elo:
        print(f"  ⚠ 没有 pool_elo 的 run（主指标缺失，别按胜率替代）：{missing_elo}")
    no_pool = sorted({outcome.game for outcome in outcomes if outcome.elo_percentile is None})
    if no_pool:
        print(f"  ⚠ 这些游戏算不出人类池百分位（缺 measured_elo.json）：{no_pool}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

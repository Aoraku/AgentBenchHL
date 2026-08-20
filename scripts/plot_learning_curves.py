"""SOTA run 的学习曲线。

四组纵坐标 × 两种横坐标 = 8 个子图。

纵坐标
------
1. **静态池胜率**：候选版本对冻结人类池的总胜率（(胜+0.5×平)/局数）。
2. **静态池 Elo**：以池选手 measured_elo 为固定锚点的一维 BT-MLE，
   附池内插入名次。
3. **IG**：behavioral information gain（nats），支撑集由状态探针逐点枚举。
4. **token 消耗**：累计曲线 + 逐轮增量柱（两者都保留：累计看总成本，
   增量看哪一轮特别贵）。

横坐标
------
迭代轮数 / 看过的轨迹数。

数据来源
--------
胜率与 Elo 只取全池实测结果（``pool-elo/<candidate>/challenger-elo.json``）；
迭代过程里对单个对手的 ``win_rate`` 与全 run 累计的 ``pool_elo`` 都不画。

**未评测完的版本不画。**数据没到就是没到，宁可图上少几个点，
也不要用半份数据画出一条会被误读的曲线。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

METRICS_EVENT = "IterationMetricsFinalized"

# 四组纵坐标各有固定配色，跨图一致，方便并排比较。
COLORS = {
    "win_rate": "#1f77b4",  # 蓝
    "elo": "#2ca02c",  # 绿
    "ig": "#8c564b",  # 棕
    "tokens": "#e377c2",  # 粉
}
TOKEN_BAR_COLOR = "#c5b0d5"
REFERENCE_COLOR = "#d62728"

plt.rcParams.update(
    {
        "font.sans-serif": [
            "PingFang SC",
            "Hiragino Sans GB",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    }
)


@dataclass
class Point:
    iteration: int
    trajectories_seen: int | None
    candidate_id: str | None
    win_rate: float | None
    elo: float | None
    pool_rank: int | None
    matches: int
    ig: float | None
    tokens: int | None


@dataclass
class Run:
    game: str
    points: list[Point]
    pool_size: int
    pool_top_elo: float | None
    pool_median_elo: float | None
    evaluated: int


def _read_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in (run_dir / "events.jsonl").read_text(
        encoding="utf-8", errors="ignore"
    ).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _pool_meta(run_dir: Path) -> tuple[int, float | None, float | None]:
    path = run_dir / "measured_elo.json"
    if not path.is_file():
        return 0, None, None
    document = json.loads(path.read_text(encoding="utf-8"))
    values = sorted(
        float(row["measured_elo"])
        for row in document.get("ratings") or []
        if isinstance(row, dict) and row.get("measured_elo") is not None
    )
    if not values:
        return 0, None, None
    return len(values), values[-1], values[len(values) // 2]


def _static_results(pool_elo_dir: Path) -> dict[str, dict[str, Any]]:
    """candidate_id → 全池实测结果。只收**完整跑完**的版本。"""

    out: dict[str, dict[str, Any]] = {}
    if not pool_elo_dir.is_dir():
        return out
    for directory in sorted(pool_elo_dir.iterdir()):
        summary = directory / "challenger-elo.json"
        if not summary.is_file():
            continue
        try:
            document = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if document.get("elo") is None or document.get("partial"):
            continue
        played = int(document.get("complete_matches") or 0)
        if played <= 0:
            continue
        wins = int(document.get("wins") or 0)
        draws = int(document.get("draws") or 0)
        document["_win_rate"] = (wins + 0.5 * draws) / played
        out[str(document.get("challenger_id"))] = document
    return out


def _exact_ig(run_dir: Path) -> dict[int, float]:
    path = run_dir / "recomputed-behavioral-ig.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(row["research_iteration"]): float(row["behavioral_ig_exact"])
        for row in document.get("rows") or []
        if row.get("research_iteration") is not None
        and row.get("behavioral_ig_exact") is not None
    }


def _tokens(events: list[dict[str, Any]]) -> dict[str, int]:
    """全 run 累计 token：thread 段内取峰值、跨段求和。

    事件里缓存的 ``total_tokens`` 曾按全 run 取全局 max，而每轮换 thread 后
    计数归零，于是只记住最贵那一段（实测低估 29 倍），所以这里从原始用量事件重算。
    """

    totals: dict[str, int] = {}
    segments: list[int] = []
    current: int | None = None
    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        if event_type in ("GoalLedStarted", "GoalSessionReset", "GoalSessionRotated"):
            if current is not None:
                segments.append(current)
            current = None
        elif event_type == "AgentTokenUsage":
            value = payload.get("total_tokens")
            if isinstance(value, int):
                current = value if current is None else max(current, value)
        elif event_type == METRICS_EVENT:
            request = payload.get("request_id")
            if isinstance(request, str):
                totals[request] = sum(segments) + (current or 0)
    return totals


def load_run(run_dir: Path, pool_elo_dir: Path | None) -> Run:
    events = _read_events(run_dir)
    static = _static_results(pool_elo_dir or (run_dir / "pool-elo"))
    exact = _exact_ig(run_dir)
    tokens = _tokens(events)
    pool_size, pool_top, pool_median = _pool_meta(run_dir)

    game = ""
    points: list[Point] = []
    for event in events:
        if event.get("event_type") != METRICS_EVENT:
            continue
        payload = event.get("payload") or {}
        iteration = payload.get("research_iteration")
        if not isinstance(iteration, int):
            continue
        game = str(payload.get("game") or game)
        champion = payload.get("best_candidate_id")
        result = static.get(str(champion)) if champion else None
        points.append(
            Point(
                iteration=iteration,
                trajectories_seen=payload.get("trajectories_seen"),
                candidate_id=champion if isinstance(champion, str) else None,
                win_rate=None if result is None else float(result["_win_rate"]),
                elo=None if result is None else float(result["elo"]),
                pool_rank=None if result is None else int(result["pool_rank"]),
                matches=0 if result is None else int(result["complete_matches"]),
                ig=exact.get(iteration),
                tokens=tokens.get(str(payload.get("request_id"))),
            )
        )
    points.sort(key=lambda item: item.iteration)
    return Run(game, points, pool_size, pool_top, pool_median, len(static))


X_AXES = (("iteration", "迭代轮数"), ("trajectories_seen", "看过的轨迹数"))


def _xy(
    run: Run, x_key: str, y_key: str
) -> tuple[list[float], list[float], list[Point]]:
    xs: list[float] = []
    ys: list[float] = []
    items: list[Point] = []
    for point in run.points:
        x = point.iteration if x_key == "iteration" else point.trajectories_seen
        y = getattr(point, y_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        items.append(point)
    return xs, ys, items


def _draw_win_rate(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys, _ = _xy(run, x_key, "win_rate")
    axis.plot(
        xs,
        ys,
        marker="o",
        markersize=4,
        linewidth=1.6,
        color=COLORS["win_rate"],
        label="对全池总胜率",
    )
    axis.axhline(0.5, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, label="0.5 基准线")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel(x_label)
    axis.set_ylabel("静态池胜率")
    axis.set_title("静态池胜率")
    axis.legend(loc="lower right")


def _draw_elo(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys, items = _xy(run, x_key, "elo")
    axis.plot(xs, ys, linewidth=1.6, color=COLORS["elo"], zorder=2, label="静态池 Elo")
    if items:
        sizes = [18 + 0.35 * point.matches for point in items]
        axis.scatter(xs, ys, s=sizes, color=COLORS["elo"], zorder=3, edgecolor="white")
        for x, y, point in zip(xs, ys, items, strict=True):
            if point.pool_rank is not None:
                axis.annotate(
                    f"#{point.pool_rank}",
                    xy=(x, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=7.5,
                    color="0.35",
                    ha="center",
                )
    handles, labels = axis.get_legend_handles_labels()
    if run.pool_top_elo is not None:
        axis.axhline(run.pool_top_elo, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0)
        handles.append(
            Line2D([], [], color=REFERENCE_COLOR, linestyle="--", linewidth=1.0)
        )
        labels.append(f"池内榜首 {run.pool_top_elo:.0f}")
    if run.pool_median_elo is not None:
        axis.axhline(run.pool_median_elo, color="0.55", linestyle=":", linewidth=1.0)
        handles.append(Line2D([], [], color="0.55", linestyle=":", linewidth=1.0))
        labels.append(f"池内中位 {run.pool_median_elo:.0f}")
    axis.set_xlabel(x_label)
    axis.set_ylabel("静态池 Elo")
    axis.set_title("静态池 Elo（标注为池内名次，点面积∝对局数）")
    axis.legend(handles, labels, loc="lower right")


def _draw_ig(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys, _ = _xy(run, x_key, "ig")
    axis.plot(
        xs,
        ys,
        marker="o",
        markersize=4,
        linewidth=1.6,
        color=COLORS["ig"],
        label="behavioral IG",
    )
    if ys:
        mean = sum(ys) / len(ys)
        axis.axhline(mean, color="0.55", linestyle=":", linewidth=1.0, label=f"均值 {mean:.3f}")
    axis.set_xlabel(x_label)
    axis.set_ylabel("IG (nats)")
    axis.set_title("行为信息增益")
    axis.legend(loc="upper right")


def _draw_tokens(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys, _ = _xy(run, x_key, "tokens")
    if not xs:
        axis.set_xlabel(x_label)
        axis.set_ylabel("累计 token (百万)")
        axis.set_title("token 消耗")
        return

    deltas = [ys[0]] + [max(0.0, ys[i] - ys[i - 1]) for i in range(1, len(ys))]
    bars = axis.twinx()
    bars.set_zorder(1)
    axis.set_zorder(2)
    axis.patch.set_visible(False)
    width = (max(xs) - min(xs)) / max(len(xs) * 1.5, 1) if len(xs) > 1 else 0.8
    bars.bar(
        xs,
        [value / 1000.0 for value in deltas],
        width=width,
        color=TOKEN_BAR_COLOR,
        alpha=0.75,
        label="逐轮增量",
    )
    bars.set_ylabel("逐轮增量 token (千)")
    bars.grid(False)
    axis.plot(
        xs,
        [value / 1_000_000.0 for value in ys],
        marker="o",
        markersize=4,
        linewidth=1.8,
        color=COLORS["tokens"],
        label="累计",
    )
    axis.set_xlabel(x_label)
    axis.set_ylabel("累计 token (百万)")
    axis.set_title("token 消耗（线=累计，柱=逐轮增量）")
    line_handles, line_labels = axis.get_legend_handles_labels()
    bar_handles, bar_labels = bars.get_legend_handles_labels()
    axis.legend(line_handles + bar_handles, line_labels + bar_labels, loc="upper left")


PANELS = (
    ("win_rate", _draw_win_rate),
    ("elo", _draw_elo),
    ("ig", _draw_ig),
    ("tokens", _draw_tokens),
)


def render(run: Run, output: Path) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(15, 18))
    for row, (_key, draw) in enumerate(PANELS):
        for column, (x_key, x_label) in enumerate(X_AXES):
            draw(axes[row][column], run, x_key, x_label)
    figure.suptitle(
        f"{run.game}　迭代 {len(run.points)} 轮　"
        f"人类池 {run.pool_size} 人（冻结）　已全池评测 {run.evaluated} 版",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.982))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def render_comparison(runs: list[Run], output: Path) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(15, 18))
    styles = [("#1f77b4", "-", "o"), ("#d62728", "--", "s")]
    rows = (
        ("win_rate", "静态池胜率", 1.0),
        ("elo", "静态池 Elo", None),
        ("ig", "IG (nats)", None),
        ("tokens", "累计 token (百万)", None),
    )
    for row_index, (y_key, y_label, _) in enumerate(rows):
        for column, (x_key, x_label) in enumerate(X_AXES):
            axis = axes[row_index][column]
            for index, run in enumerate(runs):
                color, linestyle, marker = styles[index % len(styles)]
                xs, ys, _ = _xy(run, x_key, y_key)
                if y_key == "tokens":
                    ys = [value / 1_000_000.0 for value in ys]
                axis.plot(
                    xs,
                    ys,
                    marker=marker,
                    markersize=3.5,
                    linewidth=1.6,
                    linestyle=linestyle,
                    color=color,
                    label=run.game,
                )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(y_label + ("（两游戏 Elo 刻度不同）" if y_key == "elo" else ""))
            if y_key == "win_rate":
                axis.set_ylim(-0.05, 1.05)
            axis.legend(loc="best")
    figure.suptitle("antwar 与 antwar2 对比", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.982))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def table(run: Run) -> list[dict[str, Any]]:
    return [
        {
            "iteration": point.iteration,
            "trajectories_seen": point.trajectories_seen,
            "candidate_id": point.candidate_id,
            "win_rate": None if point.win_rate is None else round(point.win_rate, 4),
            "elo": None if point.elo is None else round(point.elo, 2),
            "pool_rank": point.pool_rank,
            "matches": point.matches,
            "ig": point.ig,
            "tokens": point.tokens,
        }
        for point in run.points
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", required=True, type=Path)
    parser.add_argument("--pool-elo-dir", nargs="*", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--require-evaluated",
        type=int,
        default=1,
        help="全池评测覆盖的轮数少于这个值就拒绝出图（默认 1，设 0 可强制出图）",
    )
    arguments = parser.parse_args(argv)

    pool_dirs = list(arguments.pool_elo_dir or [])
    runs: list[Run] = []
    for index, run_dir in enumerate(arguments.run_dir):
        pool_dir = pool_dirs[index] if index < len(pool_dirs) else None
        run = load_run(run_dir.resolve(), pool_dir)
        name = run.game or run_dir.name
        rows = table(run)
        rated = [row for row in rows if row["elo"] is not None]
        if len(rated) < arguments.require_evaluated:
            print(
                f"[{name}] 跳过出图：全池评测只覆盖 {len(rated)} 轮，"
                f"低于要求的 {arguments.require_evaluated} 轮。"
                "等后台队列跑完再画，不要用半份数据画会被误读的曲线。"
            )
            continue
        runs.append(run)
        image = render(run, arguments.out_dir / f"curves-{name}.png")
        (arguments.out_dir / f"curves-{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{name}] {len(rows)} 轮，全池评测覆盖 {len(rated)} 轮 -> {image}")
        if rated:
            peak = max(rated, key=lambda row: row["elo"])
            print(
                f"        Elo 峰值 {peak['elo']}（池内 #{peak['pool_rank']}）"
                f" iter {peak['iteration']} {peak['candidate_id']}"
                f" 胜率 {peak['win_rate']} / {peak['matches']} 局"
            )
    if len(runs) > 1:
        print(f"[compare] {render_comparison(runs, arguments.out_dir / 'curves-compare.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

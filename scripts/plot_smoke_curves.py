"""烟测曲线：4 轮迭代能画出什么、不能画出什么。

与 ``plot_learning_curves.py`` 的关系
------------------------------------
那个脚本画的是**主线实验**的学习曲线，胜率与 Elo 都取自"每个版本独立打完
冻结人类池"的实测结果（``pool-elo/index.json``）。那需要每版 188~458 局，
不是 4 轮烟测该做的事。

烟测只有 4 轮 × k 个候选 × 少量对局，能诚实支撑的只有这几条：

* **对当轮对手的胜率**：注意它的口径是"对固定的那一个对手"，
  不是"在人类池里的位置"。烟测固定打同一个名次，所以跨轮可比；
* **分差**：连续量，比胜率灵敏得多——4 轮里胜率可能一直是 0，
  但分差在收窄就说明方向对了；
* **behavioral IG**：精确支撑集下的决策分布 KL；
* **token 消耗**：累计 + 逐轮增量。

**不画静态池 Elo**：4 轮里每个候选只打 4 局，反解出的 Elo 会被正则先验顶到
固定值，没有分辨力（实测 2 局估计与 188 局实测最大差 15 个名次）。
硬画一条只会被误读。

所以这张图的标题里必须写明"烟测口径"，避免和主线曲线混看。
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

METRICS_EVENT = "IterationMetricsFinalized"

COLORS = {
    "win_rate": "#1f77b4",
    "margin": "#2ca02c",
    "ig": "#8c564b",
    "tokens": "#e377c2",
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
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    }
)


@dataclass
class Point:
    iteration: int
    trajectories_seen: int | None
    win_rate: float | None
    margin_mean: float | None
    margin_best: float | None
    ig: float | None
    support_mode: str | None
    tokens: int | None
    opponent_rank: int | None
    candidate_id: str | None


@dataclass
class Run:
    label: str
    game: str
    points: list[Point]
    opponent_rank: int | None
    pool_size: int


def _read_events(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = run_dir / "events.jsonl"
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _tokens(events: list[dict[str, Any]]) -> dict[str, int]:
    """全 run 累计 token：thread 段内取峰值、跨段求和。

    事件里缓存的 ``total_tokens`` 在会话每轮轮转时只记住最贵那一段
    （实测低估 29 倍），所以从原始用量事件重算。
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


def _margins_from_matches(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """从对局事件直接算逐轮分差。

    为什么不直接读指标里的 ``margin_mean``：那个字段是后来才补进指标事件的，
    早先跑的 run 里是 null。而 ``GoalMatchCompleted`` 一直带 ``score_margin``，
    所以从对局重算，让历史 run 也能出图。
    """

    per_request: dict[str, list[float]] = {}
    for event in events:
        if event.get("event_type") != "GoalMatchCompleted":
            continue
        payload = event.get("payload") or {}
        if payload.get("status") != "complete":
            continue
        request = payload.get("request_id")
        margin = payload.get("score_margin")
        if isinstance(request, str) and isinstance(margin, (int, float)):
            per_request.setdefault(request, []).append(float(margin))
    return {
        request: {"mean": sum(values) / len(values), "best": max(values)}
        for request, values in per_request.items()
        if values
    }


def load_run(run_dir: Path, label: str) -> Run:
    events = _read_events(run_dir)
    tokens = _tokens(events)
    margins = _margins_from_matches(events)

    ranks: dict[str, int] = {}
    board = run_dir / "public-leaderboard.json"
    if board.is_file():
        payload = json.loads(board.read_text(encoding="utf-8"))
        rows = payload.get("opponents") if isinstance(payload, dict) else payload
        ranks = {
            str(row["opponent_id"]): int(row["rank"])
            for row in rows or []
            if isinstance(row, dict) and row.get("rank") is not None
        }

    pool_size = 0
    measured = run_dir / "measured_elo.json"
    if measured.is_file():
        document = json.loads(measured.read_text(encoding="utf-8"))
        pool_size = sum(
            1
            for row in document.get("ratings") or []
            if isinstance(row, dict) and row.get("measured_elo") is not None
        )

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
        request = str(payload.get("request_id"))
        opponents = payload.get("opponent_ids") or []
        opponent_rank = min(
            (ranks[str(item)] for item in opponents if str(item) in ranks), default=None
        )
        recomputed = margins.get(request)
        points.append(
            Point(
                iteration=iteration,
                trajectories_seen=payload.get("trajectories_seen"),
                win_rate=payload.get("win_rate"),
                margin_mean=(
                    payload.get("margin_mean")
                    if payload.get("margin_mean") is not None
                    else (recomputed or {}).get("mean")
                ),
                margin_best=(
                    payload.get("margin_best")
                    if payload.get("margin_best") is not None
                    else (recomputed or {}).get("best")
                ),
                ig=payload.get("behavioral_ig"),
                support_mode=payload.get("behavioral_ig_support_mode"),
                tokens=tokens.get(request),
                opponent_rank=opponent_rank,
                candidate_id=payload.get("best_candidate_id"),
            )
        )
    points.sort(key=lambda item: item.iteration)
    fixed_rank = next((p.opponent_rank for p in points if p.opponent_rank), None)
    return Run(label, game or label, points, fixed_rank, pool_size)


X_AXES = (("iteration", "迭代轮数"), ("trajectories_seen", "看过的轨迹数"))


def _xy(run: Run, x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for point in run.points:
        x = point.iteration if x_key == "iteration" else point.trajectories_seen
        y = getattr(point, y_key)
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def _draw_win_rate(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys = _xy(run, x_key, "win_rate")
    axis.plot(xs, ys, marker="o", markersize=5, linewidth=1.8, color=COLORS["win_rate"],
              label="对当轮对手胜率")
    axis.axhline(0.5, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, label="0.5 基准")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel(x_label)
    axis.set_ylabel("胜率")
    suffix = f"（固定打 #{run.opponent_rank}）" if run.opponent_rank else ""
    axis.set_title(f"胜率{suffix}")
    axis.legend(loc="best")


def _draw_margin(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys = _xy(run, x_key, "margin_mean")
    bx, by = _xy(run, x_key, "margin_best")
    if xs:
        axis.plot(xs, ys, marker="o", markersize=5, linewidth=1.8,
                  color=COLORS["margin"], label="平均分差")
    if bx:
        axis.plot(bx, by, marker="^", markersize=4.5, linewidth=1.2, linestyle="--",
                  color="#98df8a", label="最好分差")
    axis.axhline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, label="0（胜负分界）")
    axis.set_xlabel(x_label)
    axis.set_ylabel("终局分差")
    axis.set_title("分差（连续奖励，比胜率灵敏）")
    if xs or bx:
        axis.legend(loc="best")


def _draw_ig(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys = _xy(run, x_key, "ig")
    axis.plot(xs, ys, marker="o", markersize=5, linewidth=1.8, color=COLORS["ig"],
              label="behavioral IG")
    modes = {p.support_mode for p in run.points if p.support_mode}
    axis.set_xlabel(x_label)
    axis.set_ylabel("IG (nats)")
    label = "、".join(sorted(modes)) if modes else "无记录"
    axis.set_title(f"行为信息增益（支撑集：{label}）")
    if xs:
        axis.legend(loc="best")


def _draw_tokens(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys = _xy(run, x_key, "tokens")
    axis.set_xlabel(x_label)
    axis.set_ylabel("累计 token (百万)")
    if not xs:
        axis.set_title("token 消耗")
        return
    deltas = [ys[0]] + [max(0.0, ys[i] - ys[i - 1]) for i in range(1, len(ys))]
    bars = axis.twinx()
    bars.set_zorder(1)
    axis.set_zorder(2)
    axis.patch.set_visible(False)
    width = (max(xs) - min(xs)) / max(len(xs) * 1.5, 1) if len(xs) > 1 else 0.6
    bars.bar(xs, [v / 1000.0 for v in deltas], width=width, color=TOKEN_BAR_COLOR,
             alpha=0.75, label="逐轮增量")
    bars.set_ylabel("逐轮增量 token (千)")
    bars.grid(False)
    axis.plot(xs, [v / 1_000_000.0 for v in ys], marker="o", markersize=5,
              linewidth=1.8, color=COLORS["tokens"], label="累计")
    axis.set_title("token 消耗（线=累计，柱=逐轮增量）")
    lh, ll = axis.get_legend_handles_labels()
    bh, bl = bars.get_legend_handles_labels()
    axis.legend(lh + bh, ll + bl, loc="upper left")


PANELS = (_draw_win_rate, _draw_margin, _draw_ig, _draw_tokens)


def render(run: Run, output: Path) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(14, 17))
    for row, draw in enumerate(PANELS):
        for column, (x_key, x_label) in enumerate(X_AXES):
            draw(axes[row][column], run, x_key, x_label)
    figure.suptitle(
        f"{run.game}　烟测口径：{len(run.points)} 轮 × 固定对手"
        + (f" #{run.opponent_rank}" if run.opponent_rank else "")
        + f"　人类池 {run.pool_size} 人\n"
        "（胜率是对该固定对手的，不是人类池排名；静态池 Elo 需主线全池评测，此处不画）",
        fontsize=12.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def render_comparison(runs: list[Run], output: Path) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(14, 17))
    styles = [("#1f77b4", "-", "o"), ("#d62728", "--", "s"), ("#2ca02c", "-.", "^")]
    rows = (
        ("win_rate", "对当轮对手胜率"),
        ("margin_mean", "平均分差"),
        ("ig", "behavioral IG (nats)"),
        ("tokens", "累计 token (百万)"),
    )
    for row_index, (y_key, y_label) in enumerate(rows):
        for column, (x_key, x_label) in enumerate(X_AXES):
            axis = axes[row_index][column]
            for index, run in enumerate(runs):
                color, linestyle, marker = styles[index % len(styles)]
                xs, ys = _xy(run, x_key, y_key)
                if y_key == "tokens":
                    ys = [v / 1_000_000.0 for v in ys]
                axis.plot(xs, ys, marker=marker, markersize=4.5, linewidth=1.6,
                          linestyle=linestyle, color=color, label=run.game)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(y_label)
            if y_key == "win_rate":
                axis.set_ylim(-0.05, 1.05)
                axis.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)
            if y_key == "margin_mean":
                axis.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0)
            axis.legend(loc="best")
    figure.suptitle(
        "三个新接游戏的烟测对比（各自固定对手，分差/Elo 刻度不同，只可比趋势）",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.978))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def table(run: Run) -> list[dict[str, Any]]:
    return [
        {
            "iteration": p.iteration,
            "trajectories_seen": p.trajectories_seen,
            "win_rate": p.win_rate,
            "margin_mean": None if p.margin_mean is None else round(p.margin_mean, 2),
            "margin_best": None if p.margin_best is None else round(p.margin_best, 2),
            "ig": p.ig,
            "support_mode": p.support_mode,
            "tokens": p.tokens,
            "best_candidate_id": p.candidate_id,
        }
        for p in run.points
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)

    runs: list[Run] = []
    for run_dir in arguments.run_dir:
        run = load_run(run_dir.resolve(), run_dir.name)
        if not run.points:
            print(f"[{run_dir.name}] 没有指标事件，跳过")
            continue
        runs.append(run)
        image = render(run, arguments.out_dir / f"smoke-{run.game}.png")
        rows = table(run)
        (arguments.out_dir / f"smoke-{run.game}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{run.game}] {len(rows)} 轮 -> {image}")
        for row in rows:
            print(
                f"    iter {row['iteration']}: win={row['win_rate']} "
                f"margin={row['margin_mean']} ig={row['ig']} "
                f"mode={row['support_mode']}"
            )
    if len(runs) > 1:
        print(f"[compare] {render_comparison(runs, arguments.out_dir / 'smoke-compare.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""HL run 的学习曲线 —— 三组纵坐标 × 两种横坐标 = 6 个子图。

三组纵坐标（IG 已从主线移除，不再画）
------------------------------------
1. **胜率**，两条线画在同一张图上：
   * 虚线（快通道）：迭代过程里那 ``b`` 局实时对战的胜率。零成本，每轮都有，
     但口径是"对本轮那几个对手"，会随对手变强而下降 —— 它衡量的是**难度**，
     不是绝对强度。
   * 实线（慢通道）：中间版本对**冻结人类池**的总胜率。每 3 轮取一版，另起
     后台进程评测（不影响迭代），一版 188~458 局。这条才是绝对强度。
2. **Elo**，同样两条：
   * 橙色点（零成本反解）：拿"该候选在迭代中已经打过的那 b 局" + 冻结池锚点
     做 logistic 反解。一分钱不多花，但样本只有 b 局，噪声大。
   * 实线（慢通道）：全池实测的一维 BT-MLE，附池内插入名次。
3. **token 消耗**：柱=每一轮的增量，线=累计。

为什么胜率与 Elo 都要"快 + 慢"两条
--------------------------------
只有快通道时，曲线会骗人：``progress`` 课程下 agent 变强了就换更强的对手，
于是实时胜率长期贴在 0.5 附近 —— 看起来毫无进展，实际在稳步上升。
只有慢通道时，每 3 轮才一个点，短 run 根本画不出线。两条一起画，
快通道给密度，慢通道给标尺。

**慢通道未评测完的版本不画。**数据没到就是没到，宁可少几个点，
也不要用半份数据画出一条会被误读的曲线。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_fonts import sans_serif_stack  # noqa: E402

METRICS_EVENT = "IterationMetricsFinalized"

# 三组纵坐标各有固定配色，跨图一致，方便并排比较。
# 快通道统一用同色系的浅色 + 虚线，慢通道用深色实线。
COLORS = {
    "win_rate": "#1f77b4",  # 蓝（慢通道）
    "win_rate_live": "#9ecae1",  # 浅蓝（快通道）
    "elo": "#2ca02c",  # 绿（慢通道）
    "elo_live": "#ff7f0e",  # 橙（零成本反解，沿用历史配色）
    "tokens": "#e377c2",  # 粉
}
TOKEN_BAR_COLOR = "#c5b0d5"
REFERENCE_COLOR = "#d62728"

plt.rcParams.update(
    {
        "font.sans-serif": sans_serif_stack(),
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
    # 慢通道（冻结人类池实测）
    win_rate: float | None
    elo: float | None
    pool_rank: int | None
    matches: int
    # 快通道（迭代过程里的 b 局）
    live_win_rate: float | None
    live_elo: float | None
    live_matches: int
    opponents: int
    tokens: int | None
    tokens_delta: int | None


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


def _token_series(events: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """request_id → (累计 token, 该轮增量)。

    口径：每条 ``AgentTokenUsage`` 是**一次模型请求**的花费
    （codex 的 ``tokenUsage.last``），所以逐次相加。

    历史 bug（这段注释存在的理由）：原实现把它当"会话累计值"，按会话事件切段、
    段内取 max、跨段相加。语义反了，实测两种错法同时出现 ——
    ``sota-antwar`` 连续 10 轮报同一个数 137631 一动不动；
    6 个 4 轮 run 逐轮精确翻倍（snakego4: 112526 → 260286 → 520572 → 1041144）。
    """

    out: dict[str, tuple[int, int]] = {}
    running = 0
    last_checkpoint = 0
    for event in events:
        event_type = event.get("event_type")
        payload = event.get("payload") or {}
        if event_type == "AgentTokenUsage":
            value = payload.get("total_tokens")
            if isinstance(value, int):
                running += value
        elif event_type == METRICS_EVENT:
            request = payload.get("request_id")
            if isinstance(request, str):
                out[request] = (running, running - last_checkpoint)
                last_checkpoint = running
    return out


def load_run(run_dir: Path, pool_elo_dir: Path | None) -> Run:
    events = _read_events(run_dir)
    static = _static_results(pool_elo_dir or (run_dir / "pool-elo"))
    tokens = _token_series(events)
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
        cumulative, delta = tokens.get(str(payload.get("request_id")), (None, None))
        live_rate = payload.get("win_rate")
        points.append(
            Point(
                iteration=iteration,
                trajectories_seen=payload.get("trajectories_seen"),
                candidate_id=champion if isinstance(champion, str) else None,
                win_rate=None if result is None else float(result["_win_rate"]),
                elo=None if result is None else float(result["elo"]),
                pool_rank=None if result is None else int(result["pool_rank"]),
                matches=0 if result is None else int(result["complete_matches"]),
                live_win_rate=(
                    float(live_rate) if isinstance(live_rate, (int, float)) else None
                ),
                live_elo=(
                    float(payload["elo_vs_opponent"])
                    if isinstance(payload.get("elo_vs_opponent"), (int, float))
                    else None
                ),
                live_matches=int(payload.get("matches") or 0),
                opponents=len(payload.get("opponent_ids") or []),
                tokens=cumulative,
                tokens_delta=delta,
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
    live_x, live_y, live_items = _xy(run, x_key, "live_win_rate")
    if live_x:
        batch = max((point.opponents for point in live_items), default=0)
        axis.plot(
            live_x,
            live_y,
            marker="s",
            markersize=3.5,
            linewidth=1.2,
            linestyle="--",
            color=COLORS["win_rate_live"],
            label=f"迭代中实时胜率（每轮 {batch} 个对手，零成本）" if batch else "迭代中实时胜率",
        )
    xs, ys, _ = _xy(run, x_key, "win_rate")
    axis.plot(
        xs,
        ys,
        marker="o",
        markersize=4,
        linewidth=1.8,
        color=COLORS["win_rate"],
        label="慢评测：对冻结人类池总胜率",
    )
    axis.axhline(0.5, color=REFERENCE_COLOR, linestyle=":", linewidth=1.0, label="0.5 基准线")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel(x_label)
    axis.set_ylabel("胜率")
    # 标题必须写明两条线口径不同：实时胜率会随对手变强而下降，
    # 不写清楚的话"上升的静态胜率 + 下降的实时胜率"会被读成数据自相矛盾。
    axis.set_title("胜率（虚线=对本轮对手，实线=对全池；口径不同，别直接比高低）")
    axis.legend(loc="lower right")


def _draw_elo(axis, run: Run, x_key: str, x_label: str) -> None:
    live_x, live_y, live_items = _xy(run, x_key, "live_elo")
    if live_x:
        sizes = [12 + 2.5 * point.live_matches for point in live_items]
        axis.scatter(
            live_x,
            live_y,
            s=sizes,
            color=COLORS["elo_live"],
            alpha=0.75,
            zorder=2,
            edgecolor="white",
            linewidth=0.5,
            label="零成本反解（本轮那几局 + 冻结池锚点）",
        )
    xs, ys, items = _xy(run, x_key, "elo")
    axis.plot(xs, ys, linewidth=1.8, color=COLORS["elo"], zorder=3, label="慢评测：全池实测 Elo")
    if items:
        sizes = [18 + 0.35 * point.matches for point in items]
        axis.scatter(xs, ys, s=sizes, color=COLORS["elo"], zorder=4, edgecolor="white")
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
    axis.set_ylabel("Elo")
    axis.set_title("Elo（橙点=零成本反解，绿线=全池实测；标注为池内名次，点面积∝对局数）")
    axis.legend(handles, labels, loc="lower right")


def _draw_tokens(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys, items = _xy(run, x_key, "tokens")
    if not xs:
        axis.set_xlabel(x_label)
        axis.set_ylabel("累计 token (百万)")
        axis.set_title("token 消耗")
        return

    # 增量直接取记账里的逐轮值，而不是对累计曲线做差。
    # 做差在 x 轴是"看过的轨迹数"时会算错（相邻两点未必是相邻两轮）。
    deltas = [float(point.tokens_delta or 0) for point in items]
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
        label="每一轮的增量",
    )
    bars.set_ylabel("逐轮 token (千)")
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
    axis.set_title("token 消耗（线=累计，柱=每一轮）")
    line_handles, line_labels = axis.get_legend_handles_labels()
    bar_handles, bar_labels = bars.get_legend_handles_labels()
    axis.legend(line_handles + bar_handles, line_labels + bar_labels, loc="upper left")


PANELS = (
    ("win_rate", _draw_win_rate),
    ("elo", _draw_elo),
    ("tokens", _draw_tokens),
)


def render(run: Run, output: Path) -> Path:
    figure, axes = plt.subplots(3, 2, figsize=(15, 14))
    for row, (_key, draw) in enumerate(PANELS):
        for column, (x_key, x_label) in enumerate(X_AXES):
            draw(axes[row][column], run, x_key, x_label)
    figure.suptitle(
        f"{run.game}　迭代 {len(run.points)} 轮　"
        f"人类池 {run.pool_size} 人（冻结）　已全池评测 {run.evaluated} 版",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.978))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def render_comparison(runs: list[Run], output: Path) -> Path:
    figure, axes = plt.subplots(3, 2, figsize=(15, 14))
    # 每个游戏一套独立配色：原先只有两组样式循环复用，5 个 run 会让
    # generals / miracle / rollman 和 antwar 同色同线型，图例根本分不出谁是谁。
    styles = [
        ("#1f77b4", "-", "o"),
        ("#d62728", "--", "s"),
        ("#2ca02c", "-.", "^"),
        ("#9467bd", ":", "D"),
        ("#ff7f0e", "-", "v"),
        ("#17becf", "--", "P"),
        ("#8c564b", "-.", "X"),
        ("#e377c2", ":", "*"),
    ]
    rows = (
        ("win_rate", "静态池胜率"),
        ("elo", "静态池 Elo"),
        ("tokens", "累计 token (百万)"),
    )
    for row_index, (y_key, y_label) in enumerate(rows):
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
            # Elo 必须写明不可比：每个游戏的人类池是各自独立拟合的，
            # 锚点不同、量纲不同，放同一根轴上只能看趋势不能比高低。
            axis.set_title(
                y_label + ("（各游戏池刻度独立，纵向不可比）" if y_key == "elo" else "")
            )
            if y_key == "win_rate":
                axis.set_ylim(-0.05, 1.05)
            axis.legend(loc="best")
    figure.suptitle(
        "各游戏 HL 迭代对比（" + "、".join(run.game for run in runs) + "）",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.978))
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
            "live_win_rate": (
                None if point.live_win_rate is None else round(point.live_win_rate, 4)
            ),
            "live_elo": None if point.live_elo is None else round(point.live_elo, 2),
            "live_matches": point.live_matches,
            "opponents": point.opponents,
            "tokens": point.tokens,
            "tokens_delta": point.tokens_delta,
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

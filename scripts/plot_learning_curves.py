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
    "margin": "#8c564b",  # 棕（平均分差）
    "margin_best": "#c49c94",  # 浅棕（最好的一局）
    "tokens": "#e377c2",  # 粉
}
TOKEN_BAR_COLOR = "#c5b0d5"
REFERENCE_COLOR = "#d62728"

#: 分差要有多少个不同取值才算"有分辨力"。
#:
#: 为什么需要这道闸门：**分差不是所有游戏都有意义的**。实测 8 个游戏的
#: ``score_margin`` 取值个数（同一批对局内）：
#:
#:     antwar     54 种 / 344 局      antwar2   55 种 / 440 局
#:     snakego    13 种 /  16 局      rollman    9 种 /  16 局
#:     miracle     4 种 /  16 局（值域 ±3 万，准连续但样本少）
#:     ---- 以下几乎没有信息 ----
#:     generals    2 种：{-1, +1}   ← 分差**就是**胜负本身
#:     lostspace   2 种：{-3, 0}
#:     aquawar     2 种：{-2, 0}
#:
#: generals 的分差字面上等于胜负（赢 +1 / 输 −1），画它等于把胜率图再画一遍；
#: aquawar / lostspace 只有 2 档，比胜率略好但撑不起一条曲线。
#:
#: 硬画的代价不是"多一张没用的图"，而是**误导**：一条在 {−3, 0} 之间跳的折线
#: 看起来就像"分差没在改善"，而事实是这个游戏没有"分差"这个连续量。
#:
#: 4 是分界点：胜负本身最多 3 档（win / draw / loss），要比它多才算带来新信息。
MARGIN_DISTINCT_MIN = 4

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
    #: 终局分差（平均 / 最好的一局）。
    #:
    #: 为什么必须单独画一栏：胜率与 Elo 对"长期打不赢"的组**完全是盲的**。
    #: 实测 fix 组（固定打榜单前 4 名）14 轮胜率恒 0、反解 Elo 恒 1431.37
    #: 一动不动，看图会得出"这一组没有任何学习"的结论；
    #: 而同期分差从 -36.12 收窄到 -28.12（最好局 -32 → -18，改善 44%）——
    #: agent 确实在稳步变强，只是还没跨过"能赢前 4 名"那道门槛。
    #: 分差是连续量，没有胜率那种 0/1 的阈值效应，所以它才能显示这段进展。
    margin_mean: float | None = None
    margin_best: float | None = None


@dataclass
class Run:
    game: str
    points: list[Point]
    pool_size: int
    pool_top_elo: float | None
    pool_median_elo: float | None
    evaluated: int
    #: run 目录名，用来区分**同一游戏的多个 run**。
    #:
    #: 为什么不能只用 game 当标识：ablation 的四个 run 都是 antwar2，
    #: 只按 game 命名会让四张图写到同一个文件名上互相覆盖（最后只剩一张），
    #: 对比图的图例也会是四个一模一样的 "antwar2"，完全分不出谁是谁。
    run_id: str = ""
    #: 对手选择策略，ablation 的自变量。图例里要显示它。
    policy: str | None = None
    batch: int = 0
    #: 逐局**原始**分差的取值集合（判断分辨力用，不画）。
    #:
    #: 为什么要留原始值而不只看逐轮平均：逐局值在 aquawar 这类游戏里只有
    #: {−2, 0} 两档，但一轮 8 局的**平均**能凑出十几个不同小数，
    #: 只看平均会误判成"有分辨力"，而那只是 2 个离散值的组合噪声。
    raw_margin_values: frozenset[float] = frozenset()

    @property
    def label(self) -> str:
        """图例与文件名用的标识：优先显示自变量（policy），其次 run 目录名。"""

        if self.policy:
            suffix = f"b={self.batch}" if self.batch else ""
            return f"{self.policy}{('/' + suffix) if suffix else ''}"
        return self.run_id or self.game

    @property
    def margin_is_informative(self) -> bool:
        """这个游戏的分差值不值得画。

        判据是**实测取值个数**而不是游戏名白名单：白名单要人去维护，
        接第 9 个游戏时一定忘（然后默默画出一张误导的图）。
        取值个数是数据自己说的，新游戏接进来自动就对。
        """

        return len(self.raw_margin_values) >= MARGIN_DISTINCT_MIN


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

    # 逐局原始分差：用来判断这个游戏的分差**是不是一个连续量**。
    # 见 MARGIN_DISTINCT_MIN 的详注（generals 的分差就是胜负 ±1）。
    raw_margins = {
        float(payload["score_margin"])
        for event in events
        if event.get("event_type") == "GoalMatchCompleted"
        for payload in [event.get("payload") or {}]
        if payload.get("status") == "complete"
        and isinstance(payload.get("score_margin"), (int, float))
    }

    game = ""
    policy: str | None = None
    batch = 0
    points: list[Point] = []
    for event in events:
        if event.get("event_type") != METRICS_EVENT:
            continue
        payload = event.get("payload") or {}
        iteration = payload.get("research_iteration")
        if not isinstance(iteration, int):
            continue
        game = str(payload.get("game") or game)
        policy = payload.get("opponent_policy") or policy
        if isinstance(payload.get("batch"), int):
            batch = payload["batch"]
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
                margin_mean=(
                    float(payload["margin_mean"])
                    if isinstance(payload.get("margin_mean"), (int, float))
                    else None
                ),
                margin_best=(
                    float(payload["margin_best"])
                    if isinstance(payload.get("margin_best"), (int, float))
                    else None
                ),
            )
        )
    points.sort(key=lambda item: item.iteration)
    return Run(
        game,
        points,
        pool_size,
        pool_top,
        pool_median,
        len(static),
        run_id=run_dir.name,
        policy=policy,
        batch=batch,
        raw_margin_values=frozenset(raw_margins),
    )


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
    if xs:
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
    # 慢通道没数据时更要写清楚，否则"只有一条虚线"会被当成图坏了。
    axis.set_title(
        "胜率（虚线=对本轮对手，实线=对全池；口径不同，别直接比高低）"
        if xs
        else "胜率（仅对本轮对手；对手难度逐轮变化，未做全池慢评测）"
    )
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
    if xs:
        axis.plot(
            xs, ys, linewidth=1.8, color=COLORS["elo"], zorder=3, label="慢评测：全池实测 Elo"
        )
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
    # 标题要如实说明这张图里**有什么**：慢通道没跑时只有橙点，
    # 写"绿线=全池实测"会让人以为图缺了东西。
    axis.set_title(
        "Elo（橙点=零成本反解，绿线=全池实测；标注为池内名次，点面积∝对局数）"
        if xs
        else "Elo（仅零成本反解：本轮那几局 + 冻结池锚点；未做全池慢评测）"
    )
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


def _draw_margin(axis, run: Run, x_key: str, x_label: str) -> None:
    """终局分差：胜率与 Elo 都看不见的那段进展。

    实测 fix 组 14 轮胜率恒 0、反解 Elo 恒 1431.37，看那两栏会以为这一组
    完全没在学；而分差同期从 -36.12 收窄到 -28.12（最好局 -32 → -18）。
    胜率有阈值效应（赢不下来就一直是 0），分差是连续量，所以它先动。

    **但分差不是所有游戏都有意义**（见 ``MARGIN_DISTINCT_MIN``）：
    generals 的分差字面上就是胜负 ±1。那种情况下如实说明并留白 ——
    一条在两三个离散值之间跳的折线会被读成"分差没在改善"，
    而事实是这个游戏没有"分差"这个连续量。
    """

    axis.set_xlabel(x_label)
    axis.set_ylabel("终局分差")
    if run.raw_margin_values and not run.margin_is_informative:
        observed = sorted(run.raw_margin_values)
        axis.set_title(f"分差（{run.game or run.run_id} 不适用）")
        axis.text(
            0.5,
            0.5,
            f"这个游戏的分差只有 {len(observed)} 种取值：{observed[:6]}\n"
            "它等价于胜负本身，画成折线会被误读成\n"
            "「分差没在改善」，所以这里留白。\n"
            "该游戏的进展请看胜率与 Elo 两栏。",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
        )
        axis.set_xticks([])
        axis.set_yticks([])
        return

    xs, ys, _ = _xy(run, x_key, "margin_mean")
    bx, by, _ = _xy(run, x_key, "margin_best")
    if xs:
        axis.plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=COLORS["margin"],
            label="平均分差",
        )
    if bx:
        axis.plot(
            bx,
            by,
            marker="^",
            markersize=3.5,
            linewidth=1.2,
            linestyle="--",
            color=COLORS["margin_best"],
            label="最好的一局",
        )
    axis.axhline(0.0, color=REFERENCE_COLOR, linestyle=":", linewidth=1.0, label="0（胜负分界）")
    axis.set_xlabel(x_label)
    axis.set_ylabel("终局分差")
    axis.set_title("分差（连续量；胜率卡在 0/1 时靠它看进展）")
    if xs or bx:
        axis.legend(loc="best")


PANELS = (
    ("win_rate", _draw_win_rate),
    ("elo", _draw_elo),
    ("margin", _draw_margin),
    ("tokens", _draw_tokens),
)


def render(run: Run, output: Path) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(15, 18))
    for row, (_key, draw) in enumerate(PANELS):
        for column, (x_key, x_label) in enumerate(X_AXES):
            draw(axes[row][column], run, x_key, x_label)
    figure.suptitle(
        f"{run.game}"
        + (f"　对手策略 {run.policy}" if run.policy else "")
        + (f"　b={run.batch}" if run.batch else "")
        + f"　迭代 {len(run.points)} 轮　"
        f"人类池 {run.pool_size} 人（冻结）　已全池评测 {run.evaluated} 版",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.978))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def render_comparison(runs: list[Run], output: Path) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(15, 18))
    # 一条曲线一套配色：原先只有两组样式循环复用，5 个 run 会让
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

    # 慢通道（全池实测）有没有数据，决定胜率/Elo 用哪个字段。
    #
    # 为什么必须自动回落：ablation 为了不占满机器会关掉 background_pool，
    # 那时 win_rate / elo 全是 None。硬画慢通道字段的结果是**两张空白图**
    # ——图上什么都没有，却不会报错，看图的人只会以为"实验没产生信号"。
    slow = any(point.elo is not None for run in runs for point in run.points)
    if slow:
        rows = (
            ("win_rate", "对冻结人类池胜率", "（慢通道：全池实测）"),
            ("elo", "全池实测 Elo", "（各游戏池刻度独立，纵向不可比）"),
            ("margin_mean", "平均终局分差", "（连续量；胜率卡在 0/1 时靠它看进展）"),
            ("tokens", "累计 token (百万)", ""),
        )
    else:
        rows = (
            ("live_win_rate", "对当轮对手胜率", "（快通道；对手难度逐轮变化，只看趋势）"),
            ("live_elo", "反解 Elo（全池尺度）", "（用本轮那几局 + 冻结池锚点反解）"),
            ("margin_mean", "平均终局分差", "（连续量；胜率卡在 0/1 时靠它看进展）"),
            ("tokens", "累计 token (百万)", ""),
        )

    for row_index, (y_key, y_label, note) in enumerate(rows):
        for column, (x_key, x_label) in enumerate(X_AXES):
            axis = axes[row_index][column]
            for index, run in enumerate(runs):
                color, linestyle, marker = styles[index % len(styles)]
                # 分差对某些游戏等价于胜负（generals 是 ±1），那种曲线不画。
                # 多游戏对比时逐个判断：跳过没意义的，保留有意义的。
                if y_key == "margin_mean" and not run.margin_is_informative:
                    continue
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
                    # 同游戏多 run（ablation）时 game 全一样，必须用 label
                    # 显示自变量，否则图例是 4 个一模一样的 "antwar2"。
                    label=run.label,
                )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            title = y_label + note
            if y_key == "margin_mean":
                skipped = [
                    run.label for run in runs if not run.margin_is_informative
                ]
                if skipped:
                    # 必须说出谁被跳过，否则图例里少一条线会被当成"数据缺失"。
                    title += f"　※ 分差不适用：{'、'.join(skipped)}"
            axis.set_title(title)
            if y_key in ("win_rate", "live_win_rate"):
                axis.set_ylim(-0.05, 1.05)
                axis.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0)
            if y_key == "margin_mean":
                axis.axhline(0.0, color="0.6", linestyle=":", linewidth=1.0)
            # 一条线都没画时不要调 legend（matplotlib 会打警告，且空图例更费解）。
            if axis.get_legend_handles_labels()[0]:
                axis.legend(loc="best")

    games = sorted({run.game for run in runs})
    if len(games) == 1 and len({run.policy for run in runs}) > 1:
        # ablation 口径：同游戏、同模型，只有对手策略不同。
        heading = (
            f"{games[0]}　对手选择方式 ablation（"
            + "、".join(run.label for run in runs)
            + "）"
        )
    else:
        heading = "各游戏 HL 迭代对比（" + "、".join(run.label for run in runs) + "）"
    figure.suptitle(heading + ("" if slow else "　※ 未做全池慢评测，图为快通道口径"), fontsize=14)
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
            "margin_mean": None if point.margin_mean is None else round(point.margin_mean, 2),
            "margin_best": None if point.margin_best is None else round(point.margin_best, 2),
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
        # 文件名用 **run 目录名**，不能用 game。
        # ablation 的四个 run 都是 antwar2，按 game 命名会让四张图写到
        # 同一个 curves-antwar2.png 上互相覆盖，最后只剩最后跑的那一张
        # ——而命令行输出看起来是"四张都出好了"。
        name = run_dir.name or run.game
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
        live = [row for row in rows if row["live_win_rate"] is not None]
        print(
            f"[{name}] policy={run.policy} b={run.batch} {len(rows)} 轮，"
            f"全池评测覆盖 {len(rated)} 轮 -> {image}"
        )
        if live:
            # 用 live_matches（快通道对局数），不是 matches（慢通道全池局数）。
            # 两个字段名只差一个前缀，写错的表现是"每轮 4 个对手 / 0 局"——
            # 自相矛盾但不会报错。
            print(
                f"        快通道胜率 首轮 {live[0]['live_win_rate']} → "
                f"最近 {live[-1]['live_win_rate']}"
                f"（每轮 {live[-1]['opponents']} 个对手 / {live[-1]['live_matches']} 局）"
            )
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

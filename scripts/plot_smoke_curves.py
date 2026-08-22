"""烟测曲线：短 run 能画出什么、不能画出什么。

与 ``plot_learning_curves.py`` 的关系
------------------------------------
那个脚本画的是**主线实验**的曲线，胜率与 Elo 都以"每个版本独立打完冻结人类池"
的实测结果为标尺（``pool-elo/index.json``）。那需要每版 188~458 局，
不是几轮烟测该做的事。

烟测只有几轮 × b 个对手，能诚实支撑的只有这几条：

* **对当轮对手的胜率**：口径是"对本轮那 b 个对手"。b>1 时它有
  0 / 1/b / … / 1 的分辨率；b=1 时只有 {0, 0.5, 1} 三档，几乎必然画成直线
  （实测 4 个游戏四轮胜率恒 0、rollman 恒 1）。
* **分差**：连续量，比胜率灵敏得多——几轮里胜率可能一直是 0，
  但分差在收窄就说明方向对了。
* **零成本 Elo 反解**：拿本轮那 b 局 + 冻结池锚点反解。b 个不同强度的对手
  一起约束时比单个对手稳，但样本仍然只有 b 局，只看趋势。
* **token 消耗**：每轮增量 + 累计。

**不画全池实测 Elo**：几轮里每个候选只打 b 局，反解出的 Elo 会被正则先验顶到
固定值，没有分辨力（实测 2 局估计与 188 局实测最大差 15 个名次）。
硬画一条只会被误读。

**不画 IG**：信息增益已从主线指标移除。
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

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from plot_fonts import sans_serif_stack  # noqa: E402

METRICS_EVENT = "IterationMetricsFinalized"

COLORS = {
    "win_rate": "#1f77b4",
    "margin": "#2ca02c",
    "elo": "#ff7f0e",
    "tokens": "#e377c2",
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
    elo: float | None
    tokens: int | None
    tokens_delta: int | None
    matches: int
    opponents: int
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


def _tokens(events: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """request_id → (累计 token, 该轮增量)。

    口径：每条 ``AgentTokenUsage`` 是**一次模型请求**的花费
    （codex 的 ``tokenUsage.last``），所以逐次相加。

    历史 bug：原实现把它当"会话累计值"，按会话事件切段、段内取 max、跨段相加。
    语义反了，实测两种错法同时出现 —— ``sota-antwar`` 连续 10 轮报同一个数
    137631 一动不动；6 个 4 轮 run 逐轮精确翻倍
    （snakego4: 112526 → 260286 → 520572 → 1041144）。
    """

    totals: dict[str, tuple[int, int]] = {}
    running = 0
    checkpoint = 0
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
                totals[request] = (running, running - checkpoint)
                checkpoint = running
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
        cumulative, delta = tokens.get(request, (None, None))
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
                elo=payload.get("elo_vs_opponent"),
                tokens=cumulative,
                tokens_delta=delta,
                matches=int(payload.get("matches") or 0),
                opponents=len(opponents),
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
    batch = max((point.opponents for point in run.points), default=0)
    axis.plot(xs, ys, marker="o", markersize=5, linewidth=1.8, color=COLORS["win_rate"],
              label=f"对当轮 {batch} 个对手的胜率" if batch else "对当轮对手胜率")
    axis.axhline(0.5, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, label="0.5 基准")
    axis.set_ylim(-0.05, 1.05)
    axis.set_xlabel(x_label)
    axis.set_ylabel("胜率")
    if run.opponent_rank and batch <= 1:
        # b=1 时胜率只有 {0, 0.5, 1} 三档，几乎必然是直线。标题要写清楚，
        # 否则"四轮恒 0"会被读成"这个模型完全不会玩"，而实际是分辨率不够。
        suffix = f"（固定打 #{run.opponent_rank}，b=1 时只有 3 档取值）"
    elif batch > 1:
        suffix = f"（b={batch}，分辨率 1/{batch}）"
    else:
        suffix = ""
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
    axis.set_title("分差（连续量，比胜率灵敏）")
    if xs or bx:
        axis.legend(loc="best")


def _draw_elo(axis, run: Run, x_key: str, x_label: str) -> None:
    """零成本 Elo 反解：本轮那 b 局 + 冻结池锚点。

    b>1 时是**逐对手反解再平均**，比拿总胜率去配一个"平均锚点"稳得多：
    后者在对手强度差得远时会系统性偏掉（打赢 Elo 500 的、打输 1500 的，
    总胜率 0.5 配均值锚 1000 会报 1000，而真实水平明显更接近 500~700 那段）。
    """

    xs, ys = _xy(run, x_key, "elo")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Elo（反解）")
    if not xs:
        axis.set_title("零成本 Elo 反解（本轮无锚点，未记录）")
        return
    sizes = [
        14 + 3.0 * point.matches
        for point in run.points
        if point.elo is not None
        and (point.iteration if x_key == "iteration" else point.trajectories_seen) is not None
    ]
    axis.plot(xs, ys, linewidth=1.4, color=COLORS["elo"], alpha=0.8)
    axis.scatter(xs, ys, s=sizes, color=COLORS["elo"], edgecolor="white", linewidth=0.5,
                 zorder=3, label="反解 Elo（点面积∝对局数）")
    axis.set_title("零成本 Elo 反解（样本仅本轮 b 局，只看趋势）")
    axis.legend(loc="best")


def _draw_tokens(axis, run: Run, x_key: str, x_label: str) -> None:
    xs, ys = _xy(run, x_key, "tokens")
    axis.set_xlabel(x_label)
    axis.set_ylabel("累计 token (百万)")
    if not xs:
        axis.set_title("token 消耗")
        return
    # 增量取记账里的逐轮值，不对累计曲线做差：x 轴是"看过的轨迹数"时，
    # 相邻两点未必是相邻两轮，做差会算错。
    deltas = [
        float(point.tokens_delta or 0)
        for point in run.points
        if point.tokens is not None
        and (point.iteration if x_key == "iteration" else point.trajectories_seen) is not None
    ]
    bars = axis.twinx()
    bars.set_zorder(1)
    axis.set_zorder(2)
    axis.patch.set_visible(False)
    width = (max(xs) - min(xs)) / max(len(xs) * 1.5, 1) if len(xs) > 1 else 0.6
    bars.bar(xs, [v / 1000.0 for v in deltas], width=width, color=TOKEN_BAR_COLOR,
             alpha=0.75, label="每一轮")
    bars.set_ylabel("逐轮 token (千)")
    bars.grid(False)
    axis.plot(xs, [v / 1_000_000.0 for v in ys], marker="o", markersize=5,
              linewidth=1.8, color=COLORS["tokens"], label="累计")
    axis.set_title("token 消耗（线=累计，柱=每一轮）")
    lh, ll = axis.get_legend_handles_labels()
    bh, bl = bars.get_legend_handles_labels()
    axis.legend(lh + bh, ll + bl, loc="upper left")


PANELS = (_draw_win_rate, _draw_margin, _draw_elo, _draw_tokens)


def render(run: Run, output: Path) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(14, 17))
    for row, draw in enumerate(PANELS):
        for column, (x_key, x_label) in enumerate(X_AXES):
            draw(axes[row][column], run, x_key, x_label)
    batch = max((point.opponents for point in run.points), default=0)
    figure.suptitle(
        f"{run.game}　烟测口径：{len(run.points)} 轮"
        + (f" × 每轮 {batch} 个对手" if batch else "")
        + f"　人类池 {run.pool_size} 人\n"
        "（胜率与 Elo 都是对当轮对手的口径，不是人类池排名；"
        "全池实测需主线慢评测，此处不画）",
        fontsize=12.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def render_comparison(runs: list[Run], output: Path) -> Path:
    figure, axes = plt.subplots(4, 2, figsize=(14, 17))
    # 一游戏一色：原先只有 3 套样式循环，8 个游戏会撞色撞线型，图例分不出谁是谁。
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
        ("win_rate", "对当轮对手胜率"),
        ("margin_mean", "平均分差"),
        ("elo", "零成本反解 Elo"),
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
            # Elo 跨游戏不可比：每个游戏的人类池各自独立拟合，锚点与量纲都不同。
            axis.set_title(
                y_label + ("（各游戏池刻度独立，纵向不可比）" if y_key == "elo" else "")
            )
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
            "elo": None if p.elo is None else round(p.elo, 2),
            "matches": p.matches,
            "opponents": p.opponents,
            "tokens": p.tokens,
            "tokens_delta": p.tokens_delta,
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
                f"({row['matches']} 局 / {row['opponents']} 个对手) "
                f"margin={row['margin_mean']} elo={row['elo']} "
                f"tok+{row['tokens_delta']}"
            )
        # 胜率恒定是短 run 最常见的"看起来坏了其实是分辨率不够"，
        # 直接在命令行点出来，省得看图的人误判。
        rates = {row["win_rate"] for row in rows if row["win_rate"] is not None}
        if len(rates) == 1 and rows:
            only = next(iter(rates))
            print(
                f"    ⚠️ {len(rows)} 轮胜率恒为 {only}："
                f"本轮只打了 {rows[-1]['opponents']} 个对手，分辨率不足。"
                "看分差列是否在改善；要提高分辨率就加大 curriculum.batch。"
            )
    if len(runs) > 1:
        print(f"[compare] {render_comparison(runs, arguments.out_dir / 'smoke-compare.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

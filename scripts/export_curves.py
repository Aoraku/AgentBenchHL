#!/usr/bin/env python3
"""把 run 的逐轮指标导出成曲线数据（4 纵 × 2 横），支持多 run 对比。

为什么不用 ``reporting/curves.py``
---------------------------------
那个模块只接受旧 Plan II 的 ``IterationMetrics`` 对象（要 ``elo_p0``/``elo_p1``/
``champion_id``），与 goal-led 的 ``IterationMetricsFinalized`` 字段不兼容，
横坐标还硬编码成迭代轮数。实验 2/4/5 要的是"同一组纵坐标、两种横坐标、
多条 run 叠在一张图上比"，所以这里直接吃事件账本。

覆盖的实验
----------
* **实验 2**（HL vs 人类）：单 run 的 4×2 曲线；
* **实验 4**（历史消融）：``--label`` 按 ``history_mode`` 分组叠图；
* **实验 5**（对手顺序消融）：按 ``opponent_policy`` 分组叠图，
  另出一条"征服到第几个对手"（``conquest.cleared``）；
* **实验 1**（主表）：见 ``build_leaderboard.py``，它复用本文件的读取函数。

纵坐标（4 条，都来自事件，不做二次加工）
----------------------------------------
=================  =========================================================
键                 含义与注意事项
=================  =========================================================
``pool_elo``       相对固定人类池的 Elo（累积锚定 MLE）。**跨轮/跨游戏可比**，
                   画"刷到 SOTA"用这条。
``win_rate``       当轮胜率（含平局 0.5）。对手会换，所以它不是能力的单调度量。
``outcome_ig``     结果分布信息增益 nats/局（``outcome_ig_nats``）。
``tokens``         累计 token（``total_tokens``）。
=================  =========================================================

另外三条与"行为"相关的纵坐标（口径见 A 仓 ``decision_space.yaml`` 的
``information_gain`` 段，事件里带 ``behavioral_ig_support_mode`` / ``…_cardinality``）：

=========================  =================================================
键                         含义
=========================  =================================================
``behavioral_ig``          决策级行为信息增益 nats/决策（参考占据上的 policy KL）。
``behavioral_disagreement``动作分歧率，无测量假设；KL 是它在声明 |A| 下的重标度。
``occupancy_shift``        状态占据位移 TV。**单独看**，不与 KL 相加。
=========================  =================================================

横坐标（2 条）
--------------
``iteration``（``research_iteration``）与 ``trajectories``（``trajectories_seen``，
= agent 真正读到过的完整回放局数）。后者才是"看了多少经验"的诚实刻度：
一轮里 ``rollout_k × roles × seeds`` 局，不同配置下一轮的信息量差很多。

输出
----
``--out DIR`` 下写：

* ``curves.csv``：长表（run,label,x_key,x,y_key,y），可直接喂给任何画图工具；
* ``series.json``：按 (label, y_key, x_key) 分组的序列；
* ``summary.md``：每个 run 的末轮值与最大值速览；
* ``--plot`` 时若装了 matplotlib，另出 8 张 png（``<y>_vs_<x>.png``）。

用法::

    # 单 run
    python3 scripts/export_curves.py --run-root runs/antwar2-conquest --out reports/exp2

    # 消融对比（实验 4/5）
    python3 scripts/export_curves.py \\
        --run-root runs/abl-fixed_top --run-root runs/abl-ladder_up \\
        --label-by opponent_policy --out reports/exp5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

METRIC_EVENT = "IterationMetricsFinalized"

# 纵坐标：展示名 -> 事件字段名。
Y_KEYS: dict[str, str] = {
    "pool_elo": "pool_elo",
    "win_rate": "win_rate",
    "outcome_ig": "outcome_ig_nats",
    "tokens": "total_tokens",
    # 决策级行为信息增益（nats/决策）。口径由 A 仓 decision_space.yaml 的
    # information_gain.support 声明，事件里同时带 support_mode/|A|；
    # behavioral_disagreement 是它的无假设对照量（KL 只是它的单调重标度）。
    "behavioral_ig": "behavioral_ig",
    "behavioral_disagreement": "behavioral_action_disagreement",
    # 状态占据位移（TV）。与 policy KL **分开**画，永不相加。
    "occupancy_shift": "behavioral_occupancy_shift",
}
# 横坐标：展示名 -> 事件字段名。
X_KEYS: dict[str, str] = {
    "iteration": "research_iteration",
    "trajectories": "trajectories_seen",
}
# 可用来给曲线分组的字段（消融维度）。
LABEL_FIELDS = (
    "opponent_policy",
    "history_mode",
    "model",
    "harness",
    "game",
    "run",
)


@dataclass
class RunSeries:
    run_id: str
    label: str
    rows: list[dict[str, object]] = field(default_factory=list)

    def points(self, y_key: str, x_key: str) -> list[tuple[float, float]]:
        """取 (x, y) 序列，**跳过 y 为 null 的轮次**。

        跳过而不是补 0：``pool_elo`` 在还没有任何 complete 局时是 null，
        ``outcome_ig`` 在没有影子对局时是 null。补 0 会画出一条假的"从零上升"。
        """

        x_field = X_KEYS[x_key]
        y_field = Y_KEYS[y_key]
        series: list[tuple[float, float]] = []
        for row in self.rows:
            x_value = row.get(x_field)
            y_value = row.get(y_field)
            if x_value is None or y_value is None:
                continue
            try:
                series.append((float(x_value), float(y_value)))
            except (TypeError, ValueError):
                continue
        return series


def read_metric_rows(run_root: Path) -> list[dict[str, object]]:
    """从 ``events.jsonl`` 读逐轮指标。

    直接读事件而不是调 ``abhl metrics export``：省一次子进程，且 run 还在跑时
    也能读到已完成的轮次（事件是追加写的）。
    """

    events_path = run_root / "events.jsonl"
    if not events_path.is_file():
        raise SystemExit(f"{run_root}: 没有 events.jsonl，这不是一个 run 目录")
    rows: list[dict[str, object]] = []
    for line in events_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") != METRIC_EVENT:
            continue
        payload = event.get("payload")
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    rows.sort(key=lambda row: row.get("research_iteration") or 0)
    return rows


def load_runs(run_roots: Sequence[Path], label_by: str) -> list[RunSeries]:
    series: list[RunSeries] = []
    for root in run_roots:
        root = root.resolve()
        rows = read_metric_rows(root)
        if label_by == "run":
            label = root.name
        else:
            # 取第一轮的值当标签；同一个 run 内这些字段不会变。
            label = str(rows[0].get(label_by)) if rows else "unknown"
            if label in {"None", "unknown"}:
                label = root.name
        series.append(RunSeries(run_id=root.name, label=label, rows=rows))
    return series


def conquest_points(run: RunSeries, x_key: str) -> list[tuple[float, float]]:
    """"征服到第几个对手"曲线（实验 2/5 的核心图）。

    ``conquest`` 只在有序课程（ladder_up / ladder_down）下非空；
    其它策略返回空列表，调用方会跳过。
    """

    x_field = X_KEYS[x_key]
    points: list[tuple[float, float]] = []
    for row in run.rows:
        state = row.get("conquest")
        x_value = row.get(x_field)
        if not isinstance(state, Mapping) or x_value is None:
            continue
        cleared = state.get("cleared")
        if cleared is None:
            continue
        points.append((float(x_value), float(cleared)))
    return points


def write_csv(runs: Iterable[RunSeries], out: Path) -> Path:
    path = out / "curves.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "label", "x_key", "x", "y_key", "y"])
        for run in runs:
            for y_key in Y_KEYS:
                for x_key in X_KEYS:
                    for x_value, y_value in run.points(y_key, x_key):
                        writer.writerow([run.run_id, run.label, x_key, x_value, y_key, y_value])
            for x_key in X_KEYS:
                for x_value, cleared in conquest_points(run, x_key):
                    writer.writerow(
                        [run.run_id, run.label, x_key, x_value, "conquest_cleared", cleared]
                    )
    return path


def write_series_json(runs: Sequence[RunSeries], out: Path) -> Path:
    document: dict[str, object] = {"schema_version": "1.0", "runs": []}
    for run in runs:
        entry: dict[str, object] = {
            "run_id": run.run_id,
            "label": run.label,
            "iterations": len(run.rows),
            "series": {},
        }
        for y_key in Y_KEYS:
            for x_key in X_KEYS:
                points = run.points(y_key, x_key)
                if points:
                    entry["series"][f"{y_key}_vs_{x_key}"] = points  # type: ignore[index]
        for x_key in X_KEYS:
            points = conquest_points(run, x_key)
            if points:
                entry["series"][f"conquest_cleared_vs_{x_key}"] = points  # type: ignore[index]
        document["runs"].append(entry)  # type: ignore[attr-defined]
    path = out / "series.json"
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _last_and_best(run: RunSeries, y_key: str) -> str:
    """末轮值 / 全程最好值。抽成模块级函数而不是闭包——在循环里定义闭包并捕获
    循环变量是经典陷阱（这里虽然立刻调用没出错，但没有理由留这个坑）。"""

    points = run.points(y_key, "iteration")
    if not points:
        return "—"
    values = [value for _, value in points]
    return f"{values[-1]:.1f} / {max(values):.1f}"


def write_summary(runs: Sequence[RunSeries], out: Path) -> Path:
    lines = [
        "# 曲线速览",
        "",
        "`末轮` = 最后一轮的值；`最好` = 全程最大值（token 例外，它单调增）。",
        "空值表示该量在这个 run 里全程为 null——**别把它当 0 读**。",
        "",
        "| run | 标签 | 轮数 | 轨迹数 | pool_elo 末轮/最好 | 胜率 末轮/最好 "
        "| outcome_ig 末轮 | tokens |",
        "|---|---|---:|---:|---|---|---|---:|",
    ]
    for run in runs:
        last = run.rows[-1] if run.rows else {}
        tokens = last.get("total_tokens")
        trajectories = last.get("trajectories_seen")
        ig_points = run.points("outcome_ig", "iteration")
        lines.append(
            f"| {run.run_id} | {run.label} | {len(run.rows)} | "
            f"{trajectories if trajectories is not None else '—'} | "
            f"{_last_and_best(run, 'pool_elo')} | {_last_and_best(run, 'win_rate')} | "
            f"{f'{ig_points[-1][1]:.4f}' if ig_points else '—'} | "
            f"{tokens if tokens is not None else '—'} |"
        )
    lines.append("")
    # 征服进度单独列：它是实验 2/5 最直观的"打到第几名"。
    pairs = [(run, conquest_points(run, "iteration")) for run in runs]
    conquest_runs = [(run, points) for run, points in pairs if points]
    if conquest_runs:
        lines += [
            "## 征服进度（cleared = 已稳定击败的对手数）",
            "",
            "| run | 标签 | 末轮 cleared | ladder_size |",
            "|---|---|---:|---:|",
        ]
        for run, points in conquest_runs:
            ladder = run.rows[-1].get("ladder_size") if run.rows else None
            lines.append(
                f"| {run.run_id} | {run.label} | {int(points[-1][1])} | {ladder or '—'} |"
            )
        lines.append("")
    path = out / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_plots(runs: Sequence[RunSeries], out: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")  # 服务器无显示器
        import matplotlib.pyplot as plt
    except ImportError:
        print("  （未安装 matplotlib，跳过 png；curves.csv / series.json 已足够画图）")
        return []

    written: list[Path] = []
    panels = [(y, x) for y in Y_KEYS for x in X_KEYS]
    panels += [("conquest_cleared", x) for x in X_KEYS]
    for y_key, x_key in panels:
        figure, axes = plt.subplots(figsize=(7, 4.5))
        drawn = 0
        for run in runs:
            points = (
                conquest_points(run, x_key)
                if y_key == "conquest_cleared"
                else run.points(y_key, x_key)
            )
            if not points:
                continue
            axes.plot(
                [x for x, _ in points], [y for _, y in points], marker="o", label=run.label
            )
            drawn += 1
        if not drawn:
            plt.close(figure)
            continue
        axes.set_xlabel({"iteration": "迭代轮数", "trajectories": "看过的完整轨迹数"}[x_key])
        axes.set_ylabel(y_key)
        axes.set_title(f"{y_key} vs {x_key}")
        axes.grid(alpha=0.3)
        if drawn > 1:
            axes.legend(fontsize=8)
        figure.tight_layout()
        path = out / f"{y_key}_vs_{x_key}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        required=True,
        help="run 目录（可多次给，用于对比）",
    )
    parser.add_argument(
        "--label-by",
        default="run",
        choices=LABEL_FIELDS,
        help="曲线标签取自哪个字段（消融对比时用 opponent_policy / history_mode）",
    )
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    parser.add_argument("--plot", action="store_true", help="额外输出 png（需要 matplotlib）")
    arguments = parser.parse_args(argv)

    runs = load_runs(arguments.run_root, arguments.label_by)
    empty = [run.run_id for run in runs if not run.rows]
    if empty:
        print(f"⚠ 这些 run 还没有任何 {METRIC_EVENT} 事件（可能刚起或第一轮没跑完）：{empty}")

    arguments.out.mkdir(parents=True, exist_ok=True)
    print(f"=== 导出 {len(runs)} 个 run 到 {arguments.out}")
    for path in (
        write_csv(runs, arguments.out),
        write_series_json(runs, arguments.out),
        write_summary(runs, arguments.out),
    ):
        print(f"  → {path.name}")
    if arguments.plot:
        for path in write_plots(runs, arguments.out):
            print(f"  → {path.name}")

    # 把"哪些量全程为 null"直接说出来：这类缺口如果只体现在空白图里，很容易被误读成 0。
    for run in runs:
        missing = [y for y in Y_KEYS if not run.points(y, "iteration")]
        if missing:
            print(f"  ⚠ {run.run_id}: 全程为 null 的纵坐标 → {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

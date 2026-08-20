#!/usr/bin/env python3
"""把一次 goal-led run 的墙钟时间按阶段拆开，回答"到底是谁慢"。

为什么需要它
------------

"迭代一轮很久"有三种完全不同的成因，对策相反：

* **agent 思考慢**（LLM 在写代码）——只能换模型/降 effort，加 CPU 没用；
* **对局慢**（rollout 那 k 局）——可以靠 ``match_parallelism`` 摊平；
* **测量/评测慢**（影子对局、行为 IG 录制+重放、慢通道池评测）——这部分**与 agent 迭代无关**，
  是给曲线用的，可以异步化或抽样降频。

靠感觉猜会把力气用错地方，所以这里直接读事件时间戳算。

用法::

    python scripts/diag_iteration_timing.py <run_root> [...]
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

#: 阶段划分锚点：事件类型 -> 该事件标记"哪个阶段结束"。
THINKING_END = "GoalMatchRequested"
MATCH_EVENT = "GoalMatchCompleted"
MEASURE_EVENTS = ("ShadowMatchCompleted", "InformationGainMeasured")
ITERATION_END = "IterationMetricsFinalized"


def _stamp(value: object) -> float:
    text = str(value or "").replace("Z", "+00:00")
    return datetime.fromisoformat(text).timestamp()


def _rows(run_root: Path) -> list[dict]:
    path = run_root / "events.jsonl"
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except json.JSONDecodeError:
            continue
        record["_t"] = _stamp(record.get("occurred_at"))
        rows.append(record)
    return rows


def report(run_root: Path) -> None:
    rows = _rows(run_root)
    if not rows:
        print(f"== {run_root}: 没有事件")
        return
    start = rows[0]["_t"]
    print(f"\n===== {run_root.name}（{len(rows)} 个事件，共 {rows[-1]['_t'] - start:.0f}s）")

    # 按迭代切段：每个 GoalMatchRequested 开一段，IterationMetricsFinalized 收一段。
    cursor = start
    iteration = 0
    for row in rows:
        kind = str(row.get("event_type"))
        if kind == THINKING_END:
            iteration += 1
            print(f"\n  ── 迭代 {iteration}")
            print(f"     agent 思考（写代码 → 交 action.json）：{row['_t'] - cursor:>7.0f}s")
            cursor = row["_t"]
        elif kind == ITERATION_END:
            print(f"     指标收口：                           {row['_t'] - cursor:>7.0f}s")
            cursor = row["_t"]

    # 分阶段汇总：对局 / 测量各占多少。
    matches = [row for row in rows if str(row.get("event_type")) == MATCH_EVENT]
    measures = [row for row in rows if str(row.get("event_type")) in MEASURE_EVENTS]
    requests = [row for row in rows if str(row.get("event_type")) == THINKING_END]
    finals = [row for row in rows if str(row.get("event_type")) == ITERATION_END]

    print("\n  ── 阶段汇总")
    if requests:
        thinking = sum(
            request["_t"] - (start if index == 0 else finals[index - 1]["_t"])
            for index, request in enumerate(requests)
            if index == 0 or index - 1 < len(finals)
        )
        print(f"     agent 思考合计：      {thinking:>7.0f}s")
    for label, group in (("对局", matches), ("测量（影子/IG）", measures)):
        if not group:
            continue
        span = group[-1]["_t"] - group[0]["_t"]
        print(f"     {label} {len(group):>3} 次，首尾跨度 {span:>7.0f}s")
    if finals:
        last = finals[-1]["_t"]
        tail = rows[-1]["_t"] - last
        print(f"     末轮指标之后还在跑：  {tail:>7.0f}s（慢通道池评测等）")

    # 每局对局自身的耗时（payload 里若有 wall time 就用它）。
    durations = [
        float(row["payload"]["wall_time_s"])
        for row in matches
        if isinstance(row.get("payload"), dict)
        and isinstance(row["payload"].get("wall_time_s"), (int, float))
    ]
    if durations:
        durations.sort()
        total = sum(durations)
        print(
            f"     单局墙钟：中位 {durations[len(durations) // 2]:.1f}s，"
            f"最长 {durations[-1]:.1f}s，串行总和 {total:.0f}s"
        )


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for item in argv:
        report(Path(item).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

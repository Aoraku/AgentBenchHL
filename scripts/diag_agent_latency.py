#!/usr/bin/env python3
"""诊断：agent 那几百秒到底花在"等模型"还是"跑命令"上。

为什么需要它
------------

``events.jsonl`` 里的 ``AgentTokenUsage`` **不是**模型调用次数（它是 SSE 流里的 usage
更新），而且 ``occurred_at`` 是批量写入的（间隔中位数 0.0s），拿它算延迟会得出完全错误的
结论。真实时间戳只在 codex 自己的 ``logs_*.sqlite`` 里（``ts_nanos``）。

本脚本按 codex 的日志把 agent 阶段拆成：

* **等模型**：一次 turn 内 HTTP 请求发出 → SSE 流结束；
* **跑命令**：两次模型往返之间的空档（agent 在跑本地对局、读文件、grep）。

这两者的对策完全相反：等模型久 ⇒ 降 reasoning effort / 换模型 / 开 prompt caching；
跑命令久 ⇒ 收敛 agent 的探索行为（少读大文件、少跑重复对局）。

用法::

    python scripts/diag_agent_latency.py <run_root>
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

#: 一次模型往返的锚点 target。
HTTP_TARGET = "codex_http_client::client"
SSE_TARGET = "codex_api::sse::responses"
TURN_TARGET = "codex_core::session::turn"
TOOL_TARGET = "codex_core::tools::parallel"
#: 相邻 SSE 事件间隔超过这个秒数，就认为模型这一轮已经吐完、agent 转去执行工具。
IDLE_GAP_S = 2.0


def _rows(path: Path) -> list[tuple[float, str, str]]:
    """读日志行。

    ``ts`` 是 unix 秒、``ts_nanos`` 是**纳秒余数**（不是完整时间戳）——只取后者会得出
    "1130 条日志跨度 1 秒"这种荒谬结论。
    """

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cursor = connection.execute(
            "SELECT ts, ts_nanos, target, substr(feedback_log_body, 1, 160) FROM logs ORDER BY id"
        )
        return [
            (int(second) + int(nanos) / 1e9, str(target), str(body or ""))
            for second, nanos, target, body in cursor.fetchall()
        ]
    finally:
        connection.close()


def report(run_root: Path) -> None:
    logs = sorted(run_root.glob("codex-home/logs_*.sqlite"))
    if not logs:
        print(f"== {run_root}: 没有 codex 日志")
        return
    rows: list[tuple[float, str, str]] = []
    for path in logs:
        rows.extend(_rows(path))
    rows.sort(key=lambda item: item[0])
    if not rows:
        print(f"== {run_root}: 日志为空")
        return

    base = rows[0][0]
    span = rows[-1][0] - base
    counts: dict[str, int] = {}
    for _, target, _ in rows:
        counts[target] = counts.get(target, 0) + 1
    print(f"\n===== {run_root.name}（日志跨度 {span:.0f}s）")
    print(
        "  ⚠️ codex 会滚动清理日志，这里通常只覆盖 run 的**最后一段**；"
        "占比只对这段窗口成立。"
    )
    print(f"  真实模型往返（{HTTP_TARGET}）：{counts.get(HTTP_TARGET, 0)} 次")
    print(f"  turn（{TURN_TARGET}）：{counts.get(TURN_TARGET, 0)} 次")
    print(f"  SSE 事件（{SSE_TARGET}）：{counts.get(SSE_TARGET, 0)} 条")
    print(f"  并行工具批（{TOOL_TARGET}）：{counts.get(TOOL_TARGET, 0)} 次")

    # 用 SSE 事件聚类来分离"模型在输出"与"agent 在跑命令"。
    #
    # 不能拿 HTTP_TARGET 那条日志当"请求发出时刻"——实测它是在**响应完成时**才写的
    # （按它算会得出"等模型 0.0s"这种荒谬结论）。SSE 事件才是模型真正在吐字的证据：
    # 相邻 SSE 间隔小 ⇒ 同一次流式输出；出现大空档 ⇒ 模型已停、agent 在执行工具。
    sse = [stamp for stamp, target, _ in rows if target == SSE_TARGET]
    if len(sse) < 2:
        print("\n  SSE 事件太少，无法拆分")
        return
    bursts: list[tuple[float, float, int]] = []
    start = sse[0]
    previous = sse[0]
    count = 1
    for stamp in sse[1:]:
        if stamp - previous > IDLE_GAP_S:
            bursts.append((start, previous, count))
            start = stamp
            count = 1
        else:
            count += 1
        previous = stamp
    bursts.append((start, previous, count))

    print(f"\n  ── 模型输出段（相邻 SSE 间隔 > {IDLE_GAP_S}s 视为断开）")
    model_time = 0.0
    cursor: float | None = None
    tool_time = 0.0
    for index, (begin, end, events) in enumerate(bursts):
        duration = end - begin
        model_time += duration
        gap = 0.0 if cursor is None else begin - cursor
        tool_time += gap
        cursor = end
        print(
            f"     #{index + 1:2}  +{begin - base:7.1f}s  模型输出 {duration:6.1f}s"
            f"（{events:3} 个 SSE）   之前空档 {gap:6.1f}s"
        )

    print("\n  ── 汇总")
    print(f"     模型输出合计 {model_time:.0f}s（{model_time / span * 100:.0f}%）")
    print(f"     空档合计     {tool_time:.0f}s（{tool_time / span * 100:.0f}%）= agent 在跑命令")
    print(f"     模型往返 {counts.get(HTTP_TARGET, 0)} 次 ⇒ 平均 "
          f"{model_time / max(counts.get(HTTP_TARGET, 1), 1):.1f}s/次输出")


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for item in argv:
        report(Path(item).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

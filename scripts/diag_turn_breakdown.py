#!/usr/bin/env python3
"""诊断：一轮迭代的墙钟时间到底花在哪。

回答的问题
----------
"写一轮为什么要十分钟？一次 60k token 的请求哪有这么慢？"

前半句的前提是错的：一轮**不是**一次请求。agent 在一个 turn 里要读材料、读 8 份
replay.md、写 k 个候选、跑自检，每一次工具调用都是一次完整的模型往返，而且
每次都要把**当前全部上下文**重新发一遍。所以真实成本是

    Σ (第 i 次请求的上下文 → 模型思考 + 输出)  +  Σ 工具执行

而不是单次请求的延迟。这个脚本把两项分开量出来。

口径
----
读 codex 的 session rollout（``codex-home/sessions/**/rollout-*.jsonl``）：

* ``task_started`` / ``task_complete``  : 一个 turn 的边界
* ``token_count``                      : 每次模型请求一条 —— 请求次数就是它的条数
* ``function_call`` → ``function_call_output`` : 工具执行区间（这段时间模型在等）
* 其余时间归给模型（思考 + 输出 + 网络）

用法::

    python3 scripts/diag_turn_breakdown.py <run_root>            # 汇总该 run 全部 turn
    python3 scripts/diag_turn_breakdown.py <run_root> --detail   # 逐 turn 明细
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path


def _ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return _dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _rows(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def analyse(path: Path) -> list[dict]:
    """把一个 session 文件拆成若干 turn 的耗时构成。"""

    turns: list[dict] = []
    current: dict | None = None
    pending_call: float | None = None
    for row in _rows(path):
        stamp = _ts(row.get("timestamp"))
        payload = row.get("payload") or {}
        kind = payload.get("type")
        if stamp is None:
            continue
        if kind == "task_started":
            current = {
                "start": stamp,
                "end": None,
                "requests": 0,
                "tool_calls": 0,
                "tool_seconds": 0.0,
                "context_window": payload.get("model_context_window"),
                "input_tokens_last": None,
                "slowest_gaps": [],
                "prev": stamp,
            }
            pending_call = None
            continue
        if current is None:
            continue
        if kind == "token_count":
            current["requests"] += 1
            info = payload.get("info") or {}
            usage = info.get("last_token_usage") or info.get("total_token_usage") or {}
            tokens = usage.get("input_tokens")
            if isinstance(tokens, int):
                current["input_tokens_last"] = tokens
        elif kind == "function_call":
            current["tool_calls"] += 1
            pending_call = stamp
        elif kind == "function_call_output":
            if pending_call is not None:
                current["tool_seconds"] += stamp - pending_call
                pending_call = None
        elif kind == "task_complete":
            current["end"] = stamp
            turns.append(current)
            current = None
            continue
        gap = stamp - current["prev"]
        current["prev"] = stamp
        if gap > 1.0:
            current["slowest_gaps"].append((round(gap, 1), str(kind)))
    if current is not None and current["requests"]:
        current["end"] = current["prev"]
        turns.append(current)
    return turns


def main() -> int:
    parser = argparse.ArgumentParser(description="一轮迭代的耗时归因")
    parser.add_argument("run_root", help="run 目录（含 codex-home/sessions）")
    parser.add_argument("--detail", action="store_true", help="逐 turn 明细")
    args = parser.parse_args()

    sessions = sorted((Path(args.run_root) / "codex-home" / "sessions").rglob("rollout-*.jsonl"))
    if not sessions:
        print("没有找到 codex session 记录")
        return 1

    everything: list[dict] = []
    for path in sessions:
        for turn in analyse(path):
            turn["session"] = path.name[8:27]
            everything.append(turn)

    print(f"{len(sessions)} 个 session，{len(everything)} 个 turn\n")
    header = (
        f"{'session':<20} {'总时长':>8} {'模型':>8} {'工具':>8} "
        f"{'请求数':>6} {'工具数':>6} {'每请求':>7} {'末上下文':>9}"
    )
    print(header)
    print("-" * len(header))
    totals = {"wall": 0.0, "tool": 0.0, "requests": 0, "calls": 0}
    for turn in everything:
        wall = (turn["end"] or turn["start"]) - turn["start"]
        tool = turn["tool_seconds"]
        model = max(0.0, wall - tool)
        per = model / turn["requests"] if turn["requests"] else 0.0
        totals["wall"] += wall
        totals["tool"] += tool
        totals["requests"] += turn["requests"]
        totals["calls"] += turn["tool_calls"]
        context = turn["input_tokens_last"]
        print(
            f"{turn['session']:<20} {wall:>7.0f}s {model:>7.0f}s {tool:>7.0f}s "
            f"{turn['requests']:>6} {turn['tool_calls']:>6} {per:>6.1f}s "
            f"{(f'{context / 1000:.0f}k' if context else '-'):>9}"
        )
        if args.detail and turn["slowest_gaps"]:
            worst = sorted(turn["slowest_gaps"], reverse=True)[:5]
            print("      最慢间隔: " + ", ".join(f"{gap}s@{kind}" for gap, kind in worst))

    model_total = max(0.0, totals["wall"] - totals["tool"])
    print(
        f"\n合计 {totals['wall']:.0f}s = 模型 {model_total:.0f}s "
        f"({model_total / max(totals['wall'], 1) * 100:.0f}%) + 工具 {totals['tool']:.0f}s "
        f"({totals['tool'] / max(totals['wall'], 1) * 100:.0f}%)"
    )
    if totals["requests"]:
        print(
            f"共 {totals['requests']} 次模型请求、{totals['calls']} 次工具调用；"
            f"平均每次模型往返 {model_total / totals['requests']:.1f}s。"
        )
        print(
            "→ 一轮的耗时主要由**请求次数**决定，不是单次请求的大小。"
            "要压缩它，只能减少往返（更少的文件、更集中的材料），而不是换更快的模型。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

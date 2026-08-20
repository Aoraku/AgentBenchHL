#!/usr/bin/env python3
"""诊断：reasoning_effort 对**单次请求延迟**的影响。

为什么需要单独量它
------------------
一轮迭代的墙钟由 ``请求次数 × 单次延迟`` 决定（见 diag_turn_breakdown.py：
实测一轮 30~68 次模型往返，工具执行只占 1%）。而单次延迟里有一块是**看不见的**：
中转站把 ``reasoning_output_tokens`` 一律报成 0，于是模型在 effort=high 下产生的
内部思考既不出现在 ``output_tokens`` 里，也不出现在账单式的统计里，只体现为墙钟。

这个脚本用**同一个小问题**分别打不同 effort，把那块看不见的时间量出来：
输入几乎为零、可见输出也很短，两者的时间差就只能来自内部思考。

用法::

    python3 scripts/diag_effort_latency.py --efforts low,high --repeat 2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request

TASK = (
    "Here is a tower-defense rule: a tower costs 15 coins, deals 20 damage, and an enemy "
    "with 100 HP arrives every 3 rounds. In two short sentences, say how many towers to "
    "build and why."
)


def once(url: str, key: str, model: str, effort: str, attempts: int = 4) -> tuple[float, dict]:
    """打一次请求；中转站会偶发 504（网关超时），重试而不是让整次测量报废。"""

    payload = {"model": model, "input": TASK, "reasoning": {"effort": effort}, "store": False}
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                document = json.loads(response.read())
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(2.0 * (attempt + 1))
            continue
        return time.time() - started, document.get("usage") or {}
    raise RuntimeError(f"{attempts} 次尝试都失败：{last_error}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "ABHL_BASE_URL", "https://lab.cs.tsinghua.edu.cn/ai-platform/sub2api"
        ),
    )
    parser.add_argument("--efforts", default="low,high")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    key = os.environ.get("ABHL_API_KEY")
    if not key:
        print("ABHL_API_KEY 未设置")
        return 2
    url = args.base_url.rstrip("/") + "/v1/responses"
    print(f"model={args.model}  同一个小问题，只变 reasoning_effort\n")
    for effort in [item.strip() for item in args.efforts.split(",") if item.strip()]:
        spans: list[float] = []
        last: dict = {}
        for _ in range(max(1, args.repeat)):
            span, usage = once(url, key, args.model, effort)
            spans.append(span)
            last = usage
        print(
            f"effort={effort:<5} 中位 {statistics.median(spans):5.1f}s "
            f"(样本 {[round(item, 1) for item in spans]})  "
            f"input={last.get('input_tokens')} output={last.get('output_tokens')} "
            f"reasoning_reported={last.get('reasoning_output_tokens')}"
        )
    print(
        "\n输入固定、可见输出都很短，所以时间差只能来自未被计数的内部思考。"
        "一轮的墙钟 ≈ 请求次数 × 单次延迟，effort 是其中最直接的乘数。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

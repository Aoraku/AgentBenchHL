#!/usr/bin/env python3
"""探测中转站对某个模型的**真实**可用上下文上限。

为什么必须实测：模型厂商声明的窗口（glm-5.3 官方 catalog 写 1,048,576）与
第三方中转站实际放行的大小是两件事。我们已经为此付过代价——按"1M"配置的那次，
请求在约 190~206k 处撞墙，然后 codex 走压缩、压缩失败、整个 run 死。

用法::

    python3 scripts/probe_context_limit.py --model glm-5.3 --steps 120,250,400
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request

FILLER = "word "  # 5 字符 ≈ 1~2 token，用重复词避免触发任何缓存/去重


def probe(url: str, key: str, model: str, approx_k_tokens: int, timeout_s: float) -> str:
    # 粗略按 4 字符/token 估算，只需要量级正确。
    body = FILLER * (approx_k_tokens * 1000 * 4 // len(FILLER))
    payload = {
        "model": model,
        "input": body + "\nReply with the single word OK.",
        "reasoning": {"effort": "low"},
        "store": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            document = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read()[:300].decode("utf-8", "replace")
        return f"HTTP {error.code} {detail}"
    except Exception as error:  # noqa: BLE001 - 探测脚本要如实报告任何失败
        return f"{type(error).__name__}: {str(error)[:200]}"
    usage = document.get("usage") or {}
    return (
        f"status={document.get('status')} "
        f"input_tokens={usage.get('input_tokens')} "
        f"elapsed={time.time() - started:.0f}s"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="glm-5.3")
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "ABHL_BASE_URL", "https://lab.cs.tsinghua.edu.cn/ai-platform/sub2api"
        ),
    )
    parser.add_argument("--steps", default="120,250,400", help="逗号分隔的 k tokens 档位")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    key = os.environ.get("ABHL_API_KEY")
    if not key:
        print("ABHL_API_KEY 未设置")
        return 2
    url = args.base_url.rstrip("/") + "/v1/responses"
    print(f"model={args.model} url={url}")
    for step in [int(item) for item in args.steps.split(",") if item.strip()]:
        print(f"  ~{step}k tokens -> {probe(url, key, args.model, step, args.timeout)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

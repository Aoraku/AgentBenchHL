"""Claude Code（cc）harness 适配器。

- :mod:`runtime`          Claude Code headless 驱动的 Agent 运行时
- :mod:`anthropic_bridge` Anthropic ↔ OpenAI 协议桥（中转站禁用 /v1/messages）
"""

from __future__ import annotations

from agentbench_hl.adapters.cc_goal.anthropic_bridge import AnthropicBridge
from agentbench_hl.adapters.cc_goal.runtime import (
    ClaudeCodeRuntime,
    ClaudeCodeUnavailable,
    claude_code_runtime,
    probe_claude_installation,
)

__all__ = [
    "AnthropicBridge",
    "ClaudeCodeRuntime",
    "ClaudeCodeUnavailable",
    "claude_code_runtime",
    "probe_claude_installation",
]

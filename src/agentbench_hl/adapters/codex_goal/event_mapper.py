"""Map Codex App Server notifications without inventing unavailable usage."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

_CREDENTIAL = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


@dataclass(frozen=True)
class MappedAgentEvent:
    event_type: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _clean(value: object) -> object:
    if isinstance(value, str):
        return _CREDENTIAL.sub("[REDACTED]", value)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if any(token in name.lower() for token in ("api_key", "token", "secret", "password")):
                if name not in {"tokenUsage", "tokensUsed", "tokenBudget"}:
                    result[name] = "[REDACTED]"
                    continue
            result[name] = _clean(item)
        return result
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def map_app_server_event(raw: Mapping[str, object]) -> MappedAgentEvent:
    if raw.get("type") == "usage":
        return MappedAgentEvent(
            "AgentUsageObserved",
            {
                "input_tokens": _optional_int(raw.get("input_tokens")),
                "cached_input_tokens": _optional_int(raw.get("cached_input_tokens")),
                "output_tokens": _optional_int(raw.get("output_tokens")),
                "reasoning_tokens": _optional_int(raw.get("reasoning_tokens")),
                "total_tokens": _optional_int(raw.get("total_tokens")),
            },
        )
    method = str(raw.get("method", "unknown"))
    params = raw.get("params")
    clean_params = _clean(params if isinstance(params, Mapping) else {})
    assert isinstance(clean_params, dict)
    if method == "thread/tokenUsage/updated":
        token_usage = params.get("tokenUsage", {}) if isinstance(params, Mapping) else {}
        last = token_usage.get("last", {}) if isinstance(token_usage, Mapping) else {}
        if not isinstance(last, Mapping):
            last = {}
        return MappedAgentEvent(
            "AgentUsageObserved",
            {
                "thread_id": params.get("threadId") if isinstance(params, Mapping) else None,
                "turn_id": params.get("turnId") if isinstance(params, Mapping) else None,
                "input_tokens": _optional_int(last.get("inputTokens")),
                "cached_input_tokens": _optional_int(last.get("cachedInputTokens")),
                "output_tokens": _optional_int(last.get("outputTokens")),
                "reasoning_tokens": _optional_int(last.get("reasoningOutputTokens")),
                "total_tokens": _optional_int(last.get("totalTokens")),
            },
        )
    event_type = {
        "thread/started": "AgentThreadStarted",
        "thread/goal/updated": "AgentGoalUpdated",
        "thread/goal/cleared": "AgentGoalCleared",
        "turn/started": "AgentTurnStarted",
        "turn/completed": "AgentTurnCompleted",
        "item/started": "AgentItemStarted",
        "item/completed": "AgentItemCompleted",
        "thread/compacted": "AgentContextCompacted",
        "error": "AgentRuntimeError",
        "protocol/error": "AgentProtocolError",
    }.get(method, "AgentNotificationObserved")
    return MappedAgentEvent(event_type, {"method": method, "params": clean_params})

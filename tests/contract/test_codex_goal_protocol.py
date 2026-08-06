from __future__ import annotations

from agentbench_hl.adapters.codex_goal.event_mapper import map_app_server_event


def test_runtime_maps_usage_without_inventing_missing_tokens() -> None:
    event = map_app_server_event({"type": "usage", "input_tokens": 10})

    assert event.payload["input_tokens"] == 10
    assert event.payload["output_tokens"] is None
    assert event.payload["reasoning_tokens"] is None


def test_runtime_maps_official_token_notification_fields() -> None:
    event = map_app_server_event(
        {
            "method": "thread/tokenUsage/updated",
            "params": {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "tokenUsage": {
                    "last": {
                        "inputTokens": 10,
                        "cachedInputTokens": 4,
                        "outputTokens": 3,
                        "reasoningOutputTokens": 2,
                        "totalTokens": 13,
                    },
                    "total": {
                        "inputTokens": 10,
                        "cachedInputTokens": 4,
                        "outputTokens": 3,
                        "reasoningOutputTokens": 2,
                        "totalTokens": 13,
                    },
                },
            },
        }
    )

    assert event.event_type == "AgentUsageObserved"
    assert event.payload["cached_input_tokens"] == 4
    assert event.payload["reasoning_tokens"] == 2

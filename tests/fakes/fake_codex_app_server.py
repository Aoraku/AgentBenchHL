from __future__ import annotations

import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
request_log_path = log_path.with_suffix(".requests.jsonl")
mode = sys.argv[2] if len(sys.argv) > 2 else "complete"


def send(value: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if "id" not in message:
        continue
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(str(method) + "\n")
    with request_log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(message, separators=(",", ":")) + "\n")
    request_id = message["id"]
    params = message.get("params", {})
    if method == "initialize":
        result = {
            "codexHome": "/fake/codex-home",
            "platformFamily": "unix",
            "platformOs": "macos",
            "userAgent": "fake-app-server",
        }
    elif method in {"thread/start", "thread/resume"}:
        result = {"thread": {"id": params.get("threadId", "thread-1")}}
    elif method == "thread/goal/set":
        result = {
            "goal": {
                "threadId": params["threadId"],
                "objective": params.get("objective") or "fixture objective",
                "status": params.get("status") or "active",
                "tokenBudget": params.get("tokenBudget"),
                "tokensUsed": 0,
                "timeUsedSeconds": 0,
                "createdAt": 1,
                "updatedAt": 1,
            }
        }
    elif method == "thread/goal/get":
        result = {
            "goal": {
                "threadId": params["threadId"],
                "objective": "fixture objective",
                "status": "blocked" if mode == "blocked" else "active",
                "tokenBudget": None,
                "tokensUsed": 0,
                "timeUsedSeconds": 0,
                "createdAt": 1,
                "updatedAt": 1,
            }
        }
    elif method == "thread/memoryMode/set":
        result = {}
    elif method == "turn/start":
        result = {"turn": {"id": "turn-1", "items": [], "status": "inProgress"}}
    else:
        send(
            {
                "id": request_id,
                "error": {"code": -32601, "message": f"unknown method {method}"},
            }
        )
        continue
    send({"id": request_id, "result": result})
    if method == "thread/resume" and mode == "stale_blocked":
        send(
            {
                "method": "thread/goal/updated",
                "params": {
                    "threadId": params["threadId"],
                    "goal": {"status": "blocked"},
                },
            }
        )
    if method == "thread/resume" and mode == "stale_complete":
        send(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": params["threadId"],
                    "turn": {"id": "old-turn", "status": "completed", "items": []},
                },
            }
        )
    if method == "turn/start":
        send(
            {
                "method": "turn/started",
                "params": {
                    "threadId": params["threadId"],
                    "turn": {"id": "turn-1", "status": "inProgress"},
                },
            }
        )
        if mode == "failed_turn":
            send(
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": params["threadId"],
                        "turn": {
                            "id": "turn-1",
                            "status": "failed",
                            "error": {"message": "stream disconnected before completion"},
                            "items": [],
                        },
                    },
                }
            )
            continue
        if mode == "blocked":
            send(
                {
                    "method": "error",
                    "params": {"message": "API key is required by the fixture provider"},
                }
            )
            send(
                {
                    "method": "thread/goal/updated",
                    "params": {
                        "threadId": params["threadId"],
                        "goal": {"status": "blocked"},
                    },
                }
            )
            continue
        send(
            {
                "method": "thread/tokenUsage/updated",
                "params": {
                    "threadId": params["threadId"],
                    "turnId": "turn-1",
                    "tokenUsage": {
                        "last": {
                            "inputTokens": 11,
                            "cachedInputTokens": 4,
                            "outputTokens": 6,
                            "reasoningOutputTokens": 2,
                            "totalTokens": 17,
                        }
                    },
                },
            }
        )
        send(
            {
                "method": "turn/completed",
                "params": {
                    "threadId": params["threadId"],
                    "turn": {"id": "turn-1", "status": "completed", "items": []},
                },
            }
        )

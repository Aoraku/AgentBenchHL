"""Anthropic ↔ OpenAI 桥接代理 —— 让 Claude Code 能用清华中转站。

**为什么需要**：Claude Code 只会说 Anthropic 的 ``/v1/messages`` 协议，而中转站实测
明确拒绝该端点：

```
POST /v1/messages -> 403 {"error":{"message":"This group does not allow /v1/messages dispatch"}}
POST /v1/chat/completions -> 200
```

于是本模块在本机起一个只监听回环地址的小代理：对外说 Anthropic 协议，对内翻译成
``/v1/chat/completions``（含流式 SSE 与工具调用）。Claude Code 侧只需设置
``ANTHROPIC_BASE_URL=http://127.0.0.1:<port>``。

**边界（诚实声明）**：
- 只翻译 Claude Code 实际会用到的字段：system / messages / tools / tool_choice /
  max_tokens / temperature / stream / stop_sequences。
- ``thinking`` 块无法从 OpenAI 侧还原，会被忽略（cc harness 的推理深度因此不完全等价
  于 codex 的 reasoning_effort，跨 harness 比较时必须注明）。
- 计费/用量字段按上游返回值透传；上游不给就置 0（绝不编造）。
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from agentbench_hl.adapters.http_retry import urlopen_with_backoff

TOKEN_ESTIMATE_DIVISOR = 4  # 仅在上游完全不返回 usage 时用于粗估（会标注 estimated）


# ---------------------------------------------------------------- 协议翻译


def _text_of(content: object, *, include_tool_results: bool = True) -> str:
    """把 Anthropic 的 content（字符串或块数组）拍平成纯文本。"""

    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, Mapping):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind == "tool_result" and include_tool_results:
            payload = block.get("content")
            parts.append(
                f"[tool_result {block.get('tool_use_id')}] {_text_of(payload) if payload else ''}"
            )
    return "\n".join(part for part in parts if part)


def anthropic_to_openai(request: Mapping[str, Any]) -> dict[str, Any]:
    """Anthropic ``/v1/messages`` 请求 → OpenAI ``/v1/chat/completions`` 请求。"""

    messages: list[dict[str, Any]] = []
    system = request.get("system")
    if system:
        messages.append({"role": "system", "content": _text_of(system)})
    for message in request.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")
        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(block.get("id") or uuid.uuid4().hex),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name") or ""),
                                "arguments": json.dumps(
                                    block.get("input") or {}, ensure_ascii=False
                                ),
                            },
                        }
                    )
                elif block.get("type") == "tool_result":
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(block.get("tool_use_id") or ""),
                            "content": _text_of(block.get("content")),
                        }
                    )
        # tool_result 已单独作为 tool 消息发出，普通文本里不再重复它。
        text = _text_of(content, include_tool_results=not tool_results)
        if tool_results:
            # tool 结果必须作为独立的 tool 消息，且要在 assistant 工具调用之后。
            messages.extend(tool_results)
            if text:
                messages.append({"role": role, "content": text})
        elif tool_calls:
            messages.append(
                {"role": "assistant", "content": text or None, "tool_calls": tool_calls}
            )
        else:
            messages.append({"role": role, "content": text})

    payload: dict[str, Any] = {
        "model": request.get("model"),
        "messages": messages,
        "stream": bool(request.get("stream")),
    }
    if request.get("max_tokens"):
        payload["max_tokens"] = int(request["max_tokens"])
    if request.get("temperature") is not None:
        payload["temperature"] = request["temperature"]
    if request.get("stop_sequences"):
        payload["stop"] = request["stop_sequences"]
    tools = request.get("tools")
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description") or "",
                    "parameters": tool.get("input_schema") or {"type": "object"},
                },
            }
            for tool in tools
            if isinstance(tool, Mapping)
        ]
        choice = request.get("tool_choice")
        if isinstance(choice, Mapping):
            kind = choice.get("type")
            if kind == "any":
                payload["tool_choice"] = "required"
            elif kind == "tool" and choice.get("name"):
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": choice["name"]},
                }
            else:
                payload["tool_choice"] = "auto"
    return payload


def openai_to_anthropic(response: Mapping[str, Any], model: str) -> dict[str, Any]:
    """OpenAI 非流式响应 → Anthropic ``Message``。"""

    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    blocks: list[dict[str, Any]] = []
    text = message.get("content")
    if text:
        blocks.append({"type": "text", "text": str(text)})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": function.get("arguments")}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or uuid.uuid4().hex),
                "name": str(function.get("name") or ""),
                "input": arguments,
            }
        )
    usage = response.get("usage") or {}
    finish = choice.get("finish_reason") or "stop"
    return {
        "id": str(response.get("id") or f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks or [{"type": "text", "text": ""}],
        "stop_reason": {"tool_calls": "tool_use", "length": "max_tokens"}.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def sse_events(message: Mapping[str, Any]) -> Iterator[str]:
    """把一条完整 Anthropic Message 拆成 Claude Code 期望的 SSE 事件序列。"""

    def emit(event: str, data: Mapping[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    skeleton = {key: value for key, value in message.items() if key != "content"}
    skeleton["content"] = []
    yield emit("message_start", {"type": "message_start", "message": skeleton})
    for index, block in enumerate(message.get("content") or []):
        if block.get("type") == "text":
            yield emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            yield emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block.get("text") or ""},
                },
            )
        else:
            yield emit(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": {},
                    },
                },
            )
            yield emit(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                },
            )
        yield emit("content_block_stop", {"type": "content_block_stop", "index": index})
    yield emit(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": message.get("stop_reason"),
                "stop_sequence": None,
            },
            "usage": message.get("usage") or {},
        },
    )
    yield emit("message_stop", {"type": "message_stop"})


# -------------------------------------------------------------------- 服务


class _Handler(BaseHTTPRequestHandler):
    server_version = "AgentBenchAnthropicBridge/1.0"
    protocol_version = "HTTP/1.1"

    # 由 AnthropicBridge 注入
    upstream_base: str = ""
    api_key: str = ""
    upstream_timeout: float = 600.0
    log_path: Any = None

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - 覆盖标准库
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(f"{time.time():.0f} {fmt % args}\n")

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802 - 标准库命名
        """Claude Code 用 ``HEAD /api/hello`` 做连通性探测。

        返回 501 会让它在重试里挂死（实测：turn 已完成却不退出）。这里一律回 200。
        """

        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - 标准库命名
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._json(200, {"ok": True, "upstream": self.upstream_base})
            return
        if "/api/hello" in self.path:
            self._json(200, {"ok": True})
            return
        self._json(404, {"error": {"type": "not_found", "message": self.path}})

    def do_POST(self) -> None:  # noqa: N802,C901 - 标准库命名
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw or b"{}")
        except json.JSONDecodeError as error:
            self._json(400, {"error": {"type": "invalid_request_error", "message": str(error)}})
            return

        path = self.path.split("?")[0].rstrip("/")
        if path.endswith("/count_tokens"):
            # Claude Code 会用它做上下文预算；没有上游支持时给出确定性估算并标注。
            text = json.dumps(request.get("messages") or [], ensure_ascii=False)
            self._json(
                200,
                {"input_tokens": max(1, len(text) // TOKEN_ESTIMATE_DIVISOR), "estimated": True},
            )
            return
        if not path.endswith("/messages"):
            self._json(404, {"error": {"type": "not_found", "message": self.path}})
            return

        model = str(request.get("model") or "")
        streaming = bool(request.get("stream"))
        payload = anthropic_to_openai(request)
        payload["stream"] = False  # 上游一次拿完，再按 Anthropic 事件流回放
        try:
            upstream = urllib.request.Request(
                f"{self.upstream_base.rstrip('/')}/v1/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            # 限流/上游抖动先由退避重试吸收，再决定是否把错误翻译给 CLI：
            # 一次 429 直接落地 = 这一轮 Goal turn 白跑，且在账本里像是模型问题。
            handle = urlopen_with_backoff(
                upstream,
                timeout_s=self.upstream_timeout,
                log=lambda message: print(message, file=sys.stderr),
            )
            if isinstance(handle, urllib.error.HTTPError):
                raise handle
            with handle:
                body = json.loads(handle.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="ignore")[:600]
            self._json(
                error.code,
                {"type": "error", "error": {"type": "api_error", "message": detail}},
            )
            return
        except Exception as error:  # noqa: BLE001 - 网络异常统一翻译成 Anthropic 错误
            self._json(
                502,
                {"type": "error", "error": {"type": "api_error", "message": str(error)}},
            )
            return

        message = openai_to_anthropic(body, model)
        if not streaming:
            self._json(200, message)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # HTTP/1.1 下 SSE 既没有 Content-Length 也没走 chunked，必须显式关闭连接，
        # 否则客户端会在 message_stop 之后继续等数据（实测：turn 完成却不退出）。
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        for chunk in sse_events(message):
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()


class AnthropicBridge:
    """本机回环上的 Anthropic→OpenAI 桥接代理（run 级生命周期）。"""

    def __init__(
        self,
        *,
        upstream_base: str,
        api_key: str,
        host: str = "127.0.0.1",
        port: int = 0,
        log_path: Any = None,
        upstream_timeout: float = 600.0,
    ) -> None:
        handler = type(
            "BoundHandler",
            (_Handler,),
            {
                "upstream_base": upstream_base,
                "api_key": api_key,
                "log_path": log_path,
                "upstream_timeout": upstream_timeout,
            },
        )
        self._server = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def __enter__(self) -> AnthropicBridge:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

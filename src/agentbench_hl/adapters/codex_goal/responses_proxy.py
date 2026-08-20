"""Small localhost bridge for custom Responses-compatible providers."""

from __future__ import annotations

import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, urlunsplit

from agentbench_hl.adapters.http_retry import request_bytes_with_backoff


class ResponsesCompatProxy:
    def __init__(self, upstream_base_url: str, *, timeout_s: float = 120.0) -> None:
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Responses compatibility proxy is not running")
        return f"http://127.0.0.1:{self._server.server_port}"

    def start(self) -> None:
        if self._server is not None:
            return
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                upstream = urlsplit(owner.upstream_base_url)
                path = upstream.path.rstrip("/") + self.path
                url = urlunsplit((upstream.scheme, upstream.netloc, path, "", ""))
                headers = {
                    name: value
                    for name, value in self.headers.items()
                    if name.lower()
                    not in {"host", "content-length", "connection", "accept-encoding"}
                }
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                # 限流/上游抖动会被退避重试吸收；不这么做的话，一次 429 就等于
                # 这一轮 Goal turn 白跑，事件账本上却记成"模型没产出候选"。
                #
                # 用 request_bytes_with_backoff 而不是 urlopen_with_backoff：
                # glm 的长响应经常在**读体**阶段断流（IncompleteRead），那时
                # urlopen 早已返回、退避层已经退场。实测 antwar 就是这样死的
                # （502 重试成功后读 205 KB 响应体时断掉，异常冒到 HTTP 线程，
                # 整个 run 终止）。把读体一起纳入重试才有原子语义。
                try:
                    upstream_response = request_bytes_with_backoff(
                        request,
                        timeout_s=owner.timeout_s,
                        log=lambda message: print(message, file=sys.stderr),
                    )
                except Exception as error:  # noqa: BLE001 - 必须翻译成 HTTP 响应
                    # 重试耗尽也不能让异常冒到 ThreadingHTTPServer：那会打死整个
                    # turn。翻译成 502 交回 codex CLI，由它自己的重试/报错路径处理。
                    detail = json.dumps(
                        {
                            "error": {
                                "type": "upstream_unavailable",
                                "message": f"{type(error).__name__}: {error}",
                            }
                        }
                    ).encode("utf-8")
                    print(
                        f"[llm-retry] 放弃：{type(error).__name__}: {error}",
                        file=sys.stderr,
                    )
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(detail)))
                    self.end_headers()
                    self.wfile.write(detail)
                    return
                payload = upstream_response.body
                self.send_response(upstream_response.status)
                content_type = upstream_response.content_type
                if content_type:
                    self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server.server_close()
        self._server = None
        self._thread = None

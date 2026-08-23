"""Small localhost bridge for custom Responses-compatible providers."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, urlunsplit

from agentbench_hl.adapters.http_retry import request_bytes_with_backoff


class ResponsesCompatProxy:
    #: 上游拒绝时要打印哪些请求头。
    #:
    #: 为什么需要这个：有些中转站按**客户端指纹**放行（实测 sbtunnel 返回
    #: ``403 This account only allows Codex official clients``）。这类失败里，
    #: 端点、key、模型名全都是对的，唯一的线索是 codex 发的 ``originator`` /
    #: ``user-agent`` —— 而它们默认不落盘，于是排查只能靠猜（我们为此把
    #: config.toml 的每一项都单独试了一遍，全部无关）。
    #:
    #: 只在**非 2xx** 时打印，且只打印这几个非敏感头（绝不打印 authorization）。
    _FINGERPRINT_HEADERS = (
        "originator",
        "user-agent",
        "x-codex-beta-features",
        "x-openai-internal-codex-responses-lite",
    )

    def __init__(self, upstream_base_url: str, *, timeout_s: float = 120.0) -> None:
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        # 退避上**累计等了多久**，以及被吸收掉的可重试失败次数。
        #
        # 为什么必须暴露出来：checkpoint 超时是一个**固定墙钟**
        # （app_server 的 checkpoint_timeout_s），而 503 退避的等待会直接
        # 吃掉它。实测 random 组死在第 26 轮：
        #
        #     TimeoutError: Codex Goal did not reach a checkpoint in time
        #
        # 而当时日志里是连续十几次 503 —— agent 根本没有超时，是限流把
        # checkpoint 预算耗光了。这类死法最坏的地方是它**归错了因**：
        # 看事件账本会以为"模型这一轮没产出候选"。
        #
        # 所以 app_server 会读这个计数器，把退避等待从 deadline 里扣掉。
        self._backoff_lock = threading.Lock()
        self._backoff_seconds = 0.0
        self._absorbed_failures = 0

    @property
    def backoff_seconds(self) -> float:
        """至今在退避上累计等待的秒数（供 checkpoint deadline 补偿）。"""

        with self._backoff_lock:
            return self._backoff_seconds

    @property
    def absorbed_failures(self) -> int:
        """被退避吸收掉的可重试失败次数（限流/上游抖动）。"""

        with self._backoff_lock:
            return self._absorbed_failures

    def _record_backoff(self, seconds: float, failures: int) -> None:
        with self._backoff_lock:
            self._backoff_seconds += max(0.0, seconds)
            self._absorbed_failures += max(0, failures)

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
                    retried = 0

                    def _log(message: str) -> None:
                        nonlocal retried
                        retried += 1
                        print(message, file=sys.stderr)

                    attempt_started = time.monotonic()
                    upstream_response = request_bytes_with_backoff(
                        request,
                        timeout_s=owner.timeout_s,
                        log=_log,
                    )
                    if retried:
                        # 只在真的退避过时记账，避免把正常请求的耗时也算进去。
                        owner._record_backoff(time.monotonic() - attempt_started, retried)
                except Exception as error:  # noqa: BLE001 - 必须翻译成 HTTP 响应
                    owner._record_backoff(time.monotonic() - attempt_started, 1)
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
                if not 200 <= upstream_response.status < 300:
                    # 上游拒绝：把客户端指纹打进 stderr（会进 app-server 日志）。
                    # 这是"端点/key/模型都对但仍被拒"这类故障唯一的线索来源。
                    fingerprint = {
                        name: headers[key]
                        for name in owner._FINGERPRINT_HEADERS
                        for key in headers
                        if key.lower() == name
                    }
                    print(
                        f"[llm-upstream] {upstream_response.status} {url} "
                        f"fingerprint={json.dumps(fingerprint, ensure_ascii=False)} "
                        f"body={payload[:400].decode('utf-8', 'replace')}",
                        file=sys.stderr,
                    )
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

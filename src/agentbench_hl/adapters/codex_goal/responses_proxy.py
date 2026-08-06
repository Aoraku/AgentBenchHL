"""Small localhost bridge for custom Responses-compatible providers."""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, urlunsplit


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
                try:
                    response = urllib.request.urlopen(request, timeout=owner.timeout_s)
                except urllib.error.HTTPError as error:
                    response = error
                try:
                    payload = response.read()
                    self.send_response(response.status)
                    content_type = response.headers.get("Content-Type")
                    if content_type:
                        self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                finally:
                    response.close()

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

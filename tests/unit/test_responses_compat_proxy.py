from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agentbench_hl.adapters.codex_goal.responses_proxy import ResponsesCompatProxy


def test_proxy_forwards_responses_request_and_sse_body() -> None:
    seen: dict[str, object] = {}

    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen["path"] = self.path
            seen["authorization"] = self.headers.get("Authorization")
            seen["body"] = json.loads(body)
            payload = b'data: {"type":"response.completed"}\n\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    proxy = ResponsesCompatProxy(f"http://127.0.0.1:{upstream.server_port}/v1", timeout_s=5)
    try:
        proxy.start()
        request = urllib.request.Request(
            f"{proxy.base_url}/responses",
            data=b'{"model":"gpt-test","stream":true}',
            headers={
                "Authorization": "Bearer test-secret",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "text/event-stream"
            assert response.read() == b'data: {"type":"response.completed"}\n\n'
    finally:
        proxy.close()
        upstream.shutdown()
        upstream_thread.join(timeout=2)
        upstream.server_close()

    assert seen == {
        "path": "/v1/responses",
        "authorization": "Bearer test-secret",
        "body": {"model": "gpt-test", "stream": True},
    }

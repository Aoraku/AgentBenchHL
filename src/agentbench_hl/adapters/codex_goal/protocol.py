"""Bounded newline-delimited JSON-RPC client for Codex App Server."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_CREDENTIAL = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


class AppServerProtocolError(RuntimeError):
    pass


class JsonRpcStdioClient:
    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        stderr_path: Path,
        secrets: tuple[str, ...] = (),
        request_timeout_s: float = 30.0,
    ) -> None:
        self.command = tuple(str(item) for item in command)
        self.request_timeout_s = float(request_timeout_s)
        self.stderr_path = stderr_path
        self.secrets = tuple(item for item in secrets if item)
        self._next_request_id = 1
        self._responses: dict[int, dict[str, Any]] = {}
        self._condition = threading.Condition()
        self.notifications: queue.Queue[dict[str, Any]] = queue.Queue()
        self._closed = False
        self.process = subprocess.Popen(
            self.command,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
        )
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            self.process.terminate()
            raise AppServerProtocolError("failed to open Codex App Server stdio")
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()

    def _send(self, message: Mapping[str, object]) -> None:
        if self.process.stdin is None or self.process.poll() is not None:
            raise AppServerProtocolError("Codex App Server is not running")
        self.process.stdin.write(
            json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        self.process.stdin.flush()

    def request(
        self,
        method: str,
        params: Mapping[str, object],
        *,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]:
        with self._condition:
            request_id = self._next_request_id
            self._next_request_id += 1
        self._send({"id": request_id, "method": method, "params": dict(params)})
        deadline = time.monotonic() + (
            self.request_timeout_s if timeout_s is None else float(timeout_s)
        )
        with self._condition:
            while request_id not in self._responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Codex App Server request timed out: {method}")
                if self._closed:
                    raise AppServerProtocolError(
                        f"Codex App Server closed while waiting for {method}"
                    )
                self._condition.wait(timeout=remaining)
            response = self._responses.pop(request_id)
        if "error" in response:
            raise AppServerProtocolError(
                f"Codex App Server rejected {method}: {self._redact(str(response['error']))}"
            )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise AppServerProtocolError(f"Codex App Server returned invalid {method} result")
        return result

    def notify(self, method: str, params: Mapping[str, object] | None = None) -> None:
        message: dict[str, object] = {"method": method}
        if params is not None:
            message["params"] = dict(params)
        self._send(message)

    def next_notification(self, timeout_s: float) -> dict[str, Any]:
        try:
            return self.notifications.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for Codex App Server notification") from exc

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for raw_line in self.process.stdout:
                if not raw_line.strip():
                    continue
                try:
                    message = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    self.notifications.put(
                        {
                            "method": "protocol/error",
                            "params": {"message": f"invalid JSON: {exc}"},
                        }
                    )
                    continue
                if not isinstance(message, dict):
                    continue
                if "id" in message and "method" not in message:
                    with self._condition:
                        self._responses[int(message["id"])] = message
                        self._condition.notify_all()
                elif "id" in message and "method" in message:
                    self._send(
                        {
                            "id": message["id"],
                            "error": {
                                "code": -32601,
                                "message": "server request unsupported by isolated runtime",
                            },
                        }
                    )
                else:
                    self.notifications.put(message)
        finally:
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def _redact(self, value: str) -> str:
        redacted = _CREDENTIAL.sub("[REDACTED]", value)
        for secret in self.secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        return redacted

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with self.stderr_path.open("a", encoding="utf-8") as stream:
            for line in self.process.stderr:
                stream.write(self._redact(line))
                stream.flush()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        with self._condition:
            self._closed = True
            self._condition.notify_all()

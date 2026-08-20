"""网络命名空间级隔离 + 单点放通：UDS 网关。

**要解决的问题**：cc harness（Claude Code）必须能访问本机的 Anthropic 桥接代理，
所以此前它的沙箱只能 ``allow_network=True``——那等于把整个互联网都放开了，
"禁联网"这条科学前提在 cc 上只靠"禁用 WebFetch/WebSearch 工具"兜着（弱）。

**做法**（完全不需要 root，与 codex 侧的 ``--unshare-net`` 强度一致）：

```
claude(沙箱内, 独立 netns, 只有 lo)
   └─ TCP 127.0.0.1:<sandbox_port>            ← 沙箱内的 relay 进程在听
        └─ AF_UNIX <agent_home>/bridge.sock   ← bind-mount 进沙箱的 unix socket
             └─ 宿主 UdsGateway
                  └─ TCP 127.0.0.1:<bridge_port>（Anthropic 桥接代理）
```

关键点：unix socket 不属于任何网络命名空间，它是**文件系统对象**，因此
``--unshare-net`` 之后沙箱里除了这一个 socket 文件之外**没有任何**出网路径：

- 没有默认路由、没有 DNS、连不上任何 IP（包括宿主 127.0.0.1 上的其它端口）；
- 放通的目标由宿主侧网关**硬编码**成桥接代理的地址，沙箱无法改写。

因此 cc 的网络隔离等级从 ``tools_only`` 升级为 ``netns+uds_gateway``。
"""

from __future__ import annotations

import os
import socket
import socketserver
import threading
from dataclasses import dataclass, field
from pathlib import Path

RELAY_LAUNCHER_NAME = "netns_relay_launcher.py"
DEFAULT_SANDBOX_PORT = 8118
_CHUNK = 65536


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    try:
        while True:
            chunk = source.recv(_CHUNK)
            if not chunk:
                break
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class _GatewayServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    target: tuple[str, int] = ("127.0.0.1", 0)


class _GatewayHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        target: tuple[str, int] = self.server.target  # type: ignore[attr-defined]
        try:
            upstream = socket.create_connection(target, timeout=30)
        except OSError:
            self.request.close()
            return
        upstream.settimeout(None)
        pump = threading.Thread(target=_pipe, args=(self.request, upstream), daemon=True)
        pump.start()
        _pipe(upstream, self.request)
        pump.join(timeout=5)
        upstream.close()


@dataclass
class UdsGateway:
    """宿主侧网关：unix socket → 固定的一个 TCP 目标（不可被沙箱改写）。"""

    socket_path: Path
    target_host: str
    target_port: int
    _server: _GatewayServer | None = field(default=None, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def start(self) -> UdsGateway:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        server = _GatewayServer(str(self.socket_path), _GatewayHandler)
        server.target = (self.target_host, self.target_port)
        # 沙箱内以同一个 uid 运行，0600 足够；避免同机其它用户借道出网。
        os.chmod(self.socket_path, 0o600)
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def describe(self) -> dict[str, object]:
        return {
            "gateway": "uds",
            "socket_path": str(self.socket_path),
            "allowed_target": f"{self.target_host}:{self.target_port}",
        }

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

    def __enter__(self) -> UdsGateway:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()


_RELAY_LAUNCHER_SOURCE = '''"""沙箱内的 TCP→UDS relay + 子进程启动器（由 uds_gateway 自动落盘）。

在**独立网络命名空间**里监听 127.0.0.1:<port>，把连接转发到 bind-mount 进来的
unix socket，再由宿主网关转发到唯一放通的目标。子进程（claude）只看到一个
本地回环端口，其它任何出网尝试都会失败。
"""

import argparse
import socket
import socketserver
import subprocess
import sys
import threading

CHUNK = 65536


def pipe(source, sink):
    try:
        while True:
            chunk = source.recv(CHUNK)
            if not chunk:
                break
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            upstream.connect(self.server.uds_path)
        except OSError:
            self.request.close()
            return
        pump = threading.Thread(target=pipe, args=(self.request, upstream), daemon=True)
        pump.start()
        pipe(upstream, self.request)
        pump.join(timeout=5)
        upstream.close()


class Relay(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    uds_path = ""


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--uds", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv[1:])
    command = options.command[1:] if options.command[:1] == ["--"] else options.command
    if not command:
        print("netns relay launcher: missing child command", file=sys.stderr)
        return 2
    server = Relay((options.host, options.port), Handler)
    server.uds_path = options.uds
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        completed = subprocess.run(command, check=False)
    finally:
        server.shutdown()
        server.server_close()
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
'''


def install_relay_launcher(directory: Path) -> Path:
    """把沙箱内 relay 启动器落盘到 ``directory``，返回脚本路径。"""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RELAY_LAUNCHER_NAME
    path.write_text(_RELAY_LAUNCHER_SOURCE, encoding="utf-8")
    return path


def relay_command(
    launcher: Path,
    socket_path: Path,
    sandbox_port: int,
    command: tuple[str, ...],
    *,
    python_executable: str,
) -> tuple[str, ...]:
    """把 ``command`` 包装成"先起 relay 再跑它"。"""

    return (
        python_executable,
        str(launcher),
        "--uds",
        str(socket_path),
        "--port",
        str(sandbox_port),
        "--",
        *command,
    )

"""Claude Code harness（cc）—— 与 codex harness 平行的第二种 Agent 运行时。

对齐点（与 `codex_goal` 一致的部分）：
- 实现同一个 :class:`AgentRuntime` 契约（start / resume / run_until_checkpoint / pause）；
- 事件映射成同构的 ``MappedAgentEvent``，因此 GoalLedService 完全复用，
  token 用量也会落到同一条 events.jsonl；
- 文件隔离用同一套 `adapters/isolation`（bubblewrap/Seatbelt）：GamePack 只读、
  workspace/research 可写、人类源码与凭据被遮蔽。

**必须诚实标注的差异**（写进 run 清单，跨 harness 比较时必读）：
1. **网络**：codex 在 app-server 层用 permissions 关掉全部网络；Claude Code 没有
   等价开关，这里用**命名空间级断网 + UDS 网关定点放通**补齐：沙箱
   ``--unshare-net``（只有 lo），出网唯一通道是 bind-mount 进来的 unix socket，
   由宿主网关硬编码转发到本机桥接代理（见 :mod:`..isolation.uds_gateway`）。
   隔离等级记为 ``netns+uds_gateway``；只有在隔离后端不可用时才退化为
   ``tools_only``（禁用 WebFetch/WebSearch），且会记录 ``AgentIsolationUnavailable``。
2. **推理深度**：codex 的 ``reasoning_effort`` 无对应项，cc 侧不设置。
3. **协议桥接**：中转站禁用 ``/v1/messages``，故经本地
   :mod:`anthropic_bridge` 翻译到 ``/v1/chat/completions``；``thinking`` 块无法还原。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentbench_hl.adapters.codex_goal.event_mapper import MappedAgentEvent
from agentbench_hl.adapters.isolation import select_candidate_isolation
from agentbench_hl.adapters.isolation.uds_gateway import (
    DEFAULT_SANDBOX_PORT,
    UdsGateway,
    install_relay_launcher,
    relay_command,
)
from agentbench_hl.config import ExperimentConfig
from agentbench_hl.ports.agent_runtime import AgentSession, CheckpointPredicate, RunContext
from agentbench_hl.ports.isolation import IsolationRequest

DISALLOWED_TOOLS = ("WebSearch", "WebFetch")


class ClaudeCodeUnavailable(RuntimeError):
    """cc harness 不可用（缺二进制 / 版本不支持 headless 流式输出）。"""


def probe_claude_installation(binary: str) -> dict[str, object]:
    """检查 claude 可执行文件与 headless 能力。"""

    path = Path(binary)
    if not path.exists():
        raise ClaudeCodeUnavailable(f"claude binary not found: {binary}")
    try:
        completed = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except OSError as error:
        raise ClaudeCodeUnavailable(f"cannot execute {binary}: {error}") from error
    version = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode != 0 or not version:
        raise ClaudeCodeUnavailable(f"claude --version failed: {version!r}")
    help_text = subprocess.run(
        [binary, "--help"], capture_output=True, text=True, timeout=30, check=False
    ).stdout
    for flag in ("--output-format", "--print"):
        if flag not in help_text:
            raise ClaudeCodeUnavailable(f"claude build lacks {flag}; headless mode unavailable")
    return {
        "version": version,
        "supports_session_id": "--session-id" in help_text,
        "supports_append_system_prompt": "--append-system-prompt" in help_text,
        "supports_add_dir": "--add-dir" in help_text,
        "supports_permission_mode": "--permission-mode" in help_text,
    }


@dataclass
class ClaudeCodeRuntime:
    """用 Claude Code 的 headless 模式驱动一个持久 Goal 会话。"""

    binary: str
    agent_home: Path
    base_url: str
    api_key: str
    model: str
    isolation_backend: str = "auto"
    permission_mode: str = "acceptEdits"
    turn_timeout_s: float = 3600.0
    harness: str = "cc"
    # 网络隔离：宿主网关 + 沙箱内 relay。None 表示未启用（退化为 tools_only）。
    gateway: UdsGateway | None = None
    sandbox_port: int = DEFAULT_SANDBOX_PORT
    events: list[MappedAgentEvent] = field(default_factory=list)
    checkpoint_timeout_s: float = 3600.0
    _capabilities: dict[str, object] = field(default_factory=dict, repr=False)
    _pending_system_prompt: str | None = field(default=None, repr=False)
    _turn_index: int = field(default=0, repr=False)
    _relay_launcher: Path | None = field(default=None, repr=False)
    _network_isolation: str = field(default="unknown", repr=False)

    # -- 生命周期 ---------------------------------------------------------

    def _ensure_ready(self) -> None:
        if not self._capabilities:
            self._capabilities = probe_claude_installation(self.binary)
        self.agent_home.mkdir(parents=True, exist_ok=True)
        if self.gateway is not None and self._relay_launcher is None:
            self._relay_launcher = install_relay_launcher(self.agent_home)

    def start(self, run_context: RunContext) -> AgentSession:
        run_context.validate_isolation()
        self._ensure_ready()
        # Claude Code 的会话 id 由 --session-id 指定（支持时），否则从首轮结果里回读。
        thread_id = str(uuid.uuid4())
        self._pending_system_prompt = self._system_prompt(run_context)
        self._record(
            "AgentSessionStarted",
            {
                "thread_id": thread_id,
                "harness": self.harness,
                "model": self.model,
                "claude_version": self._capabilities.get("version"),
                "network_isolation": (
                    "netns+uds_gateway" if self.gateway is not None else "tools_only"
                ),
                "bridge": self.base_url,
                "gateway": None if self.gateway is None else self.gateway.describe(),
            },
        )
        return AgentSession(thread_id=thread_id, goal_status="active", ephemeral=False)

    def resume(self, session_id: str, run_context: RunContext) -> AgentSession:
        run_context.validate_isolation()
        self._ensure_ready()
        self._pending_system_prompt = self._system_prompt(run_context)
        return AgentSession(thread_id=session_id, goal_status="active", ephemeral=False)

    def pause(self, session: AgentSession) -> AgentSession:
        session.goal_status = "paused"
        return session

    def close(self) -> None:
        return None

    # -- 一轮对话 ---------------------------------------------------------

    def run_until_checkpoint(
        self,
        session: AgentSession,
        run_context: RunContext,
        checkpoint_predicate: CheckpointPredicate,
    ) -> AgentSession:
        """跑完一个 headless 轮次（``claude -p`` 天然在一轮结束后返回）。"""

        self._ensure_ready()
        prompt = run_context.initial_prompt.strip()
        if not prompt:
            # 与 codex 一致：空 prompt 的调用只用于建立/恢复上下文，不消耗模型额度。
            return session
        command = self._command(session, run_context)
        isolated, sandbox_base_url = self._wrap(command, run_context)
        environment = self._environment(base_url=sandbox_base_url)
        self._turn_index += 1
        try:
            completed = subprocess.run(
                isolated,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.turn_timeout_s,
                cwd=str(run_context.cwd),
                env=environment,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            self._record(
                "AgentTurnFailed",
                {"reason": "timeout", "timeout_s": self.turn_timeout_s, "turn": self._turn_index},
            )
            raise ClaudeCodeUnavailable(
                f"claude headless turn exceeded {self.turn_timeout_s:.0f}s"
            ) from error
        session_id = self._consume_stream(completed.stdout or "", session.thread_id)
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-800:]
            self._record(
                "AgentTurnFailed",
                {"reason": "nonzero_exit", "exit_code": completed.returncode, "detail": tail},
            )
            raise ClaudeCodeUnavailable(f"claude exited {completed.returncode}: {tail}")
        if session_id and session_id != session.thread_id:
            # 该 claude 版本不支持 --session-id：采用它自己生成的会话 id。
            session.thread_id = session_id
        self._record(
            "AgentTurnCompleted",
            {"turn": self._turn_index, "thread_id": session.thread_id},
        )
        for event in reversed(self.events):
            if checkpoint_predicate(event):
                break
        return session

    # -- 命令与环境 -------------------------------------------------------

    def _system_prompt(self, run_context: RunContext) -> str:
        return "\n\n".join(
            part
            for part in (
                f"研究目标：{run_context.objective}",
                run_context.base_instructions,
                run_context.developer_instructions,
            )
            if part
        )

    def _command(self, session: AgentSession, run_context: RunContext) -> tuple[str, ...]:
        arguments: list[str] = [
            self.binary,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
        ]
        if self._capabilities.get("supports_permission_mode"):
            arguments.extend(("--permission-mode", self.permission_mode))
        arguments.extend(("--disallowedTools", ",".join(DISALLOWED_TOOLS)))
        if self._capabilities.get("supports_add_dir"):
            for root in run_context.runtime_workspace_roots:
                if root != run_context.cwd:
                    arguments.extend(("--add-dir", str(root)))
        if self._pending_system_prompt and self._capabilities.get(
            "supports_append_system_prompt"
        ):
            arguments.extend(("--append-system-prompt", self._pending_system_prompt))
        if self._turn_index == 0:
            if self._capabilities.get("supports_session_id"):
                arguments.extend(("--session-id", session.thread_id))
        else:
            arguments.extend(("--resume", session.thread_id))
        return tuple(arguments)

    def _environment(self, *, base_url: str) -> dict[str, str]:
        allowed = ("PATH", "LANG", "LC_ALL", "TZ", "TERM", "HOME", "USER")
        environment = {key: os.environ[key] for key in allowed if key in os.environ}
        environment.update(
            {
                # 桥接代理地址：Claude Code 只会说 Anthropic 协议。
                # 启用网关时这里是**沙箱内**的回环端口（宿主端口在 netns 里不可达）。
                "ANTHROPIC_BASE_URL": base_url,
                "ANTHROPIC_AUTH_TOKEN": self.api_key,
                "ANTHROPIC_API_KEY": self.api_key,
                "CLAUDE_CONFIG_DIR": str(self.agent_home),
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
                "DISABLE_TELEMETRY": "1",
                "DISABLE_ERROR_REPORTING": "1",
                "DISABLE_AUTOUPDATER": "1",
                "ANTHROPIC_MODEL": self.model,
            }
        )
        return environment

    def _wrap(
        self, command: tuple[str, ...], run_context: RunContext
    ) -> tuple[tuple[str, ...], str]:
        """套上隔离；返回（最终命令, cc 应使用的 base_url）。

        文件侧：GamePack 只读、workspace/research 可写、人类源码与凭据被遮蔽。
        网络侧：``allow_network=False``（命名空间级断网），仅放通网关 unix socket；
        沙箱内先起 relay，再跑 claude，claude 只看到本地回环端口。
        """

        use_gateway = self.gateway is not None and self._relay_launcher is not None
        socket_path = None if self.gateway is None else self.gateway.socket_path
        request = IsolationRequest(
            denied_read_roots=(run_context.human_pool_root, run_context.evaluator_root),
            readable_roots=(run_context.gamepack_root, self.agent_home),
            writable_roots=(*run_context.writable_workspace_roots, self.agent_home),
            allow_network=not use_gateway,
            allowed_unix_sockets=() if socket_path is None else (socket_path,),
        )
        try:
            isolation = select_candidate_isolation(
                request,
                backend=self.isolation_backend,
                profile_path=self.agent_home / "cc-isolation.sb",
            )
        except Exception as error:  # noqa: BLE001 - 隔离不可用时如实记录并继续
            self._network_isolation = "tools_only"
            self._record(
                "AgentIsolationUnavailable",
                {"harness": self.harness, "detail": f"{type(error).__name__}: {error}"},
            )
            # 没有沙箱就没有 netns，也就没有沙箱内 relay：必须直连宿主桥接代理。
            return command, self.base_url
        if not use_gateway:
            self._network_isolation = "tools_only"
            self._record(
                "AgentIsolationApplied",
                {**dict(isolation.describe()), "network_isolation": "tools_only"},
            )
            return isolation.wrap(command), self.base_url
        assert socket_path is not None and self._relay_launcher is not None
        inner = relay_command(
            self._relay_launcher,
            socket_path,
            self.sandbox_port,
            command,
            python_executable=sys.executable,
        )
        self._network_isolation = "netns+uds_gateway"
        self._record(
            "AgentIsolationApplied",
            {
                **dict(isolation.describe()),
                "network_isolation": "netns+uds_gateway",
                "sandbox_port": self.sandbox_port,
            },
        )
        return isolation.wrap(inner), f"http://127.0.0.1:{self.sandbox_port}"

    # -- 事件流 -----------------------------------------------------------

    def _record(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append(MappedAgentEvent(event_type=event_type, payload=payload))

    def _consume_stream(self, stdout: str, thread_id: str) -> str | None:
        """解析 stream-json 输出，映射成与 codex 同构的事件。"""

        session_id: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(event.get("type") or "")
            session_id = event.get("session_id") or session_id
            if kind == "assistant":
                message = event.get("message") or {}
                usage = message.get("usage") or {}
                total = int(usage.get("input_tokens") or 0) + int(
                    usage.get("output_tokens") or 0
                )
                self._record(
                    "AgentMessage",
                    {
                        "thread_id": session_id or thread_id,
                        "total_tokens": total or None,
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "stop_reason": message.get("stop_reason"),
                    },
                )
            elif kind == "result":
                usage = event.get("usage") or {}
                total = int(usage.get("input_tokens") or 0) + int(
                    usage.get("output_tokens") or 0
                )
                self._record(
                    "AgentTurnResult",
                    {
                        "thread_id": session_id or thread_id,
                        "subtype": event.get("subtype"),
                        "is_error": bool(event.get("is_error")),
                        "num_turns": event.get("num_turns"),
                        "duration_ms": event.get("duration_ms"),
                        "total_tokens": total or None,
                        "cost_usd": event.get("total_cost_usd"),
                    },
                )
            elif kind == "user":
                self._record("AgentToolResult", {"thread_id": session_id or thread_id})
        return session_id


def claude_code_runtime(
    config: ExperimentConfig,
    api_key: str,
    *,
    agent_home: Path,
) -> ClaudeCodeRuntime:
    """按冻结配置装配 cc harness（含本地 Anthropic 桥接代理 + UDS 网关）。

    桥接代理与网关都与 run 同生命周期：由 :class:`ClaudeCodeRuntime` 的调用方
    （CLI/服务端）在结束时调用 ``close()``。
    """

    from agentbench_hl.adapters.cc_goal.anthropic_bridge import AnthropicBridge  # noqa: PLC0415

    agent_home.mkdir(parents=True, exist_ok=True)
    bridge = AnthropicBridge(
        upstream_base=config.provider.base_url,
        api_key=api_key,
        log_path=agent_home / "bridge.log",
    )
    bridge.__enter__()
    gateway: UdsGateway | None = None
    if config.isolation.backend != "disabled":
        # 网关只在"有沙箱"的前提下有意义：沙箱负责断网，网关负责唯一放通。
        gateway = UdsGateway(
            socket_path=agent_home / "bridge.sock",
            target_host="127.0.0.1",
            target_port=bridge.port,
        ).start()
    runtime = ClaudeCodeRuntime(
        binary=config.runtime.harness_binary,
        agent_home=agent_home,
        base_url=bridge.base_url,
        api_key=api_key,
        model=config.provider.model,
        isolation_backend=config.isolation.backend,
        gateway=gateway,
    )
    # 让调用方 close() 时把代理与网关一起收掉。
    original_close = runtime.close

    def close() -> None:
        original_close()
        if gateway is not None:
            gateway.close()
        bridge.close()

    runtime.close = close  # type: ignore[method-assign]
    return runtime

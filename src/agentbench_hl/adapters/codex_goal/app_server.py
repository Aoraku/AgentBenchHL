"""Persistent Codex Goal runtime with a dedicated, memory-free CODEX_HOME."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from agentbench_hl.adapters.codex_goal.event_mapper import (
    MappedAgentEvent,
    map_app_server_event,
)
from agentbench_hl.adapters.codex_goal.protocol import JsonRpcStdioClient
from agentbench_hl.adapters.codex_goal.responses_proxy import ResponsesCompatProxy
from agentbench_hl.ports.agent_runtime import AgentSession, RunContext


def write_codex_config(
    codex_home: str | Path,
    *,
    base_url: str,
    model: str,
    reasoning_effort: str,
    readable_roots: Sequence[str | Path] = (),
    writable_roots: Sequence[str | Path] = (),
    tool_roots: Sequence[str | Path] = (),
    denied_roots: Sequence[str | Path] = (),
) -> Path:
    root = Path(codex_home)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.toml"
    quote = json.dumps
    readable = {Path(item).resolve() for item in (*readable_roots, *tool_roots)}
    writable = {Path(item).resolve() for item in writable_roots}
    denied = {Path(item).resolve() for item in denied_roots}
    if not writable.issubset(readable):
        raise ValueError("writable permission roots must also be readable")
    scratch_root = None
    if writable:
        scratch_root = sorted(writable)[0] / ".agentbench/runtime-tmp"
        scratch_root.mkdir(parents=True, exist_ok=True)
    permission_lines = [
        "[permissions.agentbench-hl]",
        'description = "Isolated AgentBench HL candidate research workspace"',
        "",
        "[permissions.agentbench-hl.filesystem]",
        '":root" = "deny"',
        '":minimal" = "read"',
        '":tmpdir" = "write"',
        '":slash_tmp" = "write"',
    ]
    permission_lines.extend(
        f"{quote(str(item))} = {quote('write' if item in writable else 'read')}"
        for item in sorted(readable)
    )
    permission_lines.extend(f'{quote(str(item))} = "deny"' for item in sorted(denied))
    permission_lines.extend(
        (
            "",
            "[permissions.agentbench-hl.network]",
            "enabled = false",
            "",
        )
    )
    text = "\n".join(
        (
            'model_provider = "OpenAI"',
            f"model = {quote(model)}",
            f"review_model = {quote(model)}",
            f"model_reasoning_effort = {quote(reasoning_effort)}",
            'default_permissions = "agentbench-hl"',
            "",
            "[model_providers.OpenAI]",
            'name = "OpenAI"',
            f"base_url = {quote(base_url)}",
            'wire_api = "responses"',
            'env_key = "OPENAI_API_KEY"',
            "requires_openai_auth = true",
            "",
            "[features]",
            "goals = true",
            "",
            "[analytics]",
            "enabled = false",
            "",
            "[shell_environment_policy]",
            'inherit = "core"',
            "ignore_default_excludes = false",
            *(
                (f"set = {{ TMPDIR = {quote(str(scratch_root))} }}",)
                if scratch_root is not None
                else ()
            ),
            'exclude = ["OPENAI_API_KEY", "ABHL_API_KEY", "AGENTBENCH_ROOT"]',
            "",
            *permission_lines,
        )
    )
    path.write_text(text, encoding="utf-8")
    return path


def _mapped_turn_id(event: MappedAgentEvent) -> str | None:
    direct = event.payload.get("turn_id")
    if isinstance(direct, str):
        return direct
    params = event.payload.get("params")
    if not isinstance(params, Mapping):
        return None
    direct = params.get("turnId")
    if isinstance(direct, str):
        return direct
    turn = params.get("turn")
    if isinstance(turn, Mapping) and isinstance(turn.get("id"), str):
        return str(turn["id"])
    return None


class CodexGoalRuntime:
    def __init__(
        self,
        *,
        command: Sequence[str],
        codex_home: str | Path,
        base_url: str,
        model: str,
        reasoning_effort: str,
        api_key: str,
        use_responses_proxy: bool = False,
        request_timeout_s: float = 30.0,
        # A stalled provider turn must not hold the research loop for fifteen
        # minutes.  The caller can override this for unusually long tasks,
        # but the default keeps resumable experiments responsive.
        checkpoint_timeout_s: float = 300.0,
    ) -> None:
        self.command = tuple(command)
        self.codex_home = Path(codex_home).resolve()
        self.base_url = base_url
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._api_key = api_key
        self.use_responses_proxy = use_responses_proxy
        self.request_timeout_s = request_timeout_s
        self.checkpoint_timeout_s = checkpoint_timeout_s
        self.client: JsonRpcStdioClient | None = None
        self.responses_proxy: ResponsesCompatProxy | None = None
        self.events: list[MappedAgentEvent] = []
        self._turn_telemetry: list[dict[str, object]] = []

    def set_turn_reasoning_effort(self, effort: str) -> None:
        if not effort.strip():
            raise ValueError("turn reasoning effort cannot be empty")
        self.reasoning_effort = effort.strip()

    def _environment(self) -> dict[str, str]:
        allowed = (
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TMPDIR",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "SYSTEMROOT",
        )
        environment = {name: os.environ[name] for name in allowed if name in os.environ}
        isolated_home = self.codex_home / "home"
        isolated_home.mkdir(parents=True, exist_ok=True)
        environment.update(
            {
                "HOME": str(isolated_home),
                "CODEX_HOME": str(self.codex_home),
                "OPENAI_API_KEY": self._api_key,
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        return environment

    def _ensure_client(self, run_context: RunContext) -> JsonRpcStdioClient:
        if self.client is not None:
            return self.client
        tool_roots = {
            Path("/usr"),
            Path("/bin"),
            Path("/System"),
            Path("/Library"),
            Path("/opt/homebrew"),
            Path(sys.prefix),
            Path(sys.base_prefix),
            Path(__file__).resolve().parents[4],
        }
        tool_roots.update(
            Path(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item
        )
        provider_base_url = self.base_url
        if self.use_responses_proxy:
            self.responses_proxy = ResponsesCompatProxy(
                self.base_url, timeout_s=self.checkpoint_timeout_s
            )
            self.responses_proxy.start()
            provider_base_url = self.responses_proxy.base_url
        write_codex_config(
            self.codex_home,
            base_url=provider_base_url,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            readable_roots=run_context.runtime_workspace_roots,
            writable_roots=run_context.writable_workspace_roots,
            tool_roots=tuple(item for item in tool_roots if item.exists()),
            denied_roots=(
                run_context.human_pool_root,
                run_context.evaluator_root,
            ),
        )
        self.client = JsonRpcStdioClient(
            self.command,
            cwd=run_context.cwd,
            environment=self._environment(),
            stderr_path=self.codex_home / "logs/app-server.stderr.log",
            secrets=(self._api_key,),
            request_timeout_s=self.request_timeout_s,
        )
        self.client.request(
            "initialize",
            {
                "clientInfo": {
                    "name": "agentbench-hl",
                    "title": "AgentBench HL Goal Runtime",
                    "version": "0.1.0",
                },
                "capabilities": {"experimentalApi": True},
            },
        )
        self.client.notify("initialized")
        return self.client

    @staticmethod
    def _thread_id(result: object) -> str:
        if not isinstance(result, dict) or not isinstance(result.get("thread"), dict):
            raise ValueError("Codex thread response has no thread")
        thread_id = result["thread"].get("id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("Codex thread response has no thread ID")
        return thread_id

    def start(self, run_context: RunContext) -> AgentSession:
        run_context.validate_isolation()
        client = self._ensure_client(run_context)
        result = client.request(
            "thread/start",
            {
                "model": run_context.model,
                "modelProvider": run_context.model_provider,
                "cwd": str(run_context.cwd),
                "baseInstructions": run_context.base_instructions,
                "developerInstructions": run_context.developer_instructions,
                "runtimeWorkspaceRoots": [
                    str(item) for item in run_context.runtime_workspace_roots
                ],
                "permissions": "agentbench-hl",
                "approvalPolicy": "never",
                "approvalsReviewer": "auto_review",
                "ephemeral": False,
                "historyMode": "paginated",
                "environments": [],
                "serviceName": "agentbench-hl",
            },
        )
        thread_id = self._thread_id(result)
        goal_result = client.request(
            "thread/goal/set",
            {
                "threadId": thread_id,
                "objective": run_context.objective,
                "status": "paused",
            },
        )
        client.request(
            "thread/memoryMode/set",
            {"threadId": thread_id, "mode": "disabled"},
        )
        goal = goal_result.get("goal", {})
        status = goal.get("status", "active") if isinstance(goal, dict) else "active"
        return AgentSession(thread_id, str(status), ephemeral=False)

    def resume(self, session_id: str, run_context: RunContext) -> AgentSession:
        run_context.validate_isolation()
        client = self._ensure_client(run_context)
        client.request(
            "thread/goal/set",
            {"threadId": session_id, "status": "paused"},
        )
        result = client.request(
            "thread/resume",
            {
                "threadId": session_id,
                "model": run_context.model,
                "modelProvider": run_context.model_provider,
                "cwd": str(run_context.cwd),
                "baseInstructions": run_context.base_instructions,
                "developerInstructions": run_context.developer_instructions,
                "runtimeWorkspaceRoots": [
                    str(item) for item in run_context.runtime_workspace_roots
                ],
                "approvalPolicy": "never",
                "approvalsReviewer": "auto_review",
                "excludeTurns": True,
                "permissions": "agentbench-hl",
            },
        )
        thread_id = self._thread_id(result)
        client.request(
            "thread/memoryMode/set",
            {"threadId": thread_id, "mode": "disabled"},
        )
        goal_result = client.request("thread/goal/get", {"threadId": thread_id})
        goal = goal_result.get("goal")
        status = goal.get("status", "paused") if isinstance(goal, dict) else "paused"
        if status != "paused":
            goal_result = client.request(
                "thread/goal/set", {"threadId": thread_id, "status": "paused"}
            )
            goal = goal_result.get("goal")
            status = goal.get("status", "paused") if isinstance(goal, dict) else "paused"
        return AgentSession(thread_id, str(status), ephemeral=False)

    def run_until_checkpoint(
        self,
        session: AgentSession,
        run_context: RunContext,
        checkpoint_predicate: Callable[[MappedAgentEvent], bool],
    ) -> AgentSession:
        client = self._ensure_client(run_context)
        event_offset = len(self.events)
        started_at = time.monotonic()
        turn = client.request(
            "turn/start",
            {
                "threadId": session.thread_id,
                "input": [{"type": "text", "text": run_context.initial_prompt}],
                "cwd": str(run_context.cwd),
                "runtimeWorkspaceRoots": [
                    str(item) for item in run_context.runtime_workspace_roots
                ],
                "effort": self.reasoning_effort,
                "permissions": "agentbench-hl",
            },
        )
        turn_value = turn.get("turn", {})
        if isinstance(turn_value, dict) and isinstance(turn_value.get("id"), str):
            session.active_turn_id = turn_value["id"]
        deadline = time.monotonic() + self.checkpoint_timeout_s
        runtime_error: str | None = None
        turn_started = False
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Codex Goal did not reach a checkpoint in time")
            try:
                notification = client.next_notification(min(remaining, 60.0))
            except TimeoutError:
                continue
            mapped = map_app_server_event(notification)
            self.events.append(mapped)
            mapped_turn_id = _mapped_turn_id(mapped)
            is_active_turn = mapped_turn_id == session.active_turn_id
            if mapped.event_type == "AgentTurnStarted" and is_active_turn:
                turn_started = True
            if mapped.event_type == "AgentRuntimeError":
                params = mapped.payload.get("params", {})
                if isinstance(params, dict) and isinstance(params.get("message"), str):
                    runtime_error = params["message"]
            if mapped.event_type == "AgentGoalUpdated" and turn_started:
                params = mapped.payload.get("params", {})
                if isinstance(params, dict):
                    goal = params.get("goal", {})
                    if isinstance(goal, dict) and isinstance(goal.get("status"), str):
                        session.goal_status = goal["status"]
            if mapped.event_type == "AgentTurnCompleted" and is_active_turn:
                params = mapped.payload.get("params", {})
                turn = params.get("turn", {}) if isinstance(params, Mapping) else {}
                status = turn.get("status") if isinstance(turn, Mapping) else None
                if isinstance(status, str) and status != "completed":
                    error = turn.get("error") if isinstance(turn, Mapping) else None
                    message = error.get("message") if isinstance(error, Mapping) else None
                    self._finalize_turn_telemetry(event_offset, started_at, session.active_turn_id)
                    detail = f": {message}" if isinstance(message, str) else ""
                    raise RuntimeError(f"Codex turn ended with status {status}{detail}")
            if is_active_turn and checkpoint_predicate(mapped):
                self._finalize_turn_telemetry(event_offset, started_at, session.active_turn_id)
                return session
            if turn_started and session.goal_status == "complete":
                self._finalize_turn_telemetry(event_offset, started_at, session.active_turn_id)
                return session
            if turn_started and session.goal_status in {
                "blocked",
                "usageLimited",
                "budgetLimited",
            }:
                self._finalize_turn_telemetry(event_offset, started_at, session.active_turn_id)
                detail = f": {runtime_error}" if runtime_error else ""
                raise RuntimeError(f"Codex Goal stopped with status {session.goal_status}{detail}")

    def _finalize_turn_telemetry(
        self,
        event_offset: int,
        started_at: float,
        turn_id: str | None,
    ) -> None:
        usage = next(
            (
                event.payload
                for event in reversed(self.events[event_offset:])
                if event.event_type == "AgentUsageObserved"
                and _mapped_turn_id(event) in {None, turn_id}
            ),
            {},
        )
        self._turn_telemetry.append(
            {
                "input_tokens": usage.get("input_tokens"),
                "cached_input_tokens": usage.get("cached_input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "wall_time_s": time.monotonic() - started_at,
            }
        )

    def consume_turn_telemetry(self) -> tuple[dict[str, object], ...]:
        values = tuple(self._turn_telemetry)
        self._turn_telemetry.clear()
        return values

    def pause(self, session: AgentSession) -> AgentSession:
        if self.client is None:
            raise RuntimeError("Codex Goal runtime is not connected")
        result = self.client.request(
            "thread/goal/set",
            {"threadId": session.thread_id, "status": "paused"},
        )
        goal = result.get("goal", {})
        session.goal_status = (
            str(goal.get("status", "paused")) if isinstance(goal, dict) else "paused"
        )
        return session

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.responses_proxy is not None:
            self.responses_proxy.close()
            self.responses_proxy = None

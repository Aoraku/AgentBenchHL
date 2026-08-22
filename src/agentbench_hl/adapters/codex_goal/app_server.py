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

#: codex 自带 OS 沙箱的可选口径（``codex exec -s`` 的取值）。
AGENT_SANDBOX_MODES = ("read-only", "workspace-write", "danger-full-access")

#: 厂商官方 model catalog（``~/.codex/models.json`` 的内容），逐字照录厂商文档。
#:
#: 为什么必须用它，而不是自己写 ``model_context_window`` 两个键：
#: 中转模型不在 codex 自带目录里，codex 会报 "Unknown model … will use fallback
#: model metadata" 并用**兜底**的模型元数据。兜底值里包含压缩相关的窗口，
#: 于是压缩会在一个我们没设过、也看不到的点上触发——实测 antwar2 在 97k 上下文
#: 触发压缩并死掉，而我们当时在 config.toml 里写的是 200000。
#: 给出完整 catalog 后，context_window / effective_context_window_percent 等
#: 全部由这份声明决定，压缩线才变成一个我们能控制的数。
#:
#: 来源：https://docs.bigmodel.cn/cn/coding-plan/tool/codex
ZHIPU_MODEL_CATALOG: tuple[dict[str, object], ...] = (
    {
        "slug": "glm-5.3",
        "display_name": "glm-5.3",
        "description": "Z.ai's latest flagship model",
        "default_reasoning_level": "max",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Light reasoning"},
            {"effort": "high", "description": "Enhanced reasoning"},
            {"effort": "max", "description": "Deep reasoning"},
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 0,
        "base_instructions": "",
        "supports_reasoning_summaries": True,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "context_window": 1048576,
        "max_context_window": 1048576,
        "effective_context_window_percent": 95,
        "supports_parallel_tool_calls": True,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
    },
    {
        "slug": "glm-5-turbo",
        "display_name": "glm-5-turbo",
        "description": "Agent-optimized model",
        "default_reasoning_level": "max",
        "supported_reasoning_levels": [],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,
        "base_instructions": "",
        "supports_reasoning_summaries": True,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "context_window": 204800,
        "max_context_window": 204800,
        "effective_context_window_percent": 95,
        "supports_parallel_tool_calls": True,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
    },
    # glm-5.2 / glm-5：中转站上同样在售，实测 /responses 与 /chat/completions
    # 都返回 200。必须一并声明——只要 catalog 里没有，codex 就会打印
    # "Unknown model … will use fallback model metadata" 并改用兜底元数据，
    # 于是压缩会在一个我们没设过、也看不到的上下文点上触发。
    # 上下文窗口按官方文档给 200k（低于 5.3 的 1M）。
    {
        "slug": "glm-5.2",
        "display_name": "glm-5.2",
        "description": "Z.ai GLM-5.2",
        "default_reasoning_level": "max",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Light reasoning"},
            {"effort": "high", "description": "Enhanced reasoning"},
            {"effort": "max", "description": "Deep reasoning"},
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 2,
        "base_instructions": "",
        "supports_reasoning_summaries": True,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "context_window": 204800,
        "max_context_window": 204800,
        "effective_context_window_percent": 95,
        "supports_parallel_tool_calls": True,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
    },
    {
        "slug": "glm-5",
        "display_name": "glm-5",
        "description": "Z.ai GLM-5",
        "default_reasoning_level": "max",
        "supported_reasoning_levels": [
            {"effort": "low", "description": "Light reasoning"},
            {"effort": "high", "description": "Enhanced reasoning"},
            {"effort": "max", "description": "Deep reasoning"},
        ],
        "shell_type": "shell_command",
        "visibility": "list",
        "supported_in_api": True,
        "priority": 3,
        "base_instructions": "",
        "supports_reasoning_summaries": True,
        "default_reasoning_summary": "none",
        "support_verbosity": False,
        "apply_patch_tool_type": "freeform",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "context_window": 204800,
        "max_context_window": 204800,
        "effective_context_window_percent": 95,
        "supports_parallel_tool_calls": True,
        "experimental_supported_tools": [],
        "input_modalities": ["text"],
    },
)

MODEL_CATALOGS: dict[str, tuple[dict[str, object], ...]] = {"zhipu": ZHIPU_MODEL_CATALOG}


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
    sandbox_mode: str = "danger-full-access",
    context_window: int | None = None,
    auto_compact_token_limit: int | None = None,
    model_catalog: str | None = None,
) -> Path:
    """写 codex 的 ``config.toml``。

    ``sandbox_mode`` 决定**codex 自带的 OS 级沙箱**（不是候选对局的隔离）：

    * ``danger-full-access``（默认）—— 关掉 codex 自己的沙箱。实测在本项目的服务器上，
      codex 0.147 的 ``linux_sandbox`` 连自带的 ``workspace-write`` 预设都会对每次
      ``exec_command`` 报 ``permission_denied``（连 PATH 里的 venv 都读不到），agent
      因此一个文件都写不了、永远交不出 ``action.json``。**候选代码的隔离由我们自己的
      bwrap 负责**（``adapters/isolation``），那才是科学上必须成立的一层：它把人类选手池
      与评测器 tmpfs 掉。这里关掉的只是 agent 自己那层 OS 沙箱。
    * ``read-only`` / ``workspace-write`` —— 交给 codex 管，附带下面的命名权限 profile。

    **注意**：``danger-full-access`` 下 ``denied_roots`` 不再有强制力（deny 是靠沙箱执行的），
    "agent 不许看人类选手代码"这条改由事后审计兜（``abhl run audit`` 的
    ``reference_policy_leaks``）。这是"先跑通"的显式取舍，不是忘了。
    """

    if sandbox_mode not in AGENT_SANDBOX_MODES:
        raise ValueError(f"sandbox_mode must be one of {AGENT_SANDBOX_MODES}: {sandbox_mode!r}")
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
    sandboxed = sandbox_mode != "danger-full-access"
    # 厂商官方 model catalog：给了就写 models.json，并让 codex 从那里读模型能力。
    # 此时**不再**写 model_context_window——两处都写只会让"压缩线到底是多少"
    # 变成一个要靠读 codex 源码才能回答的问题。
    catalog_path: Path | None = None
    if model_catalog is not None:
        if model_catalog not in MODEL_CATALOGS:
            raise ValueError(f"unknown model_catalog: {model_catalog!r}")
        catalog_path = root / "models.json"
        catalog_path.write_text(
            json.dumps({"models": list(MODEL_CATALOGS[model_catalog])}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
    permission_lines: list[str] = []
    if sandboxed:
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
            f"sandbox_mode = {quote(sandbox_mode)}",
            # 模型能力优先走厂商官方 catalog（见 MODEL_CATALOGS 的详注）；
            # 没有 catalog 时才退回手工声明这两个键，否则 codex 会用兜底元数据，
            # 压缩会在一个我们没设过的点上触发。
            *(
                (f"model_catalog_json = {quote(str(catalog_path))}",)
                if catalog_path is not None
                else ()
            ),
            *(
                (f"model_context_window = {int(context_window)}",)
                if context_window is not None and catalog_path is None
                else ()
            ),
            *(
                (f"model_auto_compact_token_limit = {int(auto_compact_token_limit)}",)
                if auto_compact_token_limit is not None
                else ()
            ),
            *(('default_permissions = "agentbench-hl"',) if sandboxed else ()),
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
        # codex 自带 OS 沙箱。默认关掉：实测它在服务器上会拒掉每一次 exec_command，
        # agent 因此永远交不出 action.json。候选代码的隔离由我们自己的 bwrap 负责。
        sandbox_mode: str = "danger-full-access",
        context_window: int | None = None,
        auto_compact_token_limit: int | None = None,
        model_catalog: str | None = None,
        #: JSON-RPC ``initialize`` 里报的客户端名字。
        #:
        #: 它会变成上游看到的 ``originator`` 请求头。默认报
        #: ``agentbench-hl``（诚实署名，便于我们自己在日志里区分流量），
        #: 但**有些中转站按客户端白名单放行**：实测 sbtunnel 对
        #: ``originator: agentbench-hl`` 返回
        #: ``403 This account only allows Codex official clients``，
        #: 而同一个 key、同一个端点、同一份 config.toml 换成 codex 官方
        #: originator 就 200。
        #:
        #: 所以这个值必须可配置（模型档案里的 ``client_name``），
        #: 否则那类中转站整个不可用 —— 而失败信息只会说 403，
        #: 完全指不回"是署名被拒"。
        client_name: str = "agentbench-hl",
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
        if sandbox_mode not in AGENT_SANDBOX_MODES:
            raise ValueError(f"sandbox_mode must be one of {AGENT_SANDBOX_MODES}")
        self.sandbox_mode = sandbox_mode
        self.context_window = context_window
        self.auto_compact_token_limit = auto_compact_token_limit
        self.model_catalog = model_catalog
        self.client_name = client_name
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

    def _permission_fields(self) -> dict[str, str]:
        """线程级要不要引用命名权限 profile。

        关掉 codex 自带沙箱时**必须不发**：config.toml 里没有 ``[permissions]`` 表，
        codex 会直接拒掉 ``thread/start``（``default_permissions requires a
        [permissions] table``）。少了这个判断，"关沙箱"就变成"根本起不来"。
        """

        if self.sandbox_mode == "danger-full-access":
            return {}
        return {"permissions": "agentbench-hl"}

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
            sandbox_mode=self.sandbox_mode,
            context_window=self.context_window,
            auto_compact_token_limit=self.auto_compact_token_limit,
            model_catalog=self.model_catalog,
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
                    "name": self.client_name,
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
                **self._permission_fields(),
                "approvalPolicy": "never",
                "approvalsReviewer": "auto_review",
                "ephemeral": False,                "historyMode": "paginated",
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
                **self._permission_fields(),
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
                **self._permission_fields(),
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

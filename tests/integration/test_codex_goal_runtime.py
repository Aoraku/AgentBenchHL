from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agentbench_hl.adapters.codex_goal.app_server import (
    CodexGoalRuntime,
    write_codex_config,
)
from agentbench_hl.adapters.codex_goal.protocol import JsonRpcStdioClient
from agentbench_hl.adapters.codex_goal.read_isolation import (
    isolated_app_server_command,
    write_read_isolation_profile,
)
from agentbench_hl.ports.agent_runtime import RunContext

FAKE_SERVER = Path(__file__).parents[1] / "fakes/fake_codex_app_server.py"


def run_context(root: Path) -> RunContext:
    candidate = root / "candidate"
    gamepack = root / "gamepack"
    research = root / "research"
    human_pool = root / "human-pool"
    evaluator = root / "evaluator-config"
    for path in (candidate, gamepack, research, human_pool, evaluator):
        path.mkdir()
    return RunContext(
        objective="从冻结规则生成 v000，并持续击败全部可运行人类池",
        initial_prompt=("读取 Goal charter，生成 from-scratch v000 并执行首个 checkpoint。"),
        base_instructions="只依据公开 GamePack、当前候选和本 run 证据。",
        developer_instructions="不得读取人类源码；每轮更新正负 Experience。",
        cwd=candidate,
        candidate_root=candidate,
        gamepack_root=gamepack,
        research_root=research,
        human_pool_root=human_pool,
        evaluator_root=evaluator,
        runtime_workspace_roots=(candidate, gamepack, research),
        model="gpt-5.5",
        model_provider="OpenAI",
    )


def test_goal_runtime_starts_isolated_persistent_thread(tmp_path: Path) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log)),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/responses",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
    )
    try:
        session = runtime.start(context)
    finally:
        runtime.close()

    methods = log.read_text(encoding="utf-8").splitlines()
    assert methods[:3] == ["initialize", "thread/start", "thread/goal/set"]
    assert "thread/memoryMode/set" in methods
    assert session.thread_id == "thread-1"
    assert session.goal_status == "paused"
    assert session.ephemeral is False
    requests = [
        json.loads(line) for line in log.with_suffix(".requests.jsonl").read_text().splitlines()
    ]
    goal_set = next(item for item in requests if item["method"] == "thread/goal/set")
    assert goal_set["params"]["status"] == "paused"


def test_goal_runtime_routes_custom_provider_through_local_compat_proxy(
    tmp_path: Path,
) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log)),
        codex_home=tmp_path / "codex-home",
        base_url="https://provider.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
        use_responses_proxy=True,
    )
    try:
        runtime.start(context)
        config = (tmp_path / "codex-home/config.toml").read_text(encoding="utf-8")
        assert 'base_url = "http://127.0.0.1:' in config
        assert "https://provider.invalid" not in config
    finally:
        runtime.close()

    assert runtime.responses_proxy is None


def test_goal_runtime_finalizes_per_turn_usage_and_wall_time(tmp_path: Path) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log)),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
    )
    try:
        session = runtime.start(context)
        runtime.run_until_checkpoint(
            session,
            context,
            lambda event: event.event_type == "AgentTurnCompleted",
        )
        telemetry = runtime.consume_turn_telemetry()
    finally:
        runtime.close()

    assert len(telemetry) == 1
    assert telemetry[0]["input_tokens"] == 11
    assert telemetry[0]["cached_input_tokens"] == 4
    assert telemetry[0]["total_tokens"] == 17
    assert float(telemetry[0]["wall_time_s"]) > 0


def test_goal_runtime_allows_a_per_process_reasoning_effort_fallback(
    tmp_path: Path,
) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log)),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
    )
    try:
        session = runtime.start(context)
        runtime.set_turn_reasoning_effort("high")
        runtime.run_until_checkpoint(
            session,
            context,
            lambda event: event.event_type == "AgentTurnCompleted",
        )
    finally:
        runtime.close()

    requests = [
        json.loads(line) for line in log.with_suffix(".requests.jsonl").read_text().splitlines()
    ]
    turn_start = next(item for item in requests if item["method"] == "turn/start")
    assert turn_start["params"]["effort"] == "high"


def test_goal_runtime_treats_notification_poll_timeout_as_silence(
    tmp_path: Path,
) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log)),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
    )
    try:
        session = runtime.start(context)
        assert runtime.client is not None
        original = runtime.client.next_notification
        calls = 0

        def one_silent_poll(timeout_s: float):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("fixture poll silence")
            return original(timeout_s)

        runtime.client.next_notification = one_silent_poll  # type: ignore[method-assign]
        runtime.run_until_checkpoint(
            session,
            context,
            lambda event: event.event_type == "AgentTurnCompleted",
        )
    finally:
        runtime.close()

    assert calls >= 2


def test_goal_runtime_pauses_a_blocked_goal_before_explicit_resume(
    tmp_path: Path,
) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log), "blocked"),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
        sandbox_mode="workspace-write",
    )
    try:
        session = runtime.resume("thread-1", context)
    finally:
        runtime.close()

    methods = log.read_text(encoding="utf-8").splitlines()
    assert methods.count("thread/goal/set") == 2
    requests = [
        json.loads(line) for line in log.with_suffix(".requests.jsonl").read_text().splitlines()
    ]
    thread_resume = next(item for item in requests if item["method"] == "thread/resume")
    assert thread_resume["params"]["permissions"] == "agentbench-hl"
    assert "sandbox" not in thread_resume["params"]
    goal_sets = [item for item in requests if item["method"] == "thread/goal/set"]
    assert all(item["params"]["status"] == "paused" for item in goal_sets)
    assert session.goal_status == "paused"


def test_goal_runtime_surfaces_blocked_provider_turn_as_an_error(
    tmp_path: Path,
) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log), "blocked"),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
    )
    try:
        session = runtime.start(context)
        with pytest.raises(RuntimeError, match="API key is required"):
            runtime.run_until_checkpoint(
                session,
                context,
                lambda event: event.event_type == "AgentTurnCompleted",
            )
        telemetry = runtime.consume_turn_telemetry()
    finally:
        runtime.close()

    assert len(telemetry) == 1
    assert telemetry[0]["total_tokens"] is None


def test_goal_runtime_rejects_a_failed_active_turn(tmp_path: Path) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log), "failed_turn"),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
    )
    try:
        session = runtime.start(context)
        with pytest.raises(RuntimeError, match="stream disconnected"):
            runtime.run_until_checkpoint(
                session,
                context,
                lambda event: event.event_type == "AgentTurnCompleted",
            )
        telemetry = runtime.consume_turn_telemetry()
    finally:
        runtime.close()

    assert len(telemetry) == 1
    assert telemetry[0]["total_tokens"] is None


def test_resumed_turn_ignores_a_stale_blocked_notification(
    tmp_path: Path,
) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log), "stale_blocked"),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
    )
    try:
        session = runtime.resume("thread-1", context)
        runtime.run_until_checkpoint(
            session,
            context,
            lambda event: event.event_type == "AgentTurnCompleted",
        )
    finally:
        runtime.close()

    assert session.goal_status == "paused"


def test_resumed_turn_ignores_a_stale_completed_turn(tmp_path: Path) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log), "stale_complete"),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
    )
    try:
        session = runtime.resume("thread-1", context)
        runtime.run_until_checkpoint(
            session,
            context,
            lambda event: event.event_type == "AgentTurnCompleted",
        )
        telemetry = runtime.consume_turn_telemetry()
    finally:
        runtime.close()

    completed_ids = [
        event.payload["params"]["turn"]["id"]
        for event in runtime.events
        if event.event_type == "AgentTurnCompleted"
    ]
    assert completed_ids == ["old-turn", "turn-1"]
    assert len(telemetry) == 1
    assert telemetry[0]["total_tokens"] == 17


def test_runtime_roots_exclude_humans_certification_and_reference_policies(
    tmp_path: Path,
) -> None:
    context = run_context(tmp_path)

    context.validate_isolation()

    roots = context.runtime_workspace_roots
    assert context.candidate_root in roots
    assert context.gamepack_root in roots
    assert context.human_pool_root not in roots
    assert context.evaluator_root not in roots
    assert context.gamepack_root not in context.writable_workspace_roots
    assert context.candidate_root in context.writable_workspace_roots
    assert context.research_root in context.writable_workspace_roots
    assert all("handoff_next_agent" not in str(path) for path in roots)


def test_isolation_rejects_human_pool_root(tmp_path: Path) -> None:
    context = run_context(tmp_path)
    unsafe = context.with_runtime_workspace_roots(
        (*context.runtime_workspace_roots, context.human_pool_root)
    )

    with pytest.raises(ValueError, match="human"):
        unsafe.validate_isolation()


def test_generated_codex_config_contains_no_literal_key(tmp_path: Path) -> None:
    config = write_codex_config(
        tmp_path,
        base_url="https://example.invalid/responses",
        model="gpt-5.5",
        reasoning_effort="xhigh",
    )

    text = config.read_text(encoding="utf-8")
    assert "sk-" not in text
    assert "fixture-secret-value" not in text
    assert 'base_url = "https://example.invalid/responses"' in text
    assert 'env_key = "OPENAI_API_KEY"' in text
    assert "[shell_environment_policy]" in text
    assert 'inherit = "core"' in text
    assert '"OPENAI_API_KEY"' in text


def test_generated_codex_config_defines_a_restricted_named_permission_profile(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    gamepack = tmp_path / "gamepack"
    research = tmp_path / "research"
    tools = tmp_path / "tools"
    hidden = tmp_path / "human-pool"
    config = write_codex_config(
        tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        readable_roots=(candidate, gamepack, research),
        writable_roots=(candidate, research),
        tool_roots=(tools,),
        denied_roots=(hidden,),
        sandbox_mode="workspace-write",
    )

    text = config.read_text(encoding="utf-8")
    assert 'sandbox_mode = "workspace-write"' in text
    assert 'default_permissions = "agentbench-hl"' in text
    assert "[permissions.agentbench-hl.filesystem]" in text
    assert '":root" = "deny"' in text
    assert '":tmpdir" = "write"' in text
    assert '":slash_tmp" = "write"' in text
    assert f'{json.dumps(str(tools.resolve()))} = "read"' in text
    assert f'{json.dumps(str(hidden.resolve()))} = "deny"' in text
    assert f'{json.dumps(str(candidate.resolve()))} = "write"' in text
    assert f'{json.dumps(str(research.resolve()))} = "write"' in text
    assert f'{json.dumps(str(gamepack.resolve()))} = "read"' in text
    assert "[permissions.agentbench-hl.network]" in text
    assert "enabled = false" in text


def test_default_config_turns_off_the_harness_own_sandbox(tmp_path: Path) -> None:
    """默认关掉 codex 自带 OS 沙箱，并且**不留**指向不存在 profile 的悬空引用。

    为什么默认关：实测 codex 0.147 的 linux_sandbox 在服务器上会对每次 ``exec_command``
    报 ``permission_denied``（连 PATH 里的 venv 都读不到），agent 一个文件都写不了，
    永远交不出 ``action.json``。候选**对局**的隔离由我们自己的 bwrap 负责，与这一层无关。
    """

    config = write_codex_config(
        tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        readable_roots=(tmp_path / "candidate",),
        writable_roots=(tmp_path / "candidate",),
        denied_roots=(tmp_path / "human-pool",),
    )

    text = config.read_text(encoding="utf-8")
    assert 'sandbox_mode = "danger-full-access"' in text
    # 关沙箱时命名 profile 不生效，留着 default_permissions 只会让 codex 找不到 profile。
    assert "default_permissions" not in text
    assert "[permissions.agentbench-hl" not in text


def test_model_metadata_is_written_so_the_harness_stops_guessing(tmp_path: Path) -> None:
    """中转模型不在 codex 的模型目录里，不配元数据就会套兜底参数。

    codex 0.147 只认 ``model_context_window`` 与 ``model_auto_compact_token_limit``
    （``model_max_output_tokens`` / ``model_max_tokens`` 都会被 ``--strict-config`` 拒），
    所以只写这两个。后者决定对话历史涨到多少就自动压缩 —— 实测一个 2 轮的 run
    input token 从 5888 涨到 219571 都没触发压缩，长跑必须配上。
    """

    config = write_codex_config(
        tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="glm-5.2",
        reasoning_effort="xhigh",
        context_window=200_000,
        auto_compact_token_limit=160_000,
    )

    text = config.read_text(encoding="utf-8")
    assert "model_context_window = 200000" in text
    assert "model_auto_compact_token_limit = 160000" in text


def test_model_metadata_is_omitted_when_unknown(tmp_path: Path) -> None:
    """没配就**不写**，让 harness 用它自己的兜底值 —— 绝不替模型编一个窗口大小。"""

    config = write_codex_config(
        tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="glm-5.2",
        reasoning_effort="xhigh",
    )

    text = config.read_text(encoding="utf-8")
    assert "model_context_window" not in text
    assert "model_auto_compact_token_limit" not in text


def test_unknown_agent_sandbox_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sandbox_mode"):
        write_codex_config(
            tmp_path / "codex-home",
            base_url="https://example.invalid/v1",
            model="gpt-5.6",
            reasoning_effort="xhigh",
            sandbox_mode="please-be-safe",
        )


def test_goal_runtime_uses_named_permissions_instead_of_a_nested_sandbox(
    tmp_path: Path,
) -> None:
    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log)),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
        sandbox_mode="workspace-write",
    )
    try:
        session = runtime.start(context)
        runtime.run_until_checkpoint(
            session,
            context,
            lambda event: event.event_type == "AgentTurnCompleted",
        )
    finally:
        runtime.close()

    requests = [
        json.loads(line) for line in log.with_suffix(".requests.jsonl").read_text().splitlines()
    ]
    thread_start = next(item for item in requests if item["method"] == "thread/start")
    turn_start = next(item for item in requests if item["method"] == "turn/start")
    assert thread_start["params"]["permissions"] == "agentbench-hl"
    assert "sandbox" not in thread_start["params"]
    assert turn_start["params"]["permissions"] == "agentbench-hl"
    assert "sandboxPolicy" not in turn_start["params"]


def test_runtime_omits_named_permissions_when_the_sandbox_is_off(tmp_path: Path) -> None:
    """关沙箱时不能再引用命名 profile。

    config.toml 里没有 ``[permissions]`` 表，仍然发 ``permissions: agentbench-hl``
    会让 codex 直接拒掉 ``thread/start``（实测：``default_permissions requires a
    [permissions] table``），于是"关沙箱"变成"根本起不来"。
    """

    log = tmp_path / "methods.log"
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=(sys.executable, str(FAKE_SERVER), str(log)),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="fixture-secret-value",
    )
    try:
        session = runtime.start(context)
        runtime.run_until_checkpoint(
            session,
            context,
            lambda event: event.event_type == "AgentTurnCompleted",
        )
    finally:
        runtime.close()

    requests = [
        json.loads(line) for line in log.with_suffix(".requests.jsonl").read_text().splitlines()
    ]
    for method in ("thread/start", "turn/start"):
        params = next(item for item in requests if item["method"] == method)["params"]
        assert "permissions" not in params, method
    config = (tmp_path / "codex-home" / "config.toml").read_text(encoding="utf-8")
    assert "default_permissions" not in config


@pytest.mark.live
def test_installed_app_server_accepts_generated_strict_config(tmp_path: Path) -> None:
    if os.environ.get("ABHL_RUN_CODEX_PROTOCOL_TEST") != "1":
        pytest.skip("set ABHL_RUN_CODEX_PROTOCOL_TEST=1 to test installed Codex")
    codex_home = tmp_path / "codex-home"
    write_codex_config(
        codex_home,
        base_url="https://example.invalid/responses",
        model="gpt-5.5",
        reasoning_effort="xhigh",
    )
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(codex_home),
        "OPENAI_API_KEY": "unused-protocol-test",
    }
    client = JsonRpcStdioClient(
        ("codex", "app-server", "--listen", "stdio://", "--strict-config"),
        cwd=tmp_path,
        environment=environment,
        stderr_path=tmp_path / "stderr.log",
    )
    try:
        result = client.request(
            "initialize",
            {
                "clientInfo": {"name": "agentbench-hl-test", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        client.notify("initialized")
    finally:
        client.close()

    assert result["platformOs"] == "macos"


@pytest.mark.live
def test_installed_app_server_starts_isolated_goal_without_model_turn(
    tmp_path: Path,
) -> None:
    if os.environ.get("ABHL_RUN_CODEX_PROTOCOL_TEST") != "1":
        pytest.skip("set ABHL_RUN_CODEX_PROTOCOL_TEST=1 to test installed Codex")
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=("codex", "app-server", "--listen", "stdio://", "--strict-config"),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/responses",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        api_key="unused-protocol-test",
    )
    try:
        session = runtime.start(context)
    finally:
        runtime.close()

    assert session.thread_id
    assert session.goal_status == "active"
    assert session.ephemeral is False


@pytest.mark.live
def test_installed_named_permissions_allow_public_roots_and_deny_other_files(
    tmp_path: Path,
) -> None:
    if os.environ.get("ABHL_RUN_CODEX_PERMISSION_TEST") != "1":
        pytest.skip("set ABHL_RUN_CODEX_PERMISSION_TEST=1 for permissions probe")
    context = run_context(tmp_path)
    hidden = context.human_pool_root
    public_file = context.gamepack_root / "public.txt"
    hidden_file = hidden / "policy.txt"
    public_file.write_text("public-rules", encoding="utf-8")
    hidden_file.write_text("hidden-policy", encoding="utf-8")
    runtime = CodexGoalRuntime(
        command=("codex", "app-server", "--listen", "stdio://", "--strict-config"),
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.6",
        reasoning_effort="xhigh",
        api_key="unused-permission-test",
    )
    try:
        runtime.start(context)
        assert runtime.client is not None
        allowed = runtime.client.request(
            "command/exec",
            {
                "command": ["/bin/cat", str(public_file)],
                "cwd": str(context.candidate_root),
                "permissionProfile": "agentbench-hl",
            },
        )
        candidate_output = context.candidate_root / "permission-probe.txt"
        writable = runtime.client.request(
            "command/exec",
            {
                "command": ["/usr/bin/touch", str(candidate_output)],
                "cwd": str(context.candidate_root),
                "permissionProfile": "agentbench-hl",
            },
        )
        python = runtime.client.request(
            "command/exec",
            {
                "command": ["/usr/bin/python3", "-c", "print('python-ok')"],
                "cwd": str(context.candidate_root),
                "permissionProfile": "agentbench-hl",
            },
        )
        denied = runtime.client.request(
            "command/exec",
            {
                "command": ["/bin/cat", str(hidden_file)],
                "cwd": str(context.candidate_root),
                "permissionProfile": "agentbench-hl",
            },
        )
    finally:
        runtime.close()

    assert allowed["exitCode"] == 0
    assert allowed["stdout"] == "public-rules"
    assert writable["exitCode"] == 0
    assert candidate_output.is_file()
    assert python["exitCode"] == 0
    assert python["stdout"].strip() == "python-ok"
    assert denied["exitCode"] != 0
    assert "hidden-policy" not in denied["stdout"]


@pytest.mark.live
def test_installed_custom_provider_disables_response_storage(tmp_path: Path) -> None:
    if os.environ.get("ABHL_RUN_CODEX_WIRE_TEST") != "1":
        pytest.skip("set ABHL_RUN_CODEX_WIRE_TEST=1 for local wire probe")
    seen: dict[str, object] = {}
    received = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen["path"] = self.path
            seen["authorization"] = self.headers.get("Authorization")
            seen["body"] = json.loads(body)
            received.set()
            payload = json.dumps(
                {"error": {"message": "fixture stop", "type": "invalid_request_error"}}
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    context = run_context(tmp_path)
    runtime = CodexGoalRuntime(
        command=("codex", "app-server", "--listen", "stdio://", "--strict-config"),
        codex_home=tmp_path / "codex-home",
        base_url=f"http://127.0.0.1:{server.server_port}/v1",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        api_key="unused-local-wire-test",
    )
    try:
        session = runtime.start(context)
        assert runtime.client is not None
        runtime.client.request(
            "turn/start",
            {
                "threadId": session.thread_id,
                "input": [{"type": "text", "text": "reply with ok"}],
                "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
            },
        )
        assert received.wait(10), "Codex sent no local Responses request"
    finally:
        runtime.close()
        server.shutdown()
        server_thread.join(timeout=2)
        server.server_close()

    assert seen["path"] == "/v1/responses"
    assert seen["authorization"] == "Bearer unused-local-wire-test"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["store"] is False


@pytest.mark.live
def test_seatbelt_wrapped_app_server_starts_persistent_goal(tmp_path: Path) -> None:
    if os.environ.get("ABHL_RUN_SEATBELT_CODEX_TEST") != "1":
        pytest.skip("set ABHL_RUN_SEATBELT_CODEX_TEST=1 for wrapped App Server")
    context = run_context(tmp_path)
    hidden = tmp_path / "hidden-human-pool"
    hidden.mkdir()
    profile = write_read_isolation_profile(
        tmp_path / "goal-runtime.sb",
        denied_read_roots=(hidden,),
    )
    command = isolated_app_server_command(
        ("codex", "app-server", "--listen", "stdio://", "--strict-config"),
        profile,
    )
    runtime = CodexGoalRuntime(
        command=command,
        codex_home=tmp_path / "codex-home",
        base_url="https://example.invalid/v1",
        model="gpt-5.5",
        reasoning_effort="xhigh",
        api_key="unused-protocol-test",
    )
    try:
        session = runtime.start(context)
    finally:
        runtime.close()

    assert session.thread_id
    assert session.goal_status == "active"

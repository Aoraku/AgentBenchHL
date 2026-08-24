"""cc harness（Claude Code）的会话轮转与二进制解析。

这两组断言都来自实测事故，不是假想的边界情况。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentbench_hl.adapters.cc_goal.runtime import (
    ClaudeCodeRuntime,
    resolve_harness_binary,
)
from agentbench_hl.ports.agent_runtime import RunContext


def _context(tmp_path: Path) -> RunContext:
    # 每个 root 必须是独立目录：RunContext.validate_isolation 会拒绝
    # 工作区与人类选手池/评测器目录相互包含（那正是它该拦的东西）。
    def directory(name: str) -> Path:
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    workspace = directory("workspace")
    candidate = directory("candidate")
    gamepack = directory("gamepack")
    research = directory("research")
    return RunContext(
        objective="verify",
        initial_prompt="go",
        base_instructions="",
        developer_instructions="",
        cwd=workspace,
        candidate_root=candidate,
        gamepack_root=gamepack,
        research_root=research,
        human_pool_root=directory("pool"),
        evaluator_root=directory("evaluator"),
        # validate_isolation 要求这三个 root 都被显式放通。
        runtime_workspace_roots=(workspace, candidate, gamepack, research),
        model="claude-opus-5",
        model_provider="teamorouter",
    )


def _runtime(tmp_path: Path) -> ClaudeCodeRuntime:
    runtime = ClaudeCodeRuntime(
        binary="claude",
        agent_home=tmp_path / "agent-home",
        base_url="http://127.0.0.1:1",
        api_key="x",
        model="claude-opus-5",
    )
    # 跳过真实的 claude 探测：这里测的是参数选择逻辑，不是安装检查。
    runtime._capabilities = {
        "version": "2.1.231 (Claude Code)",
        "supports_session_id": True,
        "supports_append_system_prompt": True,
        "supports_add_dir": True,
        "supports_permission_mode": True,
    }
    return runtime


def test_rotated_thread_is_created_not_resumed(tmp_path: Path) -> None:
    """``thread_rotate_each_iteration`` 换出来的新 thread 必须用 ``--session-id``。

    这是 vk4-opus-5 实测的死法：原实现按"这是第几个 turn"选参数，
    于是第 2 轮（轮转后拿到一个全新 uuid、而全局 turn 计数已经 > 0）
    对一个从未建立过的会话发了 ``--resume``::

        claude exited 1: No conversation found with session ID: 97264be0-…

    第 1 轮的 4 个候选与 8 局全部正常，run 死在第 2 轮开头 ——
    从事件账本上看像是"模型这一轮没产出候选"，指不回真正的原因。
    """

    runtime = _runtime(tmp_path)
    context = _context(tmp_path)

    first = runtime.start(context)
    assert "--session-id" in runtime._command(first, context)

    # 第一个 turn 跑完 → 这个 thread 已建立，后续 turn 应该续接。
    runtime._fresh_threads.discard(first.thread_id)
    runtime._turn_index = 1
    resumed = runtime._command(first, context)
    assert "--resume" in resumed
    assert "--session-id" not in resumed

    # 轮转：start() 再来一次，拿到全新 thread。全局 turn 计数仍是 1。
    rotated = runtime.start(context)
    assert rotated.thread_id != first.thread_id
    command = runtime._command(rotated, context)
    assert "--session-id" in command, "轮转后的新 thread 必须新建，不能 --resume"
    assert "--resume" not in command


def test_resumed_from_disk_thread_is_not_recreated(tmp_path: Path) -> None:
    """续跑（``resume``）拿到的是上一个进程建好的会话，必须 ``--resume``。

    这条和上一条是一对：判据不能简单写成"没跑过就新建"，否则重启续跑时
    会对一个**已经存在**的会话发 ``--session-id``。
    """

    runtime = _runtime(tmp_path)
    context = _context(tmp_path)
    session = runtime.resume("2b1c9f4e-0000-4000-8000-000000000000", context)
    command = runtime._command(session, context)
    assert "--resume" in command
    assert "--session-id" not in command


def test_bare_binary_name_is_resolved_through_path() -> None:
    """``agent_binary: claude`` 这种裸名字要按 PATH 解析。

    原实现只做 ``Path(binary).exists()``，把裸名字当成相对 cwd 的路径，于是
    ``which claude`` 明明找得到，也报 ``claude binary not found: claude``——
    而那个报错指向"没装 Claude Code"，方向完全是错的。
    """

    resolved = resolve_harness_binary("sh")
    assert os.sep in resolved
    assert Path(resolved).exists()


@pytest.mark.parametrize("given", ["/usr/bin/env", "./claude"])
def test_explicit_paths_are_left_alone(given: str) -> None:
    """带路径分隔符的写法保持原样，不去查 PATH。"""

    assert resolve_harness_binary(given) == given

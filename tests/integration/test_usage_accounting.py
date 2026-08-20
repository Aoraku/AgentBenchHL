"""Token 用量记账的幂等性 —— 一条遥测事件不能有能力打死一次长跑。

背景（这组测试存在的唯一理由）
------------------------------
``_persist_agent_usage`` 曾经用 harness **内存列表下标**做幂等键
（``agent-usage:{index}``）。那个下标只在当前进程里单调，而 ``events.jsonl``
是跨进程持久的，两者一旦错位就撞键：

* ``resume`` 起新进程时 harness 的 ``events`` 从 0 重新计数，于是
  ``agent-usage:0`` 会带着一份**不同的** payload 再写一次；
* 事件存储对"同键不同 payload"抛 ``ValueError``（这个严格性是对的，
  科学事件重复记账必须暴露）；
* 于是整个 run 被一个纯记账问题终止。实测 sota-antwar 死在
  ``conflicting idempotency key: agent-usage:64``，第 43 轮，丢掉的
  信息只是几条 token 计数。

两道修复，各锁一条：
1. 序号改为按**账本里已有的 AgentTokenUsage 条数**发号 —— 只依赖持久状态；
2. 遥测事件走 ``_append_telemetry``，撞键只记诊断、不抛。
"""

from __future__ import annotations

import json
from pathlib import Path

from agentbench_hl.application.goal_led_service import GoalLedService
from agentbench_hl.ports.agent_runtime import AgentSession, RunContext
from agentbench_hl.ports.arena import MatchCase, MatchResult


class _Usage:
    """最小的 harness 用量事件（鸭子兼容 MappedAgentEvent）。"""

    def __init__(self, event_type: str, payload: dict[str, object]) -> None:
        self.event_type = event_type
        self.payload = payload


class _Runtime:
    """带内存用量列表的假 harness；``reset()`` 模拟进程重启后列表归零。"""

    harness = "codex"

    def __init__(self) -> None:
        self.events: list[_Usage] = []

    def emit(self, total: int, *, input_tokens: int | None = None) -> None:
        self.events.append(
            _Usage(
                "AgentUsageObserved",
                {
                    "total_tokens": total,
                    "input_tokens": input_tokens if input_tokens is not None else total,
                    "output_tokens": 1,
                },
            )
        )

    def reset(self) -> None:
        self.events = []

    def start(self, run_context: RunContext) -> AgentSession:
        return AgentSession("thread-1", "paused", False)

    def resume(self, session_id: str, run_context: RunContext) -> AgentSession:
        return AgentSession(session_id, "paused", False)

    def run_until_checkpoint(
        self, session: AgentSession, run_context: RunContext, _predicate: object
    ) -> AgentSession:
        return session

    def pause(self, session: AgentSession) -> AgentSession:
        return session


class _IdleArena:
    def run_case(self, case: MatchCase, candidate_root: Path) -> MatchResult:  # pragma: no cover
        raise AssertionError("这些测试从不跑对局")


def _service(tmp_path: Path, runtime: _Runtime, *, run_name: str = "run") -> GoalLedService:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    (bootstrap / "main.py").write_text("print('candidate')\n", encoding="utf-8")
    gamepack = tmp_path / "gamepack"
    gamepack.mkdir(exist_ok=True)
    return GoalLedService(
        run_root=tmp_path / run_name,
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        runtime=runtime,
        arena=_IdleArena(),
        model="glm-5.3",
        model_provider="OpenAI",
        game="antwar",
        runnable_opponent_ids=("rank01",),
        public_leaderboard=({"opponent_id": "rank01", "rank": 1, "score": 2000.0},),
    )


def _usage_events(run_root: Path) -> list[dict[str, object]]:
    path = run_root / "events.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["event_type"] == "AgentTokenUsage"
    ]


def test_usage_is_persisted_incrementally(tmp_path: Path) -> None:
    runtime = _Runtime()
    service = _service(tmp_path, runtime)

    runtime.emit(1_000)
    runtime.emit(2_000)
    service._persist_agent_usage()  # noqa: SLF001 - 直接测这个记账入口
    assert [event["payload"]["total_tokens"] for event in _usage_events(service.root)] == [
        1_000,
        2_000,
    ]

    # 再来两条：只应追加新的，已写过的不重复。
    runtime.emit(3_000)
    service._persist_agent_usage()  # noqa: SLF001
    assert [event["payload"]["total_tokens"] for event in _usage_events(service.root)] == [
        1_000,
        2_000,
        3_000,
    ]


def test_restart_does_not_collide_on_usage_keys(tmp_path: Path) -> None:
    """这是 sota-antwar 的真实死因，必须锁死。

    重启后 harness 内存列表从 0 重数；若幂等键跟着下标回退，就会用
    ``agent-usage:0`` 去覆盖一份不同 payload 的旧事件并抛 ValueError。
    序号按账本条数发放之后，重启只会**接着往下写**。
    """

    runtime = _Runtime()
    service = _service(tmp_path, runtime)
    runtime.emit(1_000)
    runtime.emit(2_000)
    service._persist_agent_usage()  # noqa: SLF001

    # 进程重启：harness 列表归零、payload 与上次不同，服务对象也是新的。
    runtime.reset()
    resumed = _service(tmp_path, runtime)
    runtime.emit(7_777)  # 与 agent-usage:0 的旧 payload 完全不同
    resumed._persist_agent_usage()  # noqa: SLF001 - 不允许抛异常

    events = _usage_events(resumed.root)
    assert [event["payload"]["total_tokens"] for event in events] == [1_000, 2_000, 7_777]
    keys = [event["idempotency_key"] for event in events]
    assert keys == ["agent-usage:0", "agent-usage:1", "agent-usage:2"]
    assert len(set(keys)) == len(keys), "幂等键必须互不相同"


def test_many_restarts_keep_accounting_monotone(tmp_path: Path) -> None:
    """反复重启（长跑里会发生几十次）不应产生任何键冲突。"""

    runtime = _Runtime()
    for round_index in range(6):
        service = _service(tmp_path, runtime)
        runtime.reset()
        runtime.emit(1_000 * (round_index + 1))
        service._persist_agent_usage()  # noqa: SLF001

    events = _usage_events(tmp_path / "run")
    assert [event["payload"]["total_tokens"] for event in events] == [
        1_000,
        2_000,
        3_000,
        4_000,
        5_000,
        6_000,
    ]
    assert len({event["idempotency_key"] for event in events}) == 6


def test_telemetry_conflict_is_recorded_not_raised(tmp_path: Path) -> None:
    """遥测撞键只留诊断：为几条 token 计数终止一次长跑是明显的代价错配。"""

    runtime = _Runtime()
    service = _service(tmp_path, runtime)
    service._append_telemetry("AgentTokenUsage", {"total_tokens": 1}, "dup-key")  # noqa: SLF001
    service._append_telemetry("AgentTokenUsage", {"total_tokens": 2}, "dup-key")  # noqa: SLF001

    types = [
        json.loads(line)["event_type"]
        for line in (service.root / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert types.count("AgentTokenUsage") == 1
    assert "TelemetryAppendSkipped" in types


def test_scientific_events_still_reject_conflicts(tmp_path: Path) -> None:
    """对照：科学事件的严格幂等**不能**被这次放宽波及。

    对局结果/指标/快照撞键说明有真实的重复记账，必须立刻暴露而不是吞掉。
    """

    runtime = _Runtime()
    service = _service(tmp_path, runtime)
    service._append("IterationMetricsFinalized", {"win_rate": 1.0}, "metrics:1")  # noqa: SLF001
    try:
        service._append("IterationMetricsFinalized", {"win_rate": 0.0}, "metrics:1")  # noqa: SLF001
    except ValueError as error:
        assert "conflicting idempotency key" in str(error)
    else:  # pragma: no cover - 走到这里说明严格性被破坏了
        raise AssertionError("科学事件的同键不同 payload 必须抛错")

"""全 run token 计量：会话轮转下必须按 thread 分段求和。

背景（这组测试存在的唯一理由）
------------------------------
harness 报的 ``total_tokens`` 是**会话累计值**（codex 的 ``tokenUsage/updated``
给的是当前 thread 的总量，单调增）。因此单个 thread 内要取峰值——求和会把同一份
累计量重复叠加。

但 ``thread_rotate_each_iteration`` 之后每轮都换新 thread，计数**归零重来**。
此时对全 run 取全局 max 就只记住了"最贵的那一段"：实测 sota-antwar 跑了 44 段、
各段峰值 69k~137k，指标里报 169k，而真实总消耗约 4.9M —— 差 29 倍。

后果是两个，都很严重：
* ``budget.tokens`` 守卫永远不会触发（它拿着一个被低估 29 倍的数）；
* token 曲线画成一条阶梯（只在"出现更贵的段"时才上升），
  完全看不出真实成本走势。

正确口径：**段内取峰值，跨段求和**。分段边界用会话事件识别。
"""

from __future__ import annotations

from pathlib import Path

from agentbench_hl.application.goal_led_service import GoalLedService
from agentbench_hl.ports.agent_runtime import AgentSession, RunContext
from agentbench_hl.ports.arena import MatchCase, MatchResult


class _Runtime:
    harness = "codex"

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


def _service(tmp_path: Path) -> GoalLedService:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    (bootstrap / "main.py").write_text("print('candidate')\n", encoding="utf-8")
    gamepack = tmp_path / "gamepack"
    gamepack.mkdir(exist_ok=True)
    return GoalLedService(
        run_root=tmp_path / "run",
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        runtime=_Runtime(),
        arena=_IdleArena(),
        model="glm-5.3",
        model_provider="OpenAI",
        game="antwar",
        runnable_opponent_ids=("rank01",),
        public_leaderboard=({"opponent_id": "rank01", "rank": 1, "score": 2000.0},),
    )


def _usage(service: GoalLedService, total: int, tag: str) -> None:
    service._append(  # noqa: SLF001
        "AgentTokenUsage",
        {"harness": "codex", "input_tokens": total, "total_tokens": total},
        f"usage:{tag}",
    )


def _rotate(service: GoalLedService, tag: str) -> None:
    service._append("GoalSessionRotated", {"iteration": tag}, f"rotated:{tag}")  # noqa: SLF001


def test_single_thread_takes_peak_not_sum(tmp_path: Path) -> None:
    """段内是累计值语义：求和会把同一份用量叠加好几遍。"""

    service = _service(tmp_path)
    _usage(service, 10_000, "a")
    _usage(service, 25_000, "b")
    _usage(service, 40_000, "c")
    assert service._token_total() == 40_000  # noqa: SLF001


def test_rotated_threads_are_summed_across_segments(tmp_path: Path) -> None:
    """这是被修掉的真实 bug：跨 thread 必须相加，不能取全局 max。"""

    service = _service(tmp_path)
    _usage(service, 60_000, "s1a")
    _usage(service, 120_000, "s1b")  # 第 1 段峰值 120k
    _rotate(service, "2")
    _usage(service, 30_000, "s2a")
    _usage(service, 90_000, "s2b")  # 第 2 段峰值 90k
    _rotate(service, "3")
    _usage(service, 70_000, "s3a")  # 第 3 段峰值 70k

    # 全局 max 会给出 120k（只记住最贵那段）；正确答案是三段之和。
    assert service._token_total() == 280_000  # noqa: SLF001


def test_many_rotations_stay_monotone(tmp_path: Path) -> None:
    """每轮轮转几十次（实测 44 次）时累计量必须单调增长。"""

    service = _service(tmp_path)
    running: list[int] = []
    for index in range(1, 21):
        _rotate(service, str(index))
        _usage(service, 100_000, f"seg{index}")
        running.append(service._token_total())  # noqa: SLF001

    assert running == [100_000 * step for step in range(1, 21)]
    assert running == sorted(running), "累计 token 不允许下降"


def test_no_usage_events_reports_none(tmp_path: Path) -> None:
    """没有用量数据就返回 None —— 绝不估算一个数。"""

    service = _service(tmp_path)
    assert service._token_total() is None  # noqa: SLF001
    _rotate(service, "1")
    assert service._token_total() is None  # noqa: SLF001


def test_budget_guard_sees_the_corrected_total(tmp_path: Path) -> None:
    """预算守卫必须拿到修正后的数，否则长跑永远不会因 token 停下。"""

    service = _service(tmp_path)
    service.token_budget = 250_000
    _usage(service, 120_000, "s1")
    _rotate(service, "2")
    _usage(service, 90_000, "s2")
    assert service.budget_status()["exhausted"] is None  # 210k < 250k

    _rotate(service, "3")
    _usage(service, 80_000, "s3")  # 合计 290k
    assert service.budget_status()["exhausted"] == "token_budget"

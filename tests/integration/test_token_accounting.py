"""全 run token 计量：每次模型请求的用量必须逐次相加。

背景（这组测试存在的唯一理由）
------------------------------
codex 的 ``thread/tokenUsage/updated`` 里有两块：``tokenUsage.total``（thread
累计）与 ``tokenUsage.last``（**这一次请求**的用量）。
:mod:`agentbench_hl.adapters.codex_goal.event_mapper` 取的是 ``last``，
所以事件流里每条 ``AgentTokenUsage`` 都是一次请求的独立花费。

历史 bug：口径读反了
-------------------
原实现把它当成"会话累计值"，于是按会话事件切段、段内取 max、跨段相加。
语义反了，后果是实测的两种错法同时出现：

* ``sota-antwar`` 第 2~11 轮 token 全报 137631，**连续 10 轮一动不动**
  （段内取 max，而后续请求恰好都比那一次便宜）；
* 6 个 4 轮 run 逐轮精确翻倍（snakego4: 112526 → 260286 → 520572 → 1041144），
  因为每多一个 rotate 边界就把同一批用量再叠一遍。

正确口径：**逐次求和**。agent 每发一次模型请求都要重发整个上下文，
那次请求的 input+output 就是那次的花费，全 run 花费 = 逐次相加。
实测 r4b 4 轮 209 次请求合计 13.79M token，原口径只报 0.99M（低估 14 倍）。
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


def test_each_request_is_added(tmp_path: Path) -> None:
    """每条用量事件是一次请求的花费，所以直接相加。"""

    service = _service(tmp_path)
    _usage(service, 10_000, "a")
    _usage(service, 25_000, "b")
    _usage(service, 40_000, "c")
    assert service._token_total() == 75_000  # noqa: SLF001


def test_rotation_does_not_change_the_total(tmp_path: Path) -> None:
    """会话轮转只是换 thread，不改变已经花掉的钱。

    这是被修掉的真实 bug 的镜像：原实现让 rotate 边界参与记账，于是
    "换个 thread"就能把同一批用量再叠一遍（实测逐轮精确翻倍）。
    """

    service = _service(tmp_path)
    _usage(service, 60_000, "s1a")
    _usage(service, 120_000, "s1b")
    _rotate(service, "2")
    _usage(service, 30_000, "s2a")
    _usage(service, 90_000, "s2b")
    _rotate(service, "3")
    _usage(service, 70_000, "s3a")

    assert service._token_total() == 370_000  # noqa: SLF001


def test_totals_are_monotone_across_many_rotations(tmp_path: Path) -> None:
    """每轮轮转几十次（实测 44 次）时累计量必须单调增长。"""

    service = _service(tmp_path)
    running: list[int] = []
    for index in range(1, 21):
        _rotate(service, str(index))
        _usage(service, 100_000, f"seg{index}")
        running.append(service._token_total())  # noqa: SLF001

    assert running == [100_000 * step for step in range(1, 21)]
    assert running == sorted(running), "累计 token 不允许下降"


def test_only_usage_events_are_counted(tmp_path: Path) -> None:
    """别的事件也带 total_tokens（例如指标事件），不能一起加进来。

    不加这道过滤的话，指标事件每轮回写一次 ``total_tokens``，
    下一轮读账本时它会被当成一次新的模型请求算进去 —— 自我叠加。
    """

    service = _service(tmp_path)
    _usage(service, 50_000, "a")
    service._append(  # noqa: SLF001
        "IterationMetricsFinalized",
        {"research_iteration": 1, "total_tokens": 50_000},
        "metrics:1",
    )
    assert service._token_total() == 50_000  # noqa: SLF001


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

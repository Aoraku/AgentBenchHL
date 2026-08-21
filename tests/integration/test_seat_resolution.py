"""座次名必须以游戏定义为准，不能被 agent 写的值带跑。

背景（这组测试存在的唯一理由）
------------------------------
座次**名字**由 A 仓 ``games/<game>/game.yaml`` 唯一定义，而 agent 会照抄 prompt
示例里的 ``P0`` / ``P1``。对 antwar 这类对称游戏恰好一致，但 rollman 的座次叫
``rollman`` / ``ghost``。

原实现是 ``交集 or request.roles``：交集为空时回退到 agent 写的值，
恰好在"agent 全写错"这个唯一需要保护的场景下失效。后果是每一局都以
``role P0 is not one of ('rollman', 'ghost')`` 失败，而指标上只显示
"对局 0/N 完成"——看起来像对局跑不起来或候选有问题，完全看不出是座次名的事。
实测 rollman 烟测连续两轮 8 局全灭。

同时要保住 agent 的正当选择权：它可以决定"本轮只打一个座次"
（例如只验证先手表现），这不能被一并砍掉。
"""

from __future__ import annotations

from pathlib import Path

from agentbench_hl.application.goal_led_protocol import MatchRequest
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


def _service(tmp_path: Path, roles: tuple[str, ...]) -> GoalLedService:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir(exist_ok=True)
    (bootstrap / "main.py").write_text("print('candidate')\n", encoding="utf-8")
    gamepack = tmp_path / "gamepack"
    gamepack.mkdir(exist_ok=True)
    return GoalLedService(
        run_root=tmp_path / f"run-{'-'.join(roles)}",
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        runtime=_Runtime(),
        arena=_IdleArena(),
        model="gpt-5.6-luna",
        model_provider="OpenAI",
        game="rollman",
        roles=roles,
        runnable_opponent_ids=("rank01",),
        public_leaderboard=({"opponent_id": "rank01", "rank": 1, "score": 1000.0},),
    )


def _request(roles: tuple[str, ...]) -> MatchRequest:
    return MatchRequest(
        request_id="round-1",
        candidate_ids=("v001",),
        opponent_id="rank01",
        roles=roles,
        seeds=(1,),
        rationale="test",
    )


def test_illegal_seat_names_fall_back_to_game_definition(tmp_path: Path) -> None:
    """agent 写的座次名全不合法时，必须退回游戏定义，而不是采纳它写的。

    这是 rollman 烟测 8 局全灭的直接原因。
    """

    service = _service(tmp_path, ("rollman", "ghost"))
    resolved = service._effective_roles(_request(("P0", "P1")))  # noqa: SLF001
    assert resolved == ("rollman", "ghost")


def test_agent_may_narrow_to_a_single_valid_seat(tmp_path: Path) -> None:
    """agent 的正当选择权要保住：只打一个座次是合法诉求。"""

    service = _service(tmp_path, ("P0", "P1"))
    assert service._effective_roles(_request(("P0",))) == ("P0",)  # noqa: SLF001
    assert service._effective_roles(_request(("P1",))) == ("P1",)  # noqa: SLF001


def test_partially_valid_seats_keep_only_the_valid_ones(tmp_path: Path) -> None:
    """一半写对一半写错时，只保留写对的那部分。"""

    service = _service(tmp_path, ("rollman", "ghost"))
    resolved = service._effective_roles(_request(("ghost", "P1")))  # noqa: SLF001
    assert resolved == ("ghost",)


def test_seat_order_follows_the_request(tmp_path: Path) -> None:
    """座次顺序按 agent 请求的来——它可能想先测后手。"""

    service = _service(tmp_path, ("P0", "P1"))
    assert service._effective_roles(_request(("P1", "P0"))) == ("P1", "P0")  # noqa: SLF001


def test_empty_request_uses_all_game_seats(tmp_path: Path) -> None:
    service = _service(tmp_path, ("rollman", "ghost"))
    assert service._effective_roles(_request(())) == ("rollman", "ghost")  # noqa: SLF001


def test_prompt_tells_the_agent_the_real_seat_names(tmp_path: Path) -> None:
    """prompt 必须写明真实座次名，否则 agent 只能照抄示例里的 P0/P1。"""

    service = _service(tmp_path, ("rollman", "ghost"))
    prompt = service._developer_instructions(iteration=1, cleared=0)  # noqa: SLF001
    assert "rollman" in prompt and "ghost" in prompt

"""连续奖励：分差必须端到 agent 面前，胜率二值化会丢掉全部梯度。

背景
----
一轮全败时 ``win_rate`` 恒为 0，看起来像"没有信号"。但这不是二值 reward 的 RL：
``score_margin``（终局分差）和 ``rounds``（撑住多少回合）都是逐局连续量，
早就落在事件里，只是从来没有出现在反馈里。实测 antwar2 对 rank1 连续 15 轮
胜率恒为 0，那 15 轮里分差完全可能在收窄——那就是有梯度的。

这里锁两件事：
1. 汇总里必须有逐候选分差（``margin_by_candidate``）；
2. ``best_candidate_id`` 在得分率打平时用分差破平——否则全败轮里它退化成
   字典序，下一轮基线是随机挑的，那才是真正的 0 信号。
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
        public_leaderboard=({"opponent_id": "rank01", "rank": 1, "score": 1000.0},),
    )


def _loss(candidate: str, margin: float, rounds: int = 400) -> dict[str, object]:
    return {
        "candidate_id": candidate,
        "opponent_id": "rank01",
        "role": "P0",
        "seed": 1,
        "status": "complete",
        "result": "loss",
        "points": 0.0,
        "score_margin": margin,
        "rounds": rounds,
    }


def _win(candidate: str, margin: float) -> dict[str, object]:
    row = _loss(candidate, margin)
    row.update({"result": "win", "points": 1.0})
    return row


def test_all_loss_round_still_reports_margins(tmp_path: Path) -> None:
    """全败也要有连续量，否则这一轮对 agent 就是纯噪声。"""

    service = _service(tmp_path)
    summary = service._summarize(  # noqa: SLF001
        [_loss("a", -5.0), _loss("a", -7.0), _loss("b", -80.0), _loss("b", -90.0)]
    )

    assert summary["win_rate"] == 0.0
    assert summary["margin_mean"] == -45.5
    assert summary["margin_best"] == -5.0

    by_candidate = summary["margin_by_candidate"]
    assert by_candidate["a"]["mean"] == -6.0
    assert by_candidate["a"]["best"] == -5.0
    assert by_candidate["b"]["mean"] == -85.0
    assert by_candidate["a"]["rounds_mean"] == 400.0


def test_margin_breaks_the_tie_when_everyone_loses(tmp_path: Path) -> None:
    """全败轮里 best_candidate 必须是"离赢最近"的那个，而不是字典序第一个。"""

    service = _service(tmp_path)
    summary = service._summarize(  # noqa: SLF001
        [_loss("z_close", -3.0), _loss("a_blown_out", -120.0)]
    )
    assert summary["best_candidate_id"] == "z_close"
    assert summary["win_rate"] == 0.0


def test_points_still_dominate_margin(tmp_path: Path) -> None:
    """分差只是破平手段：赢了的候选不能被"输得漂亮"的候选挤掉。"""

    service = _service(tmp_path)
    summary = service._summarize(  # noqa: SLF001
        [_win("winner", -200.0), _loss("pretty_loser", -1.0)]
    )
    assert summary["best_candidate_id"] == "winner"


def test_missing_margins_do_not_break_summary(tmp_path: Path) -> None:
    """老对战器可能不报 score_margin，此时诚实留空而不是编 0。"""

    service = _service(tmp_path)
    row = _loss("a", -1.0)
    del row["score_margin"]
    summary = service._summarize([row])  # noqa: SLF001
    assert summary["margin_mean"] is None
    assert summary["margin_best"] is None
    assert summary["margin_by_candidate"] == {}


def test_margins_reach_the_agent_feedback(tmp_path: Path) -> None:
    """光算出来不算完 —— 必须出现在下发给 agent 的那段文字里。"""

    service = _service(tmp_path)
    summary = service._summarize([_loss("a", -5.0), _loss("b", -90.0)])  # noqa: SLF001
    headline = service._feedback_headline(summary)  # noqa: SLF001

    assert "胜率 0.00%" in headline
    assert "逐候选分差" in headline
    # 离赢最近的候选排在前面，agent 才知道该继续推哪个方向。
    assert headline.index("a: 平均分差 -5") < headline.index("b: 平均分差 -90")
    assert "均撑 400" in headline


def test_headline_without_margins_stays_clean(tmp_path: Path) -> None:
    """没有分差数据时不要留下空标题党段落。"""

    service = _service(tmp_path)
    row = _loss("a", -1.0)
    del row["score_margin"]
    headline = service._feedback_headline(service._summarize([row]))  # noqa: SLF001
    assert "逐候选分差" not in headline
    assert "胜率" in headline

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentbench_hl.application.goal_led_service import GoalLedService
from agentbench_hl.ports.agent_runtime import AgentSession, RunContext
from agentbench_hl.ports.arena import MatchCase, MatchResult


def _agentbench_root() -> Path | None:
    """定位 A 仓：自然语言回放由 A 翻译，没有 A 就只能跳过那部分断言。"""

    value = os.environ.get("AGENTBENCH_ROOT")
    if value and (Path(value) / "games").is_dir():
        return Path(value).resolve()
    sibling = Path(__file__).resolve().parents[2].parent / "AgentBench"
    return sibling.resolve() if (sibling / "games").is_dir() else None


AGENTBENCH_ROOT = _agentbench_root()


class RequestingRuntime:
    def __init__(self) -> None:
        self.thread_ids: list[str] = []
        self.prompts: list[str] = []

    def start(self, run_context: RunContext) -> AgentSession:
        self.thread_ids.append("thread-I")
        return AgentSession("thread-I", "paused", False)

    def resume(self, session_id: str, run_context: RunContext) -> AgentSession:
        self.thread_ids.append(session_id)
        return AgentSession(session_id, "paused", False)

    def run_until_checkpoint(
        self,
        session: AgentSession,
        run_context: RunContext,
        _checkpoint_predicate: object,
    ) -> AgentSession:
        self.prompts.append(run_context.initial_prompt)
        assert json.loads((run_context.cwd / "leaderboard.json").read_text()) == {
            "opponents": [{"opponent_id": "rank01", "rank": 1, "score": 2000.0}],
            "schema_version": "1.0",
        }
        request = run_context.cwd / ".agentbench" / "match_request.json"
        request.parent.mkdir(exist_ok=True)
        if len(self.prompts) == 1:
            request.write_text(
                json.dumps(
                    {
                        "request_id": "first-rank01",
                        "candidate_ids": ["v000"],
                        "opponent_id": "rank01",
                        "roles": ["P0"],
                        "seeds": [1],
                        "rationale": "Use the strongest player's public opening replay.",
                    }
                ),
                encoding="utf-8",
            )
        return session

    def pause(self, session: AgentSession) -> AgentSession:
        return session


class PublicReplayArena:
    def run_case(self, case: MatchCase, candidate_root: Path) -> MatchResult:
        replay = candidate_root / "result-replay.json"
        replay.write_text('[{"round_state":{"camps":[5,0],"winner":0}}]', encoding="utf-8")
        trace = candidate_root / "result-trace.jsonl"
        trace.write_text('{"kind":"public_state"}\n', encoding="utf-8")
        return MatchResult(
            case=case,
            status="complete",
            result="win",
            points=1.0,
            score_margin=5.0,
            rounds=1,
            payload={"terminal_base_hp": (5.0, 0.0)},
            replay_path=replay,
            trace_path=trace,
        )


class TimingOutRuntime(RequestingRuntime):
    def run_until_checkpoint(
        self,
        session: AgentSession,
        run_context: RunContext,
        _checkpoint_predicate: object,
    ) -> AgentSession:
        raise TimeoutError("deliberate long-goal interruption")


class FeedbackTimeoutRuntime(RequestingRuntime):
    def run_until_checkpoint(
        self,
        session: AgentSession,
        run_context: RunContext,
        _checkpoint_predicate: object,
    ) -> AgentSession:
        self.prompts.append(run_context.initial_prompt)
        request = run_context.cwd / ".agentbench" / "action.json"
        request.parent.mkdir(exist_ok=True)
        if len(self.prompts) == 1:
            request.write_text(
                json.dumps(
                    {
                        "request_id": "first-rank01",
                        "candidate_ids": ["v000"],
                        "opponent_id": "rank01",
                        "roles": ["P0"],
                        "seeds": [1],
                        "rationale": "Use the strongest player's public opening replay.",
                    }
                ),
                encoding="utf-8",
            )
            return session
        raise TimeoutError("feedback turn interrupted")


def test_goal_led_bridge_keeps_one_thread_and_returns_public_rank01_feedback(
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "main.py").write_text("print('candidate')\n", encoding="utf-8")
    gamepack = tmp_path / "gamepack"
    gamepack.mkdir()
    runtime = RequestingRuntime()
    service = GoalLedService(
        run_root=tmp_path / "run",
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        runtime=runtime,
        arena=PublicReplayArena(),
        model="gpt-5.6",
        model_provider="OpenAI",
        game="antwar2",
        agentbench_root=AGENTBENCH_ROOT,
        runnable_opponent_ids=("rank01",),
        public_leaderboard=({"opponent_id": "rank01", "rank": 1, "score": 2000.0},),
    )

    outcome = service.start()

    feedback = outcome.workspace / "feedback" / "first-rank01" / "feedback.json"
    assert outcome.thread_id == "thread-I"
    assert runtime.thread_ids == ["thread-I"]
    assert len(runtime.prompts) == 2
    assert json.loads(feedback.read_text(encoding="utf-8"))["matches"][0]["result"] == "win"
    case_root = outcome.workspace / "feedback" / "first-rank01" / "v000" / "P0-seed-1"
    replay = case_root / "replay.json"
    assert replay.is_file()
    events = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
    assert "GoalLedStarted" in events
    assert "GoalMatchCompleted" in events
    # 容器边界必须在 agent 开跑前被真的验过一次，并留下证据。
    assert "ContainerBoundaryVerified" in events

    # Feedback 通道的主体是**自然语言回放**（由 A 仓翻译）。
    # 没有它，agent 每轮都得自己写解析代码去读裸 JSON——那正是被砍掉的 63% 墙钟。
    if AGENTBENCH_ROOT is None:
        pytest.skip("需要 AGENTBENCH_ROOT 才能验证自然语言回放")
    narration = case_root / "replay.md"
    assert narration.is_file(), "反馈里缺少自然语言回放 replay.md"
    text = narration.read_text(encoding="utf-8")
    assert text.startswith("# ") and "## 判决" in text
    # 视角要落地：读它的是候选自己，不是中立解说。
    assert "你(P0)" in text
    row = json.loads(feedback.read_text(encoding="utf-8"))["matches"][0]
    assert row["narration_path"] == str(narration)
    # 并且要提示 agent 去读它。
    assert "replay.md" in runtime.prompts[-1]


def test_goal_led_persists_thread_before_the_first_long_turn(tmp_path: Path) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "main.py").write_text("print('candidate')\n", encoding="utf-8")
    gamepack = tmp_path / "gamepack"
    gamepack.mkdir()
    service = GoalLedService(
        run_root=tmp_path / "run",
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        runtime=TimingOutRuntime(),
        arena=PublicReplayArena(),
        model="gpt-5.6",
        model_provider="OpenAI",
        runnable_opponent_ids=("rank01",),
        public_leaderboard=({"opponent_id": "rank01", "rank": 1, "score": 2000.0},),
    )

    with pytest.raises(TimeoutError, match="long-goal"):
        service.start()

    assert json.loads((tmp_path / "run" / "goal-led-state.json").read_text()) == {
        "request_count": 0,
        "schema_version": "1.0",
        "thread_id": "thread-I",
    }


def test_goal_led_recovers_unacknowledged_feedback_without_replaying_matches(
    tmp_path: Path,
) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "main.py").write_text("print('candidate')\n", encoding="utf-8")
    gamepack = tmp_path / "gamepack"
    gamepack.mkdir()
    interrupted = FeedbackTimeoutRuntime()
    service = GoalLedService(
        run_root=tmp_path / "run",
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        runtime=interrupted,
        arena=PublicReplayArena(),
        model="gpt-5.6",
        model_provider="OpenAI",
        runnable_opponent_ids=("rank01",),
        public_leaderboard=({"opponent_id": "rank01", "rank": 1, "score": 2000.0},),
    )

    with pytest.raises(TimeoutError, match="feedback turn"):
        service.start()

    legacy = service.workspace / ".agentbench" / "match_request.json"
    archived = service.workspace / ".agentbench" / "processed-requests" / "first-rank01.json"
    legacy.write_text(archived.read_text(encoding="utf-8"), encoding="utf-8")

    resumed = RequestingRuntime()
    recovery = GoalLedService(
        run_root=tmp_path / "run",
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        runtime=resumed,
        arena=PublicReplayArena(),
        model="gpt-5.6",
        model_provider="OpenAI",
        runnable_opponent_ids=("rank01",),
        public_leaderboard=({"opponent_id": "rank01", "rank": 1, "score": 2000.0},),
    )

    outcome = recovery.advance()

    assert outcome.request_id == "first-rank01"
    assert outcome.match_count == 1
    assert len(resumed.prompts) == 1
    events = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8")
    assert events.count('"event_type":"GoalMatchCompleted"') == 1
    assert '"event_type":"GoalFeedbackDelivered"' in events

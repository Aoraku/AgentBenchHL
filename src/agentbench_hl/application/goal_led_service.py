"""Thin bridge for a single Goal-led research run.

The Goal decides every scientific action.  This module only executes the
declarative official matches that the Goal writes into its isolated workspace
and returns public evidence to that same Goal thread.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.application.goal_led_protocol import MatchRequest
from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.ports.agent_runtime import AgentRuntime, AgentSession, RunContext
from agentbench_hl.ports.arena import Arena, MatchCase, MatchResult


@dataclass(frozen=True)
class GoalLedOutcome:
    thread_id: str
    workspace: Path
    request_id: str
    match_count: int


class GoalLedService:
    def __init__(
        self,
        *,
        run_root: str | Path,
        bootstrap_root: str | Path,
        gamepack_root: str | Path,
        runtime: AgentRuntime,
        arena: Arena,
        model: str,
        model_provider: str,
        runnable_opponent_ids: tuple[str, ...],
        public_leaderboard: tuple[dict[str, object], ...],
    ) -> None:
        self.root = Path(run_root).resolve()
        self.bootstrap_root = Path(bootstrap_root).resolve()
        self.gamepack_root = Path(gamepack_root).resolve()
        self.runtime = runtime
        self.arena = arena
        self.model = model
        self.model_provider = model_provider
        self.runnable_opponent_ids = frozenset(runnable_opponent_ids)
        self.public_leaderboard = tuple(
            sorted(public_leaderboard, key=lambda row: int(row["rank"]))
        )
        self.events = JsonlEventStore(self.root / "events.jsonl")

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def _state_path(self) -> Path:
        return self.root / "goal-led-state.json"

    def _append(self, event_type: str, payload: dict[str, object], key: str) -> None:
        self.events.append(FinalizedEvent.create(event_type, payload, key))

    def _context(self, prompt: str) -> RunContext:
        research = self.workspace / "research"
        return RunContext(
            objective="从规则出发，经公开 rank01 回放持续研究并最终击败全部可运行人类",
            initial_prompt=prompt,
            base_instructions=(
                "你是唯一的研究控制器。自行规划、改代码、阅读公开回放、写入经验；"
                "不要联网、不要读取人类源码、不要网格搜索。"
            ),
            developer_instructions=(
                "通过 .agentbench/action.json 提交 Action（兼容 match_request.json）。"
                "首个请求必须是 rank01；后续由你决定对手、角色、seed 与候选数量。"
            ),
            cwd=self.workspace,
            candidate_root=self.workspace,
            gamepack_root=self.gamepack_root,
            research_root=research,
            human_pool_root=self.root / "hidden-human-pool",
            evaluator_root=self.root / "hidden-certification",
            runtime_workspace_roots=(self.workspace, self.gamepack_root, research),
            writable_workspace_roots=(self.workspace, research),
            model=self.model,
            model_provider=self.model_provider,
        )

    def _write_state(self, thread_id: str, request_count: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "thread_id": thread_id,
                    "request_count": request_count,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _load_state(self) -> tuple[str, int]:
        value = json.loads(self._state_path.read_text(encoding="utf-8"))
        thread_id = value.get("thread_id")
        count = value.get("request_count")
        if not isinstance(thread_id, str) or not thread_id or not isinstance(count, int):
            raise ValueError("goal-led state is invalid")
        return thread_id, count

    def _turn(self, session: AgentSession, prompt: str) -> AgentSession:
        context = self._context(prompt)
        return self.runtime.run_until_checkpoint(
            session,
            context,
            lambda event: getattr(event, "event_type", "") == "AgentTurnCompleted",
        )

    def _request_path(self) -> Path | None:
        action = self.workspace / ".agentbench" / "action.json"
        if action.is_file():
            return action
        legacy = self.workspace / ".agentbench" / "match_request.json"
        if not legacy.is_file():
            return None
        try:
            request = MatchRequest.from_path(legacy)
        except ValueError:
            return legacy
        archived = (
            self.workspace
            / ".agentbench"
            / "processed-requests"
            / f"{request.request_id}.json"
        )
        return None if archived.is_file() else legacy

    def _consume_request(self) -> MatchRequest:
        path = self._request_path()
        if path is None:
            raise ValueError("Goal did not submit a new action.json")
        request = MatchRequest.from_path(path)
        if request.opponent_id not in self.runnable_opponent_ids:
            raise ValueError(f"request names unknown or unrunnable opponent: {request.opponent_id}")
        archive = self.workspace / ".agentbench" / "processed-requests"
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / f"{request.request_id}.json"
        if destination.exists():
            raise ValueError(f"request_id was already consumed: {request.request_id}")
        path.replace(destination)
        return request

    def _snapshot_root(self, request: MatchRequest, candidate_id: str) -> Path:
        if len(request.candidate_ids) == 1:
            source = self.workspace
        else:
            source = self.workspace / ".agentbench" / "rollouts" / candidate_id
        if not (source / "main.py").is_file():
            raise ValueError(f"candidate snapshot {candidate_id} has no main.py")
        destination = self.root / "snapshots" / candidate_id
        if destination.exists():
            raise ValueError(f"candidate snapshot already exists: {candidate_id}")
        ignored = shutil.ignore_patterns("feedback", "research", "snapshots", "processed-requests")
        shutil.copytree(source, destination, ignore=ignored)
        self._append(
            "GoalVersionSnapshot",
            {"candidate_id": candidate_id, "path": str(destination)},
            f"goal-led-snapshot:{candidate_id}",
        )
        return destination

    @staticmethod
    def _result_row(
        result: MatchResult, replay: Path | None, trace: Path | None
    ) -> dict[str, object]:
        return {
            "candidate_id": result.case.candidate_id,
            "opponent_id": result.case.opponent_id,
            "role": result.case.role,
            "seed": result.case.seed,
            "status": result.status,
            "result": result.result,
            "points": result.points,
            "score_margin": result.score_margin,
            "rounds": result.rounds,
            "replay_path": None if replay is None else str(replay),
            "trace_path": None if trace is None else str(trace),
            "error": result.error,
        }

    def _execute(self, request: MatchRequest) -> tuple[Path, int]:
        feedback_root = self.workspace / "feedback" / request.request_id
        feedback_root.mkdir(parents=True, exist_ok=False)
        self._append(
            "GoalMatchRequested",
            {
                "request_id": request.request_id,
                "candidate_ids": list(request.candidate_ids),
                "opponent_id": request.opponent_id,
                "roles": list(request.roles),
                "seeds": list(request.seeds),
                "rationale": request.rationale,
            },
            f"goal-led-request:{request.request_id}",
        )
        rows: list[dict[str, object]] = []
        for candidate_id in request.candidate_ids:
            candidate_root = self._snapshot_root(request, candidate_id)
            for role in request.roles:
                for seed in request.seeds:
                    result = self.arena.run_case(
                        MatchCase(candidate_id, request.opponent_id, role, seed),
                        candidate_root,
                    )
                    case_root = feedback_root / candidate_id / f"{role}-seed-{seed}"
                    case_root.mkdir(parents=True, exist_ok=True)
                    replay = None
                    trace = None
                    if result.replay_path is not None and result.replay_path.is_file():
                        replay = case_root / "replay.json"
                        shutil.copy2(result.replay_path, replay)
                    if result.trace_path is not None and result.trace_path.is_file():
                        trace = case_root / "public-trace.jsonl"
                        shutil.copy2(result.trace_path, trace)
                    row = self._result_row(result, replay, trace)
                    rows.append(row)
                    self._append(
                        "GoalMatchCompleted",
                        {"request_id": request.request_id, **row},
                        f"goal-led-match:{request.request_id}:{candidate_id}:{role}:{seed}",
                    )
        feedback = feedback_root / "feedback.json"
        feedback.write_text(
            json.dumps(
                {"request_id": request.request_id, "rationale": request.rationale, "matches": rows},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return feedback, len(rows)

    def _pending_feedback(self) -> tuple[MatchRequest, Path, int] | None:
        delivered = {
            str(event.payload["request_id"])
            for event in self.events.read_all()
            if event.event_type == "GoalFeedbackDelivered"
            and isinstance(event.payload.get("request_id"), str)
        }
        for event in reversed(self.events.read_all()):
            if event.event_type != "GoalMatchRequested":
                continue
            request_id = event.payload.get("request_id")
            if not isinstance(request_id, str) or request_id in delivered:
                continue
            archived = self.workspace / ".agentbench" / "processed-requests" / f"{request_id}.json"
            feedback = self.workspace / "feedback" / request_id / "feedback.json"
            if not archived.is_file() or not feedback.is_file():
                continue
            request = MatchRequest.from_path(archived)
            match_count = sum(
                event.event_type == "GoalMatchCompleted"
                and event.payload.get("request_id") == request_id
                for event in self.events.read_all()
            )
            return request, feedback, match_count
        return None

    def _deliver_feedback(
        self,
        session: AgentSession,
        request: MatchRequest,
        feedback: Path,
        match_count: int,
        request_count: int,
    ) -> GoalLedOutcome:
        session = self._turn(
            session,
            (
                f"官方比赛反馈已写入 {feedback.relative_to(self.workspace)}。"
                "阅读 replay.json、public-trace.jsonl 和 feedback.json；更新 research/ 中的"
                "成功经验与失败假设，修改策略，并在准备好下一步时写新的 action.json。"
            ),
        )
        self._append(
            "GoalFeedbackDelivered",
            {"request_id": request.request_id, "feedback_path": str(feedback)},
            f"goal-led-feedback:{request.request_id}",
        )
        self._write_state(session.thread_id, request_count + 1)
        return GoalLedOutcome(session.thread_id, self.workspace, request.request_id, match_count)

    def _run_request(self, session: AgentSession, request_count: int) -> GoalLedOutcome:
        request = self._consume_request()
        if request_count == 0 and request.opponent_id != "rank01":
            raise ValueError("the first Goal-led official request must target rank01")
        feedback, match_count = self._execute(request)
        return self._deliver_feedback(
            session, request, feedback, match_count, request_count
        )

    def start(self) -> GoalLedOutcome:
        if self._state_path.exists():
            raise ValueError("goal-led run already started")
        if not self.bootstrap_root.is_dir():
            raise ValueError("bootstrap root is unavailable")
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.bootstrap_root, self.workspace)
        (self.workspace / "research").mkdir(exist_ok=True)
        (self.workspace / "leaderboard.json").write_text(
            json.dumps(
                {"schema_version": "1.0", "opponents": list(self.public_leaderboard)},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        session = self.runtime.start(self._context(""))
        self._append(
            "GoalLedStarted",
            {"thread_id": session.thread_id, "workspace": str(self.workspace)},
            "goal-led-started",
        )
        # Persist the single thread before the first long Goal turn.  A
        # client-side timeout must never orphan research that is still stored
        # by the run-local App Server.
        self._write_state(session.thread_id, 0)
        session = self._turn(
            session,
            (
                "从 rules、公开 SDK、decision_space 和 replay_skill 出发写出 v000。"
                "完成后在 .agentbench/action.json 写第一个官方 Action；"
                "对手必须是 rank01。不要等待框架替你选择对手。"
            ),
        )
        return self._run_request(session, 0)

    def advance(self) -> GoalLedOutcome:
        thread_id, request_count = self._load_state()
        session = self.runtime.resume(thread_id, self._context(""))
        pending = self._pending_feedback()
        if pending is not None:
            request, feedback, match_count = pending
            return self._deliver_feedback(
                session, request, feedback, match_count, request_count
            )
        if self._request_path() is None:
            session = self._turn(
                session,
                "继续研究：基于已有 Experience 与公开反馈改进策略，然后写下一份 action.json。",
            )
        return self._run_request(session, request_count)

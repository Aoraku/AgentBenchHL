"""Recoverable orchestration for one Goal-led scientific checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from agentbench_hl.adapters.antwar2.policy_probe import (
    PolicyEpisodeTrace,
    compare_decisions,
    compare_policy_episode,
)
from agentbench_hl.adapters.antwar2.runtime import Opponent
from agentbench_hl.application.candidate_service import CandidateService
from agentbench_hl.application.curriculum_service import (
    CurriculumComplete,
    CurriculumMatch,
    CurriculumService,
)
from agentbench_hl.application.evaluation_service import (
    EvaluationMatrix,
    EvaluationObservation,
    EvaluationResult,
    EvaluationService,
)
from agentbench_hl.application.iteration_service import (
    IterationPlan,
    IterationProposal,
    build_iteration_prompt,
    choose_iteration_parent,
)
from agentbench_hl.application.metrics_service import MetricsService
from agentbench_hl.application.replay_service import ReplayService
from agentbench_hl.application.research_service import ResearchService
from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.experience import EvidenceWindow, ExperienceRecord
from agentbench_hl.domain.lineage import CandidateWorkspace, LineageState
from agentbench_hl.domain.metrics import IterationMetrics, combine_usage
from agentbench_hl.domain.models import Usage
from agentbench_hl.ports.agent_runtime import AgentRuntime, AgentSession, RunContext
from agentbench_hl.ports.arena import Arena, MatchCase, MatchResult
from agentbench_hl.ports.artifact_store import ArtifactStore
from agentbench_hl.ports.event_store import EventStore
from agentbench_hl.reporting.curves import build_curves

_POLICY_PROBE_SCHEMA = "antwar2-round-v2"
_QUALIFICATION_SEED_OFFSETS = (0, 10_000, 20_000)
_GOAL_ROTATION_INPUT_TOKENS = 64_000


@dataclass(frozen=True)
class RunResult:
    root: Path
    lineage: LineageState
    match_id: str
    metrics: IterationMetrics
    events: tuple[FinalizedEvent, ...]

    def event_count(self, event_type: str) -> int:
        return sum(item.event_type == event_type for item in self.events)


@dataclass(frozen=True)
class IterationAdvanceResult:
    version_id: str
    parent_id: str
    target_id: str
    selection: str
    evaluation: EvaluationResult
    metrics: IterationMetrics


@dataclass(frozen=True)
class CertificationResult:
    champion_id: str
    passed: bool
    total_cases: int
    wins: int
    incomplete_cases: tuple[str, ...]
    failed_cases: tuple[str, ...]


class RunService:
    """Own durable state; delegate only research decisions to the Goal runtime."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        bootstrap_root: str | Path,
        gamepack_root: str | Path,
        human_pool_root: str | Path,
        evaluator_root: str | Path,
        event_store: EventStore,
        certification_event_store: EventStore,
        artifact_store: ArtifactStore,
        runtime: AgentRuntime,
        arena: Arena,
        certification_arena: Arena,
        opponent_id: str,
        role: str,
        seed: int,
        human_ratings: Mapping[str, float],
        epsilon: float,
        model: str = "gpt-5.5",
        model_provider: str = "OpenAI",
        candidate_validator: Callable[[Path], object] | None = None,
        opponents: tuple[Opponent, ...] = (),
        development_roles: tuple[str, ...] = ("P0", "P1"),
        development_seeds: tuple[int, ...] = (1,),
        backend_hash: str = "",
        opponent_hashes: Mapping[str, str] | None = None,
        policy_probe: Callable[..., PolicyEpisodeTrace] | None = None,
        certification_roles: tuple[str, ...] = ("P0", "P1"),
        certification_seeds: tuple[int, ...] = (11, 12, 13),
        required_win_rate: float = 1.0,
        write_manifest: bool = True,
    ) -> None:
        self.root = Path(run_root).resolve()
        self.bootstrap_root = Path(bootstrap_root).resolve()
        self.gamepack_root = Path(gamepack_root).resolve()
        self.human_pool_root = Path(human_pool_root).resolve()
        self.evaluator_root = Path(evaluator_root).resolve()
        self.event_store = event_store
        self.certification_event_store = certification_event_store
        self.artifact_store = artifact_store
        self.runtime = runtime
        self.arena = arena
        self.certification_arena = certification_arena
        self.opponent_id = opponent_id
        self.role = role
        self.seed = seed
        self.human_ratings = {str(key): float(value) for key, value in human_ratings.items()}
        self.epsilon = float(epsilon)
        self.model = model
        self.model_provider = model_provider
        self.candidate_validator = candidate_validator
        self.opponents = opponents
        self.development_roles = development_roles
        self.development_seeds = development_seeds
        self.backend_hash = backend_hash
        self.opponent_hashes = dict(opponent_hashes or {})
        self.policy_probe = policy_probe
        self.certification_roles = certification_roles
        self.certification_seeds = certification_seeds
        if not 0.0 <= required_win_rate <= 1.0:
            raise ValueError("required_win_rate must be in [0, 1]")
        self.required_win_rate = float(required_win_rate)
        if role not in {"P0", "P1"}:
            raise ValueError("run role must be P0 or P1")
        self.root.mkdir(parents=True, exist_ok=True)
        self.research_root = self.root / "research"
        self.research_root.mkdir(exist_ok=True)
        self.candidates = CandidateService(
            run_root=self.root / "candidates",
            bootstrap_root=self.bootstrap_root,
            artifact_store=self.artifact_store,
            event_store=self.event_store,
        )
        if write_manifest:
            self._write_manifest()

    def _manifest_payload(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "origin": "from_scratch",
            "research_iteration": 0,
            "bootstrap_root": str(self.bootstrap_root),
            "gamepack_root": str(self.gamepack_root),
            "human_pool_root": str(self.human_pool_root),
            "evaluator_root": str(self.evaluator_root),
            "opponent_id": self.opponent_id,
            "role": self.role,
            "seed": self.seed,
            "human_ratings": self.human_ratings,
            "epsilon": self.epsilon,
            "model": self.model,
            "model_provider": self.model_provider,
            "development_roles": list(self.development_roles),
            "development_seeds": list(self.development_seeds),
            "backend_hash": self.backend_hash,
            "opponent_hashes": self.opponent_hashes,
        }

    def _write_manifest(self) -> None:
        path = self.root / "run-manifest.json"
        payload = self._manifest_payload()
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError("run manifest is immutable")
            return
        path.write_text(encoded, encoding="utf-8")

    @classmethod
    def resume(
        cls,
        run_root: str | Path,
        *,
        runtime: AgentRuntime,
        arena: Arena,
        certification_arena: Arena | None = None,
        human_ratings: Mapping[str, float] | None = None,
        candidate_validator: Callable[[Path], object] | None = None,
        opponents: tuple[Opponent, ...] = (),
        policy_probe: Callable[..., PolicyEpisodeTrace] | None = None,
        certification_roles: tuple[str, ...] = ("P0", "P1"),
        certification_seeds: tuple[int, ...] = (11, 12, 13),
        required_win_rate: float = 1.0,
    ) -> RunService:
        root = Path(run_root).resolve()
        manifest_path = root / "run-manifest.json"
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        ratings = human_ratings or value["human_ratings"]
        if not isinstance(ratings, Mapping):
            raise ValueError("run manifest human_ratings must be a mapping")
        from agentbench_hl.adapters.filesystem.artifact_store import (
            FilesystemArtifactStore,
        )
        from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore

        return cls(
            run_root=root,
            bootstrap_root=Path(str(value["bootstrap_root"])),
            gamepack_root=Path(str(value["gamepack_root"])),
            human_pool_root=Path(str(value["human_pool_root"])),
            evaluator_root=Path(str(value["evaluator_root"])),
            event_store=JsonlEventStore(root / "events.jsonl"),
            certification_event_store=JsonlEventStore(root / "hidden-certification/events.jsonl"),
            artifact_store=FilesystemArtifactStore(root / "candidates"),
            runtime=runtime,
            arena=arena,
            certification_arena=certification_arena or arena,
            opponent_id=str(value["opponent_id"]),
            role=str(value["role"]),
            seed=int(value["seed"]),
            human_ratings={str(key): float(item) for key, item in ratings.items()},
            epsilon=float(value["epsilon"]),
            model=str(value.get("model", "gpt-5.5")),
            model_provider=str(value.get("model_provider", "OpenAI")),
            candidate_validator=candidate_validator,
            opponents=opponents,
            policy_probe=policy_probe,
            development_roles=tuple(
                str(item) for item in value.get("development_roles", ["P0", "P1"])
            ),
            development_seeds=tuple(int(item) for item in value.get("development_seeds", [1])),
            backend_hash=str(value.get("backend_hash", "")),
            opponent_hashes={
                str(key): str(item) for key, item in dict(value.get("opponent_hashes", {})).items()
            },
            certification_roles=certification_roles,
            certification_seeds=certification_seeds,
            required_win_rate=required_win_rate,
            write_manifest=False,
        )

    def _events(self, event_type: str | None = None) -> tuple[FinalizedEvent, ...]:
        events = self.event_store.read_all()
        if event_type is None:
            return events
        return tuple(item for item in events if item.event_type == event_type)

    def _append_once(
        self, event_type: str, payload: Mapping[str, object], idempotency_key: str
    ) -> bool:
        existing = next(
            (item for item in self._events() if item.idempotency_key == idempotency_key),
            None,
        )
        if existing is not None:
            return False
        return self.event_store.append(
            FinalizedEvent.create(event_type, payload, idempotency_key=idempotency_key)
        )

    def _initialize(self) -> None:
        manifest = (self.root / "run-manifest.json").read_bytes()
        self._append_once(
            "RunInitialized",
            {
                "origin": "from_scratch",
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            },
            "run-initialized",
        )

    def _workspace(self) -> Path:
        created = self._events("CandidateCreated")
        if created:
            return Path(str(created[-1].payload["path"]))
        workspace = self.candidates.create(None, "from rules without replay")
        self._append_once(
            "CandidateCreated",
            {
                "workspace_id": workspace.workspace_id,
                "path": str(workspace.path),
                "parent_id": None,
                "reason": workspace.reason,
            },
            "candidate-created:v000",
        )
        return workspace.path

    def _context(self, workspace: Path) -> RunContext:
        charter = self.gamepack_root / "GOAL_CHARTER.md"
        charter_text = (
            charter.read_text(encoding="utf-8")
            if charter.is_file()
            else "根据冻结规则从零构造可解释策略。"
        )
        return RunContext(
            objective="从冻结规则生成 v000，并持续击败全部可运行人类池",
            initial_prompt=(
                "当前没有任何人类回放或历史策略。读取 rules.md、sdk_interface.md、"
                "decision_space.yaml、replay_skill.md，以及实现所需的公开 SDK 入口；"
                "不要逐行遍历整个 SDK，也不要重复读取本 thread 已经掌握的材料。"
                "信息足够后立即从规则推导并实现完整可解释策略到当前工作区 ai.py；"
                "不要做参数网格搜索。执行可导入检查后立即结束本 checkpoint。"
            ),
            base_instructions=charter_text,
            developer_instructions=(
                "不得读取人类源码、认证矩阵、历史参考策略或跨 run memory。"
                "只修改当前候选工作区；经验必须引用公开回放 state_id。"
            ),
            cwd=workspace,
            candidate_root=workspace,
            gamepack_root=self.gamepack_root,
            research_root=self.research_root,
            human_pool_root=self.human_pool_root,
            evaluator_root=self.evaluator_root,
            runtime_workspace_roots=(workspace, self.gamepack_root, self.research_root),
            model=self.model,
            model_provider=self.model_provider,
        )

    def _session(self, context: RunContext) -> AgentSession:
        checkpoint = self.root / "checkpoint.json"
        events = self._events()
        goal_events = tuple(
            event for event in events if event.event_type in {"GoalStarted", "GoalRotated"}
        )
        if goal_events:
            latest_goal = goal_events[-1]
            thread_id = str(latest_goal.payload["thread_id"])
            usage_events = tuple(
                event for event in events if event.event_type == "ModelUsageFinalized"
            )
            latest_input = usage_events[-1].payload.get("input_tokens") if usage_events else None
            latest_rotation_usage_count = int(latest_goal.payload.get("usage_count", 0))
            rotate_for_context = (
                isinstance(latest_input, int)
                and latest_input >= _GOAL_ROTATION_INPUT_TOKENS
                and len(usage_events) > latest_rotation_usage_count
            )
            latest_goal_index = max(
                index for index, event in enumerate(events) if event is latest_goal
            )
            zero_token_failure_count = 0
            for event in events[latest_goal_index + 1 :]:
                if (
                    event.event_type == "ModelUsageFinalized"
                    and isinstance(event.payload.get("total_tokens"), int)
                    and int(event.payload["total_tokens"]) > 0
                ):
                    zero_token_failure_count = 0
                elif (
                    event.event_type == "AgentTurnRejected"
                    and event.payload.get("billed_tokens") == 0
                ):
                    zero_token_failure_count += 1
            rotate_for_failures = zero_token_failure_count >= 3
            pending_candidate = next(
                (
                    event
                    for event in reversed(events)
                    if event.event_type == "CandidateCreated"
                    and event.payload.get("parent_id") is not None
                ),
                None,
            )
            pending_missing_proposal = bool(
                pending_candidate
                and not (
                    Path(str(pending_candidate.payload["path"])) / ".agentbench/proposal.json"
                ).is_file()
            )
            if not (rotate_for_context or rotate_for_failures or pending_missing_proposal):
                return self.runtime.resume(thread_id, context)
            session = self.runtime.start(context)
            reason = (
                "consecutive_zero_token_failures"
                if rotate_for_failures
                else (
                    "pending_candidate_missing_proposal"
                    if pending_missing_proposal
                    else "input_context_limit"
                )
            )
            self._append_once(
                "GoalRotated",
                {
                    "thread_id": session.thread_id,
                    "previous_thread_id": thread_id,
                    "ephemeral": session.ephemeral,
                    "usage_count": len(usage_events),
                    "trigger_input_tokens": latest_input,
                    "reason": reason,
                    "zero_token_failure_count": zero_token_failure_count,
                },
                (f"goal-rotated:{len(goal_events)}:{len(usage_events)}:{zero_token_failure_count}"),
            )
            checkpoint.write_text(
                json.dumps(
                    {"schema_version": "1.0", "thread_id": session.thread_id},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            return session
        session = self.runtime.start(context)
        self._append_once(
            "GoalStarted",
            {"thread_id": session.thread_id, "ephemeral": session.ephemeral},
            "goal-started",
        )
        checkpoint.write_text(
            json.dumps(
                {"schema_version": "1.0", "thread_id": session.thread_id},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return session

    def _ensure_v000(self) -> None:
        if "v000" in self.candidates.state.versions:
            return
        workspace = self._workspace()
        context = self._context(workspace)
        session = self._session(context)
        if not (workspace / "ai.py").is_file():
            self._run_agent_turn(session, context, candidate_id="v000")
        if self.candidate_validator is not None:
            for attempt in range(1, 4):
                validation = self.candidate_validator(workspace)
                status = str(getattr(validation, "status", "failed"))
                error = getattr(validation, "error", "validator returned no error")
                artifacts = getattr(validation, "artifacts", {})
                self._append_once(
                    "CandidateValidated",
                    {
                        "workspace": str(workspace),
                        "attempt": attempt,
                        "status": status,
                        "error": error,
                        "artifacts": dict(artifacts),
                    },
                    f"candidate-validated:v000:{attempt}",
                )
                if status == "complete":
                    break
                if attempt == 3:
                    raise ValueError(f"v000 failed smoke validation: {error}")
                repair_context = replace(
                    context,
                    initial_prompt=(
                        "候选公开 SDK smoke 验证失败。只根据下列诊断修复 ai.py，"
                        f"不得读取隐藏资源；修复后结束 checkpoint。诊断：{error}"
                    ),
                )
                self._run_agent_turn(session, repair_context, candidate_id="v000")
        metadata = json.loads(
            (workspace / ".agentbench/workspace.json").read_text(encoding="utf-8")
        )
        version = self.candidates.seal(str(metadata["workspace_id"]))
        if version.version_id != "v000":
            raise ValueError("from-scratch initialization did not seal as v000")
        self.candidates.promote("v000")

    def _run_agent_turn(
        self,
        session: AgentSession,
        context: RunContext,
        *,
        candidate_id: str,
    ) -> None:
        consume = getattr(self.runtime, "consume_turn_telemetry", None)
        for attempt in range(1, 4):
            started_at = time.monotonic()
            try:
                self.runtime.run_until_checkpoint(
                    session,
                    context,
                    lambda event: getattr(event, "event_type", "") == "AgentTurnCompleted",
                )
            except RuntimeError as exc:
                telemetry = tuple(consume()) if callable(consume) else ()
                has_billed_tokens = any(
                    any(
                        value.get(field) not in {None, 0}
                        for field in (
                            "input_tokens",
                            "output_tokens",
                            "reasoning_tokens",
                            "total_tokens",
                        )
                    )
                    for value in telemetry
                )
                if has_billed_tokens:
                    for value in telemetry:
                        self._record_model_usage(candidate_id, value)
                    raise
                sequence = len(self._events("AgentTurnRejected"))
                self._append_once(
                    "AgentTurnRejected",
                    {
                        "sequence": sequence,
                        "candidate_id": candidate_id,
                        "attempt": attempt,
                        "reason": str(exc),
                        "counts_as_scientific_act": False,
                        "billed_tokens": 0,
                        "wall_time_s": time.monotonic() - started_at,
                    },
                    f"agent-turn-rejected:{candidate_id}:{sequence}",
                )
                if attempt == 3:
                    raise
                adapt_effort = getattr(self.runtime, "set_turn_reasoning_effort", None)
                current_effort = str(getattr(self.runtime, "reasoning_effort", "xhigh"))
                fallback_effort = {
                    "xhigh": "high",
                    "high": "medium",
                }.get(current_effort)
                if callable(adapt_effort) and fallback_effort is not None:
                    adapt_effort(fallback_effort)
                    adaptation_sequence = len(self._events("AgentReasoningEffortAdapted"))
                    self._append_once(
                        "AgentReasoningEffortAdapted",
                        {
                            "sequence": adaptation_sequence,
                            "candidate_id": candidate_id,
                            "from_effort": current_effort,
                            "to_effort": fallback_effort,
                            "reason": "zero_token_transport_timeout",
                            "counts_as_scientific_act": False,
                        },
                        (f"agent-effort-adapted:{candidate_id}:{adaptation_sequence}"),
                    )
                continue
            telemetry = tuple(consume()) if callable(consume) else ()
            if not telemetry:
                telemetry = (
                    {
                        "input_tokens": None,
                        "cached_input_tokens": None,
                        "output_tokens": None,
                        "reasoning_tokens": None,
                        "total_tokens": None,
                        "wall_time_s": time.monotonic() - started_at,
                    },
                )
            for value in telemetry:
                self._record_model_usage(candidate_id, value)
            return

    def _record_model_usage(
        self,
        candidate_id: str,
        value: Mapping[str, object],
    ) -> None:
        sequence = len(self._events("ModelUsageFinalized"))
        usage = Usage.from_mapping(value)
        self._append_once(
            "ModelUsageFinalized",
            {
                "sequence": sequence,
                "phase": "learning",
                "candidate_id": candidate_id,
                **usage.__dict__,
            },
            f"model-usage:{sequence}",
        )

    @property
    def match_id(self) -> str:
        return f"v000-{self.opponent_id}-{self.role.lower()}-s{self.seed}"

    def _match_payload(self, result: MatchResult) -> dict[str, object]:
        return {
            "match_id": self.match_id,
            "candidate_id": "v000",
            "opponent_id": self.opponent_id,
            "role": self.role,
            "seed": self.seed,
            "status": result.status,
            "result": result.result,
            "points": result.points,
            "score_margin": result.score_margin,
            "terminal_base_hp": result.terminal_base_hp,
            "rounds": result.rounds,
            "replay_path": None if result.replay_path is None else str(result.replay_path),
            "error": result.error,
        }

    def _ensure_match(self) -> Mapping[str, object]:
        existing = self._events("MatchFinalized")
        if existing:
            return existing[-1].payload
        version = self.candidates.state.versions["v000"]
        started_at = time.monotonic()
        result = self.arena.run_case(
            MatchCase("v000", self.opponent_id, self.role, self.seed),
            version.object_path,
        )
        elapsed = time.monotonic() - started_at
        payload = self._match_payload(result)
        self._append_once("MatchFinalized", payload, f"match-finalized:{self.match_id}")
        self._append_once(
            "EvaluationUsageFinalized",
            {
                "match_id": self.match_id,
                "candidate_id": "v000",
                "wall_time_s": elapsed,
                "input_tokens": None,
                "cached_input_tokens": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
            },
            f"evaluation-usage:{self.match_id}",
        )
        return payload

    def _ensure_replay_and_experience(self, match: Mapping[str, object]) -> None:
        if match["status"] != "complete" or not match.get("replay_path"):
            return
        replay = ReplayService(self.root / "replays")
        artifacts = replay.materialize(
            match_id=self.match_id,
            replay_path=Path(str(match["replay_path"])),
        )
        self._append_once(
            "ReplayDecoded",
            {
                "match_id": self.match_id,
                "narrative": str(artifacts.narrative_md),
                "timeline": str(artifacts.timeline_jsonl),
            },
            f"replay-decoded:{self.match_id}",
        )
        timeline = [
            json.loads(line)
            for line in artifacts.timeline_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not timeline:
            raise ValueError("semantic replay has no evidence state")
        won = match.get("result") == "win"
        record = ExperienceRecord(
            experience_id="exp-v000-baseline",
            scientific_iteration=0,
            target_opponent=self.opponent_id,
            role=self.role,
            verdict="supported" if won else "refuted",
            condition="仅依据冻结规则构造首版策略",
            mechanism="规则推导的资源分配与防守进攻逻辑应形成有效基线",
            proposed_change="实现 from-scratch v000",
            expected_observation=("比赛中提交合法操作并形成可解释的基地攻防轨迹"),
            parent_id="from_scratch",
            candidate_id="v000",
            selection="promoted",
            match_ids=(self.match_id,),
            evidence_windows=(
                EvidenceWindow(
                    self.match_id,
                    str(timeline[0]["state_id"]),
                    str(timeline[-1]["state_id"]),
                ),
            ),
            measured_outcome={
                "result": match.get("result"),
                "score_margin": match.get("score_margin"),
            },
        )
        research = ResearchService(
            event_store=self.event_store,
            artifact_root=self.research_root,
        )
        research.record(record)
        rendered = research.materialize()
        self._append_once(
            "SkillMaterialized",
            {
                "playbook": str(rendered.playbook),
                "failed_hypotheses": str(rendered.failed_hypotheses),
            },
            "skill-materialized:v000",
        )

    def _metrics(self, match: Mapping[str, object]) -> IterationMetrics:
        case_id = f"{self.opponent_id}:{self.role}:{self.seed}"
        service = MetricsService(
            event_store=self.event_store,
            human_ratings=self.human_ratings,
            expected_case_ids=(case_id,),
            epsilon=self.epsilon,
        )
        service.record_match(
            candidate_id="v000",
            case_id=case_id,
            opponent_id=self.opponent_id,
            role=self.role,
            scope="smoke",
            status=str(match["status"]),
            points=None if match.get("points") is None else float(match["points"]),
            score_margin=(
                None if match.get("score_margin") is None else float(match["score_margin"])
            ),
        )
        learning_values = tuple(
            Usage.from_mapping(event.payload)
            for event in self._events("ModelUsageFinalized")
            if event.payload.get("candidate_id", "v000") == "v000"
        )
        evaluation_values = tuple(
            Usage.from_mapping(event.payload)
            for event in self._events("EvaluationUsageFinalized")
            if event.payload.get("candidate_id", "v000") == "v000"
        )
        learning_usage = combine_usage(*learning_values) if learning_values else Usage()
        evaluation_usage = combine_usage(*evaluation_values) if evaluation_values else Usage()
        metrics = service.finalize_iteration(
            "v000",
            champion_id="v000",
            learning_usage=learning_usage,
            evaluation_usage=evaluation_usage,
        )
        self._append_once(
            "SmokeMetricsFinalized",
            metrics.to_row(),
            "smoke-metrics:v000",
        )
        return metrics

    def _checkpoint(self) -> None:
        checkpoint_path = self.root / "checkpoint.json"
        value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        value["stage"] = "first_match_finalized"
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, checkpoint_path)
        self._append_once(
            "CheckpointCreated",
            {"thread_id": value["thread_id"], "stage": value["stage"]},
            "checkpoint:first-match-finalized",
        )

    def execute_until(self, checkpoint: str) -> RunResult:
        if checkpoint not in {"first_match_finalized", "checkpoint"}:
            raise ValueError(f"unsupported checkpoint: {checkpoint}")
        self._initialize()
        self._ensure_v000()
        match = self._ensure_match()
        self._ensure_replay_and_experience(match)
        metrics = self._metrics(match)
        self._checkpoint()
        return RunResult(
            root=self.root,
            lineage=self.candidates.state,
            match_id=self.match_id,
            metrics=metrics,
            events=self._events(),
        )

    def _curriculum(self) -> CurriculumService:
        matches: list[CurriculumMatch] = []
        # Curriculum progress follows the selected frontier.  The champion is
        # intentionally conservative and may lag behind exploratory candidates;
        # using it here would make a successful frontier replay invisible to
        # target selection and repeatedly revisit the same opponent.
        candidate_id = self.candidates.state.frontier_id or self.candidates.state.champion_id
        for event in self._events():
            if event.event_type == "MatchFinalized":
                payload = event.payload
            elif event.event_type in {
                "EvaluationCaseCompleted",
                "EvaluationCaseIncomplete",
            }:
                payload = event.payload
            else:
                continue
            event_candidate_id = payload.get("version_id", payload.get("candidate_id"))
            if candidate_id is None or event_candidate_id != candidate_id:
                continue
            opponent = payload.get("opponent_id")
            role = payload.get("role")
            seed = payload.get("seed")
            if opponent is None or role is None or seed is None:
                continue
            status = (
                "complete"
                if event.event_type == "EvaluationCaseCompleted"
                else str(payload.get("status", "incomplete"))
            )
            matches.append(
                CurriculumMatch(
                    str(opponent),
                    str(role),
                    int(seed),
                    status,
                    None if payload.get("result") is None else str(payload["result"]),
                )
            )
        return CurriculumService(
            opponents=self.opponents,
            roles=self.development_roles,
            seeds=self.development_seeds,
            matches=tuple(matches),
            required_win_rate=self.required_win_rate,
        )

    @staticmethod
    def _load_proposal(workspace: Path) -> IterationProposal:
        path = workspace / ".agentbench/proposal.json"
        if not path.is_file():
            raise ValueError("Goal checkpoint did not produce proposal.json")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("proposal.json must contain an object")
        return IterationProposal(
            condition=str(value.get("condition", "")),
            mechanism=str(value.get("mechanism", "")),
            intervention=str(value.get("intervention", "")),
            expected_observation=str(value.get("expected_observation", "")),
            continuation_rationale=str(value.get("continuation_rationale", "")),
        )

    def _iteration_context(
        self,
        workspace: Path,
        *,
        iteration: int,
        parent_id: str,
        target_id: str,
    ) -> RunContext:
        base = self._context(workspace)
        replay_root = self.root / "replays"
        replay_root.mkdir(exist_ok=True)
        target_evidence = tuple(
            (
                f"{event.payload['result']} | {event.payload['role']} | "
                f"seed={event.payload['seed']} | margin={event.payload['score_margin']} | "
                f"{replay_root / str(event.payload['match_id']) / 'narrative.md'}"
            )
            for event in sorted(
                self._events("EvaluationCaseCompleted"),
                key=lambda item: (
                    str(item.payload.get("role", "")),
                    int(item.payload.get("seed", 0)),
                ),
            )
            if event.payload.get("version_id") == parent_id
            and event.payload.get("opponent_id") == target_id
        )
        return replace(
            base,
            objective="持续从公开回放学习，直到冻结认证矩阵全部获胜",
            initial_prompt=build_iteration_prompt(
                iteration=iteration,
                parent_id=parent_id,
                target_id=target_id,
                gamepack_root=self.gamepack_root,
                replay_root=replay_root,
                research_root=self.research_root,
                evidence_entries=target_evidence,
            ),
            runtime_workspace_roots=(
                workspace,
                self.gamepack_root,
                self.research_root,
                replay_root,
            ),
        )

    def _validate_iteration_workspace(
        self,
        *,
        workspace: Path,
        context: RunContext,
        session: AgentSession,
        candidate_id: str,
    ) -> None:
        if self.candidate_validator is None:
            return
        for attempt in range(1, 4):
            validation = self.candidate_validator(workspace)
            status = str(getattr(validation, "status", "failed"))
            error = getattr(validation, "error", "validator returned no error")
            artifacts = getattr(validation, "artifacts", {})
            self._append_once(
                "CandidateValidated",
                {
                    "candidate_id": candidate_id,
                    "workspace": str(workspace),
                    "attempt": attempt,
                    "status": status,
                    "error": error,
                    "artifacts": dict(artifacts),
                },
                f"candidate-validated:{candidate_id}:{attempt}",
            )
            if status == "complete":
                return
            if attempt == 3:
                raise ValueError(f"{candidate_id} failed smoke validation: {error}")
            repair_context = replace(
                context,
                initial_prompt=(
                    "候选公开 SDK smoke 验证失败。根据诊断修复当前 ai.py，"
                    "保留已有 proposal.json 的科研假设；不得读取隐藏资源。"
                    f"诊断：{error}"
                ),
            )
            self._run_agent_turn(
                session,
                repair_context,
                candidate_id=candidate_id,
            )

    def _record_iteration_experience(
        self,
        *,
        scientific_iteration: int,
        version_id: str,
        parent_id: str,
        target_id: str,
        proposal: IterationProposal,
        evaluation: EvaluationResult,
        selection: str,
    ) -> None:
        replay_service = ReplayService(self.root / "replays")
        research = ResearchService(
            event_store=self.event_store,
            artifact_root=self.research_root,
        )
        for role in self.development_roles:
            observations = tuple(
                item
                for item in evaluation.observations
                if item.opponent_id == target_id
                and item.role == role
                and item.status == "complete"
                and item.replay_path is not None
                and item.match_id is not None
            )
            match_ids: list[str] = []
            windows: list[EvidenceWindow] = []
            for observation in observations:
                assert observation.replay_path is not None
                assert observation.match_id is not None
                artifacts = replay_service.materialize(
                    match_id=observation.match_id,
                    replay_path=observation.replay_path,
                )
                self._append_once(
                    "ReplayDecoded",
                    {
                        "candidate_id": version_id,
                        "match_id": observation.match_id,
                        "narrative": str(artifacts.narrative_md),
                        "timeline": str(artifacts.timeline_jsonl),
                    },
                    f"replay-decoded:{observation.match_id}",
                )
                timeline = [
                    json.loads(line)
                    for line in artifacts.timeline_jsonl.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if timeline:
                    match_ids.append(observation.match_id)
                    windows.append(
                        EvidenceWindow(
                            observation.match_id,
                            str(timeline[0]["state_id"]),
                            str(timeline[-1]["state_id"]),
                        )
                    )
            if not windows:
                continue
            points = [float(item.points) for item in observations if item.points is not None]
            role_solved = bool(points) and all(point == 1.0 for point in points)
            verdict = (
                "supported"
                if role_solved
                else "mixed"
                if any(point > 0 for point in points)
                else "refuted"
            )
            research.record(
                ExperienceRecord(
                    experience_id=f"exp-{version_id}-{target_id}-{role.lower()}",
                    scientific_iteration=scientific_iteration,
                    target_opponent=target_id,
                    role=role,
                    verdict=verdict,
                    condition=proposal.condition,
                    mechanism=proposal.mechanism,
                    proposed_change=proposal.intervention,
                    expected_observation=proposal.expected_observation,
                    parent_id=parent_id,
                    candidate_id=version_id,
                    selection=selection,
                    match_ids=tuple(match_ids),
                    evidence_windows=tuple(windows),
                    measured_outcome={
                        "role_solved": role_solved,
                        "regressions_passed": evaluation.regressions_passed,
                        "points": points,
                        "score_margins": [item.score_margin for item in observations],
                    },
                )
            )
        research.materialize()

    def _iteration_metrics(
        self,
        version_id: str,
        parent_id: str,
        observations: tuple[EvaluationObservation, ...],
        *,
        research_iteration: int | None = None,
        parent_observations: tuple[EvaluationObservation, ...] = (),
        evaluation_wall_time_s: float,
    ) -> IterationMetrics:
        service = MetricsService(
            event_store=self.event_store,
            human_ratings=self.human_ratings,
            expected_case_ids=tuple(item.case_id for item in observations),
            epsilon=self.epsilon,
            probe_schema=_POLICY_PROBE_SCHEMA,
        )
        if self.policy_probe is not None and version_id != parent_id:
            parent_root = self.candidates.state.versions[parent_id].object_path
            candidate_root = self.candidates.state.versions[version_id].object_path
            parent_by_case = {item.case_id: item for item in parent_observations}
            # Policy probes are expensive and this method is intentionally
            # resumable.  A crash/interrupt after the probe event is appended
            # must not cause the same case to be probed again on resume.
            measured_cases = {
                str(event.payload.get("case_id"))
                for event in self.event_store.read_all()
                if event.event_type == "PolicyEpisodeMeasured"
                and str(event.payload.get("candidate_id")) == version_id
            }
            for item in observations:
                if item.status != "complete" or item.match_id is None or item.replay_path is None:
                    continue
                if item.case_id in measured_cases:
                    continue
                parent_trace = self.policy_probe(
                    parent_root,
                    item.replay_path,
                    match_id=item.match_id,
                    role=item.role,
                )
                candidate_trace = self.policy_probe(
                    candidate_root,
                    item.replay_path,
                    match_id=item.match_id,
                    role=item.role,
                )
                parent_observation = parent_by_case.get(item.case_id)
                if (
                    parent_observation is not None
                    and parent_observation.status == "complete"
                    and parent_observation.match_id is not None
                    and parent_observation.replay_path is not None
                ):
                    parent_own_trace = self.policy_probe(
                        parent_root,
                        parent_observation.replay_path,
                        match_id=parent_observation.match_id,
                        role=parent_observation.role,
                    )
                    service.record_occupancy_trace(
                        candidate_id=version_id,
                        case_id=item.case_id,
                        parent_state_ids=tuple(
                            decision.occupancy_id for decision in parent_own_trace.decisions
                        ),
                        candidate_state_ids=tuple(
                            decision.occupancy_id for decision in candidate_trace.decisions
                        ),
                    )
                samples = compare_policy_episode(parent_trace, candidate_trace)
                service.record_decision_samples(version_id, samples)
                comparison = compare_decisions(samples, epsilon=self.epsilon)
                self._append_once(
                    "PolicyEpisodeMeasured",
                    {
                        "candidate_id": version_id,
                        "parent_id": parent_id,
                        "match_id": item.match_id,
                        "case_id": item.case_id,
                        "role": item.role,
                        "seed": item.seed,
                        "epsilon": self.epsilon,
                        "state_ids": [sample.state_id for sample in samples],
                        "local_policy_kl_trace": [value.kl_nats for value in comparison.trace],
                        "mean_local_policy_kl_nats": comparison.mean_kl_nats,
                        "trajectory_kl_nats": sum(value.kl_nats for value in comparison.trace),
                        "action_disagreement_rate": comparison.disagreement_rate,
                        "probe_schema": _POLICY_PROBE_SCHEMA,
                    },
                    (f"policy-episode:{_POLICY_PROBE_SCHEMA}:{version_id}:{item.case_id}"),
                )
        for item in observations:
            service.record_match(
                candidate_id=version_id,
                case_id=item.case_id,
                opponent_id=item.opponent_id,
                role=item.role,
                status=item.status,
                points=item.points,
                score_margin=item.score_margin,
            )
        learning = tuple(
            Usage.from_mapping(event.payload)
            for event in self._events("ModelUsageFinalized")
            if event.payload.get("candidate_id") == version_id
        )
        evaluation_usage = Usage(wall_time_s=evaluation_wall_time_s)
        metrics = service.finalize_iteration(
            version_id,
            champion_id=self.candidates.state.champion_id or version_id,
            research_iteration=research_iteration,
            learning_usage=combine_usage(*learning) if learning else Usage(),
            evaluation_usage=evaluation_usage,
        )
        self._append_once(
            "IterationMetricsFinalized",
            metrics.to_row(),
            f"iteration-metrics:{version_id}",
        )
        return metrics

    def _evaluation_service(self) -> EvaluationService:
        return EvaluationService(
            arena=self.arena,
            event_store=self.event_store,
            backend_hash=self.backend_hash,
            candidate_root=lambda version_id: (
                self.candidates.state.versions[version_id].object_path
            ),
            candidate_hash=lambda version_id: (
                self.candidates.state.versions[version_id].content_hash
            ),
            opponent_hashes=self.opponent_hashes,
        )

    def _qualification_seeds(self) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                seed + offset
                for offset in _QUALIFICATION_SEED_OFFSETS
                for seed in self.development_seeds
            )
        )

    def _measurement_opponent_ids(self) -> tuple[str, ...]:
        values = tuple(item.opponent_id for item in self.opponents if item.runnable)
        if not values:
            raise ValueError("fixed measurement panel has no runnable opponent")
        return values

    def _ensure_baseline_metrics(self) -> None:
        has_metrics = any(
            event.payload.get("candidate_id") == "v000"
            for event in self._events("IterationMetricsFinalized")
        )
        runnable = tuple(item for item in self.opponents if item.runnable)
        if not runnable:
            raise ValueError("baseline requires a runnable opponent")
        target_id = max(runnable, key=lambda item: item.rank).opponent_id
        started_at = time.monotonic()
        evaluation = self._evaluation_service().evaluate_version(
            "v000",
            EvaluationMatrix(
                (target_id,),
                self.development_roles,
                self.development_seeds,
            ),
            target_id=target_id,
            locked_regression_ids=(),
        )
        if not has_metrics:
            self._iteration_metrics(
                "v000",
                "v000",
                evaluation.observations,
                evaluation_wall_time_s=time.monotonic() - started_at,
            )
        self._record_iteration_experience(
            scientific_iteration=0,
            version_id="v000",
            parent_id="from_scratch",
            target_id=target_id,
            proposal=IterationProposal(
                condition="首版策略在最弱可运行对手的双角色开发矩阵中接受检验",
                mechanism="规则推导的首版策略尚未吸收目标对手的公开实战轨迹",
                intervention="将目标对手的正式回放语义化并作为下一轮代码干预依据",
                expected_observation="下一版本针对败局中的具体 state_id 改善胜率和分差",
                continuation_rationale="保留首版为 Champion，并从正式胜负证据继续学习",
            ),
            evaluation=evaluation,
            selection="promoted",
        )

    def _rebuild_curves(self) -> None:
        rows = tuple(
            IterationMetrics.from_row(event.payload)
            for event in self._events("IterationMetricsFinalized")
        )
        if rows:
            build_curves(rows, self.root / "reports/curves")

    def certify_champion(self) -> CertificationResult:
        curriculum = self._curriculum().status()
        if not curriculum.all_runnable_solved:
            raise ValueError("certification requires all development cases to be solved")
        champion_id = self.candidates.state.champion_id
        if champion_id is None:
            raise ValueError("certification requires a Champion")
        opponent_ids = self._measurement_opponent_ids()
        evaluation = EvaluationService(
            arena=self.certification_arena,
            event_store=self.certification_event_store,
            backend_hash=self.backend_hash,
            candidate_root=lambda version_id: (
                self.candidates.state.versions[version_id].object_path
            ),
            candidate_hash=lambda version_id: (
                self.candidates.state.versions[version_id].content_hash
            ),
            opponent_hashes=self.opponent_hashes,
        ).evaluate_version(
            champion_id,
            EvaluationMatrix(
                opponent_ids,
                self.certification_roles,
                self.certification_seeds,
            ),
            target_id=opponent_ids[-1],
            locked_regression_ids=(),
        )
        incomplete = tuple(
            item.case_id for item in evaluation.observations if item.status != "complete"
        )
        failed = tuple(
            item.case_id
            for item in evaluation.observations
            if item.status == "complete" and item.result != "win"
        )
        wins = sum(
            item.status == "complete" and item.result == "win" for item in evaluation.observations
        )
        passed = not incomplete and not failed
        result = CertificationResult(
            champion_id,
            passed,
            len(evaluation.observations),
            wins,
            incomplete,
            failed,
        )
        self._append_once(
            "CertificationFinalized",
            {
                "champion_id": champion_id,
                "passed": passed,
                "total_cases": result.total_cases,
                "wins": wins,
                "incomplete_count": len(incomplete),
                "failed_count": len(failed),
            },
            f"certification:{champion_id}",
        )
        if passed:
            self._append_once(
                "RunCompleted",
                {
                    "champion_id": champion_id,
                    "certification_cases": result.total_cases,
                    "certification_wins": wins,
                },
                "run-completed",
            )
        return result

    def _iteration_workspace(
        self,
        *,
        iteration: int,
        version_id: str,
        parent_id: str,
        target_id: str,
    ) -> CandidateWorkspace:
        key = f"candidate-created:{version_id}"
        existing = next(
            (event for event in self._events("CandidateCreated") if event.idempotency_key == key),
            None,
        )
        if existing is not None:
            payload = existing.payload
            path = Path(str(payload["path"])).resolve()
            workspace_id = str(payload["workspace_id"])
            metadata = path / ".agentbench/workspace.json"
            if not metadata.is_file():
                raise ValueError(f"unfinished workspace is unavailable: {path}")
            return CandidateWorkspace(
                workspace_id,
                path,
                None if payload.get("parent_id") is None else str(payload["parent_id"]),
                str(payload["reason"]),
            )
        workspace = self.candidates.create(
            parent_id,
            reason=f"iteration {iteration} against {target_id}",
        )
        self._append_once(
            "CandidateCreated",
            {
                "workspace_id": workspace.workspace_id,
                "path": str(workspace.path),
                "parent_id": parent_id,
                "reason": workspace.reason,
            },
            key,
        )
        return workspace

    def _iteration_plan(self) -> IterationPlan:
        terminal = {
            str(event.payload["candidate_id"])
            for event in self._events("IterationMetricsFinalized")
        }
        terminal.update(
            str(event.payload["version_id"]) for event in self._events("IterationAborted")
        )
        pending = tuple(
            IterationPlan.from_payload(event.payload)
            for event in self._events("IterationPlanned")
            if event.payload.get("version_id") not in terminal
        )
        if len(pending) > 1:
            raise ValueError("run contains multiple unfinished iteration plans")
        if pending:
            # A crash/recovery can leave a plan created from an incomplete
            # curriculum ledger.  Revalidate it against preserved match
            # observations before allowing a Goal call to continue; otherwise
            # the agent may spend tokens on the wrong opponent or regressions.
            curriculum = self._curriculum()
            status = curriculum.status()
            try:
                expected_target_id = curriculum.default_target().opponent_id
                expected_locked = status.locked_regression_ids
            except CurriculumComplete:
                expected_target_id = pending[0].target_id
                expected_locked = pending[0].locked_regression_ids
            if pending[0].target_id == expected_target_id and tuple(
                pending[0].locked_regression_ids
            ) == tuple(expected_locked):
                return pending[0]
            self._append_once(
                "IterationAborted",
                {
                    "iteration": pending[0].iteration,
                    "version_id": pending[0].version_id,
                    "parent_id": pending[0].parent_id,
                    "target_id": pending[0].target_id,
                    "reason": "stale plan invalidated after curriculum recovery",
                },
                f"iteration-aborted:{pending[0].version_id}:stale-curriculum",
            )
        curriculum = self._curriculum()
        status = curriculum.status()
        try:
            target = curriculum.default_target()
            locked = status.locked_regression_ids
        except CurriculumComplete:
            champion_id = self.candidates.state.champion_id
            failed_certification = any(
                event.payload.get("champion_id") == champion_id
                and event.payload.get("passed") is False
                for event in self._events("CertificationFinalized")
            )
            if not failed_certification:
                raise
            runnable = tuple(item for item in self.opponents if item.runnable)
            target = min(runnable, key=lambda item: item.rank)
            locked = tuple(
                item for item in status.locked_regression_ids if item != target.opponent_id
            )
        scientific_iteration = (
            max(
                (
                    int(event.payload["research_iteration"])
                    for event in self._events("IterationMetricsFinalized")
                ),
                default=0,
            )
            + 1
        )
        version_sequence = len(self.candidates.state.versions)
        while f"v{version_sequence:03d}" in terminal:
            version_sequence += 1
        plan = IterationPlan(
            iteration=scientific_iteration,
            version_id=f"v{version_sequence:03d}",
            parent_id=choose_iteration_parent(self.candidates.state),
            target_id=target.opponent_id,
            locked_regression_ids=locked,
        )
        self._append_once(
            "IterationPlanned",
            plan.to_payload(),
            f"iteration-planned:{plan.version_id}",
        )
        return plan

    def advance_one_iteration(self) -> IterationAdvanceResult:
        if "v000" not in self.candidates.state.versions:
            raise ValueError("v000 checkpoint must exist before improvement iterations")
        if not self.opponents or not self.backend_hash:
            raise ValueError("long-run evaluation resources are not configured")
        self._ensure_baseline_metrics()
        while True:
            plan = self._iteration_plan()
            workspace = self._iteration_workspace(
                iteration=plan.iteration,
                version_id=plan.version_id,
                parent_id=plan.parent_id,
                target_id=plan.target_id,
            )
            if plan.version_id not in self.candidates.state.versions:
                break
            proposal_path = workspace.path / ".agentbench/proposal.json"
            version = self.candidates.state.versions[plan.version_id]
            parent = self.candidates.state.versions[plan.parent_id]
            rejection_reason = None
            if not proposal_path.is_file():
                rejection_reason = "sealed checkpoint is missing proposal.json"
            elif version.content_hash == parent.content_hash:
                rejection_reason = "sealed checkpoint does not change candidate code"
            if rejection_reason is None:
                break
            self._append_once(
                "IterationAborted",
                {
                    "iteration": plan.iteration,
                    "version_id": plan.version_id,
                    "parent_id": plan.parent_id,
                    "target_id": plan.target_id,
                    "reason": rejection_reason,
                    "counts_as_scientific_act": False,
                },
                f"iteration-aborted:{plan.version_id}",
            )
        if plan.version_id not in self.candidates.state.versions:
            context = self._iteration_context(
                workspace.path,
                iteration=plan.iteration,
                parent_id=plan.parent_id,
                target_id=plan.target_id,
            )
            session = self._session(context)
            proposal_path = workspace.path / ".agentbench/proposal.json"
            if not (workspace.path / "ai.py").is_file() or not proposal_path.is_file():
                self._run_agent_turn(session, context, candidate_id=plan.version_id)
            self._load_proposal(workspace.path)
            self._validate_iteration_workspace(
                workspace=workspace.path,
                context=context,
                session=session,
                candidate_id=plan.version_id,
            )
            candidate_hash, _, _ = self.artifact_store.materialize(workspace.path)
            parent_hash = self.candidates.state.versions[plan.parent_id].content_hash
            if candidate_hash == parent_hash:
                raise ValueError(
                    f"{plan.version_id} does not change candidate code from {plan.parent_id}"
                )
            version = self.candidates.seal(workspace.workspace_id)
            if version.version_id != plan.version_id:
                raise ValueError("sealed candidate does not match unfinished iteration plan")
        else:
            version = self.candidates.state.versions[plan.version_id]
        proposal = self._load_proposal(workspace.path)
        matrix_ids = tuple(dict.fromkeys((plan.target_id, *plan.locked_regression_ids)))
        evaluator = self._evaluation_service()
        started_at = time.monotonic()
        evaluation = evaluator.evaluate_version(
            version.version_id,
            EvaluationMatrix(
                matrix_ids,
                self.development_roles,
                self.development_seeds,
            ),
            target_id=plan.target_id,
            locked_regression_ids=plan.locked_regression_ids,
        )
        if evaluation.promotable:
            confirmation = evaluator.evaluate_version(
                version.version_id,
                EvaluationMatrix(
                    (plan.target_id,),
                    self.development_roles,
                    self._qualification_seeds(),
                ),
                target_id=plan.target_id,
                locked_regression_ids=(),
            )
            observations = {
                item.case_id: item
                for item in (*evaluation.observations, *confirmation.observations)
            }
            evaluation = EvaluationResult(
                observations=tuple(observations.values()),
                target_solved=confirmation.target_solved,
                regressions_passed=evaluation.regressions_passed,
                promotable=(confirmation.target_solved and evaluation.regressions_passed),
                frontier_eligible=(evaluation.frontier_eligible and confirmation.frontier_eligible),
                retry_case_ids=tuple(
                    dict.fromkeys((*evaluation.retry_case_ids, *confirmation.retry_case_ids))
                ),
            )
        if evaluation.promotable:
            self.candidates.promote(version.version_id)
            selection = "promoted"
        elif evaluation.frontier_eligible:
            if self.candidates.state.frontier_id != version.version_id:
                self.candidates.choose_frontier(
                    version.version_id,
                    rationale=proposal.continuation_rationale,
                )
            selection = "frontier"
        else:
            selection = "incomplete"
        self._record_iteration_experience(
            scientific_iteration=plan.iteration,
            version_id=version.version_id,
            parent_id=plan.parent_id,
            target_id=plan.target_id,
            proposal=proposal,
            evaluation=evaluation,
            selection=selection,
        )
        measurement = evaluation
        # An incomplete case is deliberately not a loss, but it must not
        # block settlement indefinitely.  Re-running the same timed-out
        # target for the parent solely to compute a comparison would keep the
        # iteration in the evaluation phase forever.  Settle the candidate
        # with no parent observations in this case; the next iteration will
        # retry the incomplete target, while complete cases still receive the
        # normal candidate-vs-parent measurement.
        if evaluation.retry_case_ids:
            parent_observations: tuple[EvaluationObservation, ...] = ()
        else:
            parent_measurement = evaluator.evaluate_version(
                plan.parent_id,
                EvaluationMatrix(
                    matrix_ids,
                    self.development_roles,
                    self.development_seeds,
                ),
                target_id=plan.target_id,
                locked_regression_ids=plan.locked_regression_ids,
            )
            parent_observations = parent_measurement.observations
        evaluation_wall_time = time.monotonic() - started_at
        metrics = self._iteration_metrics(
            version.version_id,
            plan.parent_id,
            measurement.observations,
            research_iteration=plan.iteration,
            parent_observations=parent_observations,
            evaluation_wall_time_s=evaluation_wall_time,
        )
        self._rebuild_curves()
        return IterationAdvanceResult(
            version.version_id,
            plan.parent_id,
            plan.target_id,
            selection,
            evaluation,
            metrics,
        )


def advance_run(
    run: RunService,
    *,
    acts: int,
) -> tuple[IterationAdvanceResult, ...]:
    if isinstance(acts, bool) or acts < 1:
        raise ValueError("acts must be a positive integer")
    return tuple(run.advance_one_iteration() for _ in range(acts))

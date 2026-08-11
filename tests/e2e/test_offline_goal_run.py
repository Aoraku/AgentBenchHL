from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agentbench_hl.adapters.antwar2.arena import match_result_from_replay
from agentbench_hl.adapters.antwar2.policy_probe import (
    PolicyDecision,
    PolicyEpisodeTrace,
)
from agentbench_hl.adapters.antwar2.replay import decode_replay
from agentbench_hl.adapters.antwar2.runtime import Opponent
from agentbench_hl.adapters.antwar2.smoke import ValidationResult
from agentbench_hl.adapters.filesystem.artifact_store import FilesystemArtifactStore
from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.application.run_service import RunService, advance_run
from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.ports.agent_runtime import AgentSession, RunContext
from agentbench_hl.ports.arena import MatchCase, MatchResult

FIXTURE = Path(__file__).parents[1] / "golden/antwar2_replays/fixture.json"


class GoalThatWritesBaseline:
    def __init__(self) -> None:
        self.start_count = 0
        self.resume_count = 0
        self.turn_count = 0

    def start(self, run_context: RunContext) -> AgentSession:
        self.start_count += 1
        return AgentSession("offline-thread", "active", ephemeral=False)

    def resume(self, session_id: str, run_context: RunContext) -> AgentSession:
        self.resume_count += 1
        return AgentSession(session_id, "active", ephemeral=False)

    def run_until_checkpoint(
        self, session: AgentSession, run_context: RunContext, checkpoint_predicate
    ) -> AgentSession:
        self.turn_count += 1
        policy = run_context.candidate_root / "ai.py"
        if not policy.exists():
            policy.write_text(
                "class AI:\n    def choose_operations(self, state):\n        return []\n",
                encoding="utf-8",
            )
        else:
            with policy.open("a", encoding="utf-8") as stream:
                stream.write(f"\n# grounded revision {self.turn_count}\n")
            (run_context.candidate_root / ".agentbench/proposal.json").write_text(
                json.dumps(
                    {
                        "condition": "基地首次受伤前仍有空闲资源",
                        "mechanism": "首轮压力形成太晚",
                        "intervention": "提前开局状态机",
                        "expected_observation": "首塔提前且首次受伤推迟",
                        "continuation_rationale": "继续验证升级衔接而非回退",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return session

    def pause(self, session: AgentSession) -> AgentSession:
        session.goal_status = "paused"
        return session

    def consume_turn_telemetry(self):
        return (
            {
                "input_tokens": 10,
                "cached_input_tokens": 3,
                "output_tokens": 5,
                "reasoning_tokens": 2,
                "total_tokens": 15,
                "wall_time_s": 1.25,
            },
        )


class FixtureArena:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.call_count = 0

    def run_case(self, case: MatchCase, candidate_root: Path):
        self.call_count += 1
        replay = self.root / "raw" / "replay.json"
        replay.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIXTURE, replay)
        return match_result_from_replay(
            replay,
            candidate_id=case.candidate_id,
            opponent_id=case.opponent_id,
            role=case.role,
            seed=case.seed,
        )


def build_offline_run(
    root: Path,
    arena: FixtureArena,
    goal: GoalThatWritesBaseline,
    *,
    candidate_validator=None,
    policy_probe=None,
):
    bootstrap = root / "bootstrap"
    bootstrap.mkdir()
    for name in ("main.py", "common.py", "protocol.py"):
        (bootstrap / name).write_text("# public support\n", encoding="utf-8")
    (bootstrap / "SDK").mkdir()
    (bootstrap / "SDK/__init__.py").write_text("", encoding="utf-8")
    gamepack = root / "gamepack"
    gamepack.mkdir()
    (gamepack / "rules.md").write_text("frozen rules", encoding="utf-8")
    hidden = root / "hidden-humans"
    hidden.mkdir()
    evaluator = root / "hidden-certification"
    evaluator.mkdir()
    opponent = Opponent(
        opponent_id="rank20",
        rank=20,
        username="fixture-human",
        score=1000,
        archive=root / "rank20.zip",
        archive_sha256="2" * 64,
        package_root=hidden,
        runnable=True,
        entry_command=("python", "main.py"),
        exclusion_diagnostic=None,
    )
    return RunService(
        run_root=root / "run",
        bootstrap_root=bootstrap,
        gamepack_root=gamepack,
        human_pool_root=hidden,
        evaluator_root=evaluator,
        event_store=JsonlEventStore(root / "run/events.jsonl"),
        certification_event_store=JsonlEventStore(root / "run/hidden-certification/events.jsonl"),
        artifact_store=FilesystemArtifactStore(root / "run/candidates"),
        runtime=goal,
        arena=arena,
        certification_arena=arena,
        opponent_id="rank20",
        role="P0",
        seed=1,
        human_ratings={"rank20": 1000.0},
        epsilon=0.01,
        opponents=(opponent,),
        development_roles=("P0", "P1"),
        development_seeds=(1,),
        backend_hash="backend-fixture",
        opponent_hashes={"rank20": opponent.archive_sha256},
        candidate_validator=candidate_validator,
        policy_probe=policy_probe,
        replay_decoder=decode_replay,
        certification_roles=("P0", "P1"),
        certification_seeds=(11,),
    )


def test_fake_goal_creates_v000_match_replay_experience_and_metrics(
    tmp_path: Path,
) -> None:
    arena = FixtureArena(tmp_path)
    goal = GoalThatWritesBaseline()
    run = build_offline_run(tmp_path, arena, goal)

    result = run.execute_until("first_match_finalized")

    assert result.lineage.champion_id == "v000"
    assert result.event_count("CandidateSealed") == 1
    assert result.event_count("MatchFinalized") == 1
    assert (result.root / "replays" / result.match_id / "narrative.md").is_file()
    assert (result.root / "research/PLAYBOOK.md").is_file()
    assert result.metrics.research_iteration == 0
    assert result.metrics.win_rate in {0.0, 1.0}
    assert result.metrics.learning_usage.total_tokens == 15
    assert result.metrics.learning_usage.wall_time_s == 1.25
    assert result.metrics.evaluation_usage.wall_time_s is not None
    assert goal.start_count == 1
    assert arena.call_count == 1
    assert result.event_count("SmokeMetricsFinalized") == 1
    assert result.event_count("IterationMetricsFinalized") == 0


def test_resume_reuses_finalized_match(tmp_path: Path) -> None:
    arena = FixtureArena(tmp_path)
    goal = GoalThatWritesBaseline()
    run = build_offline_run(tmp_path, arena, goal)
    first = run.execute_until("first_match_finalized")
    assert first.event_count("MatchFinalized") == 1

    resumed_arena = FixtureArena(tmp_path)
    resumed = RunService.resume(
        run.root,
        runtime=goal,
        arena=resumed_arena,
        human_ratings={"rank20": 1000.0},
        replay_decoder=decode_replay,
    )
    result = resumed.execute_until("checkpoint")

    assert resumed_arena.call_count == 0
    assert result.event_count("MatchFinalized") == 1
    assert json.loads((run.root / "checkpoint.json").read_text())["thread_id"]


def test_resumed_run_can_advance_the_same_goal_and_long_run_curriculum(
    tmp_path: Path,
) -> None:
    goal = GoalThatWritesBaseline()
    run = build_offline_run(tmp_path, FixtureArena(tmp_path), goal)
    run.execute_until("first_match_finalized")

    resumed = RunService.resume(
        run.root,
        runtime=goal,
        arena=FixtureArena(tmp_path),
        opponents=run.opponents,
        candidate_validator=run.candidate_validator,
        replay_decoder=decode_replay,
    )
    result = resumed.advance_one_iteration()

    assert result.version_id == "v001"
    assert result.parent_id == "v000"
    assert goal.resume_count >= 1


def test_large_goal_context_rotates_to_a_fresh_thread_before_next_iteration(
    tmp_path: Path,
) -> None:
    goal = GoalThatWritesBaseline()
    run = build_offline_run(tmp_path, FixtureArena(tmp_path), goal)
    run.execute_until("first_match_finalized")
    run.event_store.append(
        FinalizedEvent.create(
            "ModelUsageFinalized",
            {
                "sequence": 99,
                "phase": "learning",
                "candidate_id": "v000",
                "input_tokens": 70_000,
                "cached_input_tokens": 65_000,
                "output_tokens": 100,
                "reasoning_tokens": 50,
                "total_tokens": 70_150,
                "wall_time_s": 1.0,
            },
            idempotency_key="fixture-high-context",
        )
    )

    run.advance_one_iteration()

    rotations = run._events("GoalRotated")
    assert goal.start_count == 2
    assert len(rotations) == 1
    assert rotations[0].payload["trigger_input_tokens"] == 70_000
    assert rotations[0].payload["previous_thread_id"] == "offline-thread"


def test_three_zero_token_turn_failures_rotate_to_a_fresh_goal_thread(
    tmp_path: Path,
) -> None:
    goal = GoalThatWritesBaseline()
    run = build_offline_run(tmp_path, FixtureArena(tmp_path), goal)
    run.execute_until("first_match_finalized")
    for sequence in range(3):
        run.event_store.append(
            FinalizedEvent.create(
                "AgentTurnRejected",
                {
                    "sequence": sequence,
                    "candidate_id": "v001",
                    "attempt": sequence + 1,
                    "reason": "stream disconnected before completion",
                    "counts_as_scientific_act": False,
                    "billed_tokens": 0,
                    "wall_time_s": 25.0,
                },
                idempotency_key=f"fixture-zero-token-rejection:{sequence}",
            )
        )

    run.advance_one_iteration()

    rotations = run._events("GoalRotated")
    assert goal.start_count == 2
    assert len(rotations) == 1
    assert rotations[0].payload["reason"] == "consecutive_zero_token_failures"
    assert rotations[0].payload["zero_token_failure_count"] == 3


def test_advance_run_executes_exactly_the_requested_scientific_acts(
    tmp_path: Path,
) -> None:
    run = build_offline_run(
        tmp_path,
        FixtureArena(tmp_path),
        GoalThatWritesBaseline(),
    )
    run.execute_until("first_match_finalized")

    results = advance_run(run, acts=2)

    assert tuple(item.version_id for item in results) == ("v001", "v002")
    assert len(run.candidates.state.versions) == 3


def test_resume_reuses_a_generated_unsealed_workspace_without_another_paid_turn(
    tmp_path: Path,
) -> None:
    class CrashOnceAfterGeneration:
        def __init__(self) -> None:
            self.crashed = False

        def __call__(self, workspace: Path) -> ValidationResult:
            if (workspace / ".agentbench/proposal.json").is_file() and not self.crashed:
                self.crashed = True
                raise RuntimeError("fixture process interruption")
            return ValidationResult("complete", None, {"roles": ["P0", "P1"]})

    goal = GoalThatWritesBaseline()
    run = build_offline_run(
        tmp_path,
        FixtureArena(tmp_path),
        goal,
        candidate_validator=CrashOnceAfterGeneration(),
    )
    run.execute_until("first_match_finalized")

    with pytest.raises(RuntimeError, match="fixture process interruption"):
        run.advance_one_iteration()
    turns_after_generation = goal.turn_count
    result = run.advance_one_iteration()

    assert result.version_id == "v001"
    assert goal.turn_count == turns_after_generation
    assert len(run._events("CandidateCreated")) == 2


def test_resume_finishes_a_sealed_candidate_before_generating_another_version(
    tmp_path: Path,
) -> None:
    class CrashOnceDuringV001Evaluation(FixtureArena):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.crashed = False

        def run_case(self, case: MatchCase, candidate_root: Path):
            if case.candidate_id == "v001" and not self.crashed:
                self.crashed = True
                raise RuntimeError("fixture evaluator interruption")
            return super().run_case(case, candidate_root)

    goal = GoalThatWritesBaseline()
    arena = CrashOnceDuringV001Evaluation(tmp_path)
    run = build_offline_run(tmp_path, arena, goal)
    run.execute_until("first_match_finalized")

    with pytest.raises(RuntimeError, match="fixture evaluator interruption"):
        run.advance_one_iteration()
    turns_after_seal = goal.turn_count
    result = run.advance_one_iteration()

    assert result.version_id == "v001"
    assert goal.turn_count == turns_after_seal
    assert set(run.candidates.state.versions) == {"v000", "v001"}


def test_improvement_without_proposal_is_not_sealed(tmp_path: Path) -> None:
    class GoalThatOmitsImprovementProposal(GoalThatWritesBaseline):
        def run_until_checkpoint(
            self,
            session: AgentSession,
            run_context: RunContext,
            checkpoint_predicate,
        ) -> AgentSession:
            policy = run_context.candidate_root / "ai.py"
            if not policy.exists():
                return super().run_until_checkpoint(session, run_context, checkpoint_predicate)
            self.turn_count += 1
            with policy.open("a", encoding="utf-8") as stream:
                stream.write("\n# code changed but proposal omitted\n")
            return session

    goal = GoalThatOmitsImprovementProposal()
    run = build_offline_run(tmp_path, FixtureArena(tmp_path), goal)
    run.execute_until("first_match_finalized")

    with pytest.raises(ValueError, match="proposal.json"):
        run.advance_one_iteration()

    assert set(run.candidates.state.versions) == {"v000"}
    assert not any(
        event.payload.get("candidate_id") == "v001" for event in run._events("CandidateValidated")
    )


def test_improvement_without_policy_change_is_not_sealed(tmp_path: Path) -> None:
    class GoalThatWritesOnlyProposal(GoalThatWritesBaseline):
        def run_until_checkpoint(
            self,
            session: AgentSession,
            run_context: RunContext,
            checkpoint_predicate,
        ) -> AgentSession:
            policy = run_context.candidate_root / "ai.py"
            if not policy.exists():
                return super().run_until_checkpoint(session, run_context, checkpoint_predicate)
            self.turn_count += 1
            (run_context.candidate_root / ".agentbench/proposal.json").write_text(
                json.dumps(
                    {
                        "condition": "公开败局状态仍未解决",
                        "mechanism": "本 checkpoint 没有落实代码干预",
                        "intervention": "应当修改公开策略代码",
                        "expected_observation": "至少一个公开决策发生变化",
                        "continuation_rationale": "无变化候选不能进入正式评测",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return session

    goal = GoalThatWritesOnlyProposal()
    run = build_offline_run(tmp_path, FixtureArena(tmp_path), goal)
    run.execute_until("first_match_finalized")

    with pytest.raises(ValueError, match="does not change candidate code"):
        run.advance_one_iteration()

    assert set(run.candidates.state.versions) == {"v000"}


def test_resume_aborts_a_legacy_sealed_checkpoint_without_proposal(
    tmp_path: Path,
) -> None:
    goal = GoalThatWritesBaseline()
    run = build_offline_run(tmp_path, FixtureArena(tmp_path), goal)
    run.execute_until("first_match_finalized")
    run._ensure_baseline_metrics()
    corrupt_plan = run._iteration_plan()
    corrupt_workspace = run._iteration_workspace(
        iteration=corrupt_plan.iteration,
        version_id=corrupt_plan.version_id,
        parent_id=corrupt_plan.parent_id,
        target_id=corrupt_plan.target_id,
    )
    corrupt_version = run.candidates.seal(corrupt_workspace.workspace_id)
    assert corrupt_version.version_id == "v001"
    assert not (corrupt_workspace.path / ".agentbench/proposal.json").is_file()

    result = run.advance_one_iteration()

    assert result.version_id == "v002"
    assert result.metrics.research_iteration == 1
    aborted = run._events("IterationAborted")
    assert len(aborted) == 1
    assert aborted[0].payload["version_id"] == "v001"
    assert aborted[0].payload["counts_as_scientific_act"] is False
    created = run._events("CandidateCreated")
    assert len(created) == 3
    assert created[-1].idempotency_key == "candidate-created:v002"
    assert created[-1].payload["workspace_id"] != corrupt_workspace.workspace_id


def test_zero_token_transport_failure_retries_without_counting_a_scientific_turn(
    tmp_path: Path,
) -> None:
    class GoalThatDisconnectsOnce(GoalThatWritesBaseline):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = False
            self.failed_telemetry_pending = False
            self.effort_changes: list[str] = []

        def set_turn_reasoning_effort(self, effort: str) -> None:
            self.effort_changes.append(effort)

        def run_until_checkpoint(
            self,
            session: AgentSession,
            run_context: RunContext,
            checkpoint_predicate,
        ) -> AgentSession:
            if self.fail_next:
                self.fail_next = False
                self.failed_telemetry_pending = True
                self.turn_count += 1
                raise RuntimeError("stream disconnected before completion")
            return super().run_until_checkpoint(session, run_context, checkpoint_predicate)

        def consume_turn_telemetry(self):
            if self.failed_telemetry_pending:
                self.failed_telemetry_pending = False
                return (
                    {
                        "input_tokens": None,
                        "cached_input_tokens": None,
                        "output_tokens": None,
                        "reasoning_tokens": None,
                        "total_tokens": None,
                        "wall_time_s": 0.5,
                    },
                )
            return super().consume_turn_telemetry()

    goal = GoalThatDisconnectsOnce()
    run = build_offline_run(tmp_path, FixtureArena(tmp_path), goal)
    run.execute_until("first_match_finalized")
    goal.fail_next = True

    result = run.advance_one_iteration()

    assert result.version_id == "v001"
    assert goal.turn_count == 3
    usage = tuple(
        event
        for event in run._events("ModelUsageFinalized")
        if event.payload["candidate_id"] == "v001"
    )
    assert len(usage) == 1
    assert usage[0].payload["total_tokens"] == 15
    rejected = run._events("AgentTurnRejected")
    assert len(rejected) == 1
    assert rejected[0].payload["counts_as_scientific_act"] is False
    assert goal.effort_changes == ["high"]
    adaptations = run._events("AgentReasoningEffortAdapted")
    assert len(adaptations) == 1
    assert adaptations[0].payload["from_effort"] == "xhigh"
    assert adaptations[0].payload["to_effort"] == "high"


def test_all_zero_token_retry_failures_are_recorded_without_model_usage(
    tmp_path: Path,
) -> None:
    class GoalThatAlwaysDisconnectsAfterBaseline(GoalThatWritesBaseline):
        def __init__(self) -> None:
            super().__init__()
            self.disconnect = False
            self.reasoning_effort = "xhigh"
            self.effort_changes: list[str] = []

        def set_turn_reasoning_effort(self, effort: str) -> None:
            self.reasoning_effort = effort
            self.effort_changes.append(effort)

        def run_until_checkpoint(
            self,
            session: AgentSession,
            run_context: RunContext,
            checkpoint_predicate,
        ) -> AgentSession:
            if self.disconnect:
                self.turn_count += 1
                raise RuntimeError("stream disconnected before completion")
            return super().run_until_checkpoint(session, run_context, checkpoint_predicate)

        def consume_turn_telemetry(self):
            if self.disconnect:
                return (
                    {
                        "input_tokens": None,
                        "cached_input_tokens": None,
                        "output_tokens": None,
                        "reasoning_tokens": None,
                        "total_tokens": None,
                        "wall_time_s": 0.5,
                    },
                )
            return super().consume_turn_telemetry()

    goal = GoalThatAlwaysDisconnectsAfterBaseline()
    run = build_offline_run(tmp_path, FixtureArena(tmp_path), goal)
    run.execute_until("first_match_finalized")
    goal.disconnect = True

    with pytest.raises(RuntimeError, match="stream disconnected"):
        run.advance_one_iteration()

    rejected = tuple(
        event
        for event in run._events("AgentTurnRejected")
        if event.payload["candidate_id"] == "v001"
    )
    usage = tuple(
        event
        for event in run._events("ModelUsageFinalized")
        if event.payload["candidate_id"] == "v001"
    )
    assert len(rejected) == 3
    assert usage == ()
    assert goal.effort_changes == ["high", "medium"]
    adaptations = tuple(
        event
        for event in run._events("AgentReasoningEffortAdapted")
        if event.payload["candidate_id"] == "v001"
    )
    assert [event.payload["to_effort"] for event in adaptations] == [
        "high",
        "medium",
    ]


def test_iteration_records_real_policy_trace_samples_before_finalizing_ig(
    tmp_path: Path,
) -> None:
    def probe(candidate_root, replay_path, *, match_id, role):
        revised = "grounded revision" in (candidate_root / "ai.py").read_text()
        actions = ("31",) if revised else ()
        support = ("HOLD", "31")
        return PolicyEpisodeTrace(
            match_id,
            role,
            (
                PolicyDecision(
                    f"{match_id}:r0000:{role.lower()}",
                    actions,
                    (support,) if not actions else (support, ("HOLD",)),
                    "shared-public-state",
                ),
            ),
        )

    run = build_offline_run(
        tmp_path,
        FixtureArena(tmp_path),
        GoalThatWritesBaseline(),
        policy_probe=probe,
    )
    run.execute_until("first_match_finalized")

    result = run.advance_one_iteration()

    assert result.metrics.behavioral_ig is not None
    assert result.metrics.behavioral_ig > 0
    assert len(run._events("DecisionSampleRecorded")) == 2
    episode_metrics = run._events("PolicyEpisodeMeasured")
    assert len(episode_metrics) == 2
    assert episode_metrics[0].payload["local_policy_kl_trace"][0] > 0
    assert len(run._events("OccupancyTraceRecorded")) == 2
    assert result.metrics.occupancy_shift == 0.0


def test_first_improvement_act_finalizes_v000_and_v001_on_the_same_fixed_panel(
    tmp_path: Path,
) -> None:
    run = build_offline_run(
        tmp_path,
        FixtureArena(tmp_path),
        GoalThatWritesBaseline(),
    )
    run.execute_until("first_match_finalized")

    run.advance_one_iteration()

    rows = run._events("IterationMetricsFinalized")
    assert tuple(item.payload["candidate_id"] for item in rows) == ("v000", "v001")
    assert tuple(item.payload["win_rate"] for item in rows) == (0.5, 0.5)
    csv_path = run.root / "reports/curves/iteration-metrics.csv"
    assert csv_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 3


def test_baseline_only_evaluates_the_weakest_unsolved_opponent(
    tmp_path: Path,
) -> None:
    run = build_offline_run(
        tmp_path,
        FixtureArena(tmp_path),
        GoalThatWritesBaseline(),
    )
    strongest = Opponent(
        opponent_id="rank01",
        rank=1,
        username="strongest-fixture-human",
        score=2000,
        archive=tmp_path / "rank01.zip",
        archive_sha256="1" * 64,
        package_root=run.human_pool_root,
        runnable=True,
        entry_command=("python", "main.py"),
        exclusion_diagnostic=None,
    )
    run.opponents = (strongest, *run.opponents)
    run.human_ratings["rank01"] = 2000.0
    run.opponent_hashes["rank01"] = strongest.archive_sha256
    run.execute_until("first_match_finalized")

    run._ensure_baseline_metrics()

    completed = run._events("EvaluationCaseCompleted")
    assert {event.payload["opponent_id"] for event in completed} == {"rank20"}
    assert len(completed) == 2


def test_baseline_materializes_target_replays_and_experience_before_v001(
    tmp_path: Path,
) -> None:
    run = build_offline_run(
        tmp_path,
        FixtureArena(tmp_path),
        GoalThatWritesBaseline(),
    )
    run.execute_until("first_match_finalized")

    run._ensure_baseline_metrics()

    assert (run.root / "replays/v000-rank20-p0-s1/narrative.md").is_file()
    assert (run.root / "replays/v000-rank20-p1-s1/narrative.md").is_file()
    experience_ids = {event.payload["experience_id"] for event in run._events("ExperienceRecorded")}
    assert "exp-v000-rank20-p0" in experience_ids
    assert "exp-v000-rank20-p1" in experience_ids


def test_improvement_act_measures_only_the_current_curriculum_target(
    tmp_path: Path,
) -> None:
    run = build_offline_run(
        tmp_path,
        FixtureArena(tmp_path),
        GoalThatWritesBaseline(),
    )
    strongest = Opponent(
        opponent_id="rank01",
        rank=1,
        username="strongest-fixture-human",
        score=2000,
        archive=tmp_path / "rank01.zip",
        archive_sha256="1" * 64,
        package_root=run.human_pool_root,
        runnable=True,
        entry_command=("python", "main.py"),
        exclusion_diagnostic=None,
    )
    run.opponents = (strongest, *run.opponents)
    run.human_ratings["rank01"] = 2000.0
    run.opponent_hashes["rank01"] = strongest.archive_sha256
    run.execute_until("first_match_finalized")

    run.advance_one_iteration()

    v001 = tuple(
        event
        for event in run._events("EvaluationCaseCompleted")
        if event.payload["version_id"] == "v001"
    )
    assert {event.payload["opponent_id"] for event in v001} == {"rank20"}
    assert len(v001) == 2


def test_apparent_all_win_requires_two_additional_seed_waves_before_promotion(
    tmp_path: Path,
) -> None:
    class NoisyQualificationArena(FixtureArena):
        def run_case(self, case: MatchCase, candidate_root: Path):
            self.call_count += 1
            replay = self.root / f"{case.candidate_id}-{case.role}-{case.seed}.json"
            replay.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
            won = case.candidate_id != "v000" and case.seed < 10_000
            return MatchResult(
                case=case,
                status="complete",
                result="win" if won else "loss",
                points=1.0 if won else 0.0,
                score_margin=10.0 if won else -10.0,
                rounds=4,
                payload={"terminal_base_hp": (50.0, 0.0) if won else (0.0, 50.0)},
                replay_path=replay,
            )

    run = build_offline_run(
        tmp_path,
        NoisyQualificationArena(tmp_path),
        GoalThatWritesBaseline(),
    )
    run.execute_until("first_match_finalized")

    result = run.advance_one_iteration()

    cases = tuple(
        event.payload
        for event in run._events("EvaluationCaseCompleted")
        if event.payload["version_id"] == "v001"
    )
    assert {item["seed"] for item in cases} == {1, 10_001, 20_001}
    assert len(cases) == 6
    assert result.selection == "frontier"
    assert run.candidates.state.champion_id == "v000"


def test_certification_completes_only_after_frozen_matrix_all_wins(
    tmp_path: Path,
) -> None:
    class WinningArena(FixtureArena):
        def run_case(self, case: MatchCase, candidate_root: Path):
            self.call_count += 1
            replay = self.root / f"{case.candidate_id}-{case.role}-{case.seed}.json"
            replay.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
            return MatchResult(
                case=case,
                status="complete",
                result="win",
                points=1.0,
                score_margin=10.0,
                rounds=4,
                payload={"terminal_base_hp": (50.0, 0.0)},
                replay_path=replay,
            )

    run = build_offline_run(
        tmp_path,
        WinningArena(tmp_path),
        GoalThatWritesBaseline(),
    )
    public_manifest = json.loads((run.root / "run-manifest.json").read_text())
    assert "certification_roles" not in public_manifest
    assert "certification_seeds" not in public_manifest
    run.execute_until("first_match_finalized")
    run._ensure_baseline_metrics()
    public_evaluations_before = len(run._events("EvaluationCaseCompleted"))

    result = run.certify_champion()

    assert result.passed is True
    assert result.champion_id == "v000"
    assert result.total_cases == 2
    assert result.wins == 2
    assert len(run._events("RunCompleted")) == 1
    assert len(run._events("EvaluationCaseCompleted")) == public_evaluations_before
    hidden_events = JsonlEventStore(run.root / "hidden-certification/events.jsonl").read_all()
    assert sum(item.event_type == "EvaluationCaseCompleted" for item in hidden_events) == 2


def test_failed_hidden_certification_continues_learning_on_strongest_public_target(
    tmp_path: Path,
) -> None:
    class DevelopmentWinsCertificationLoses(FixtureArena):
        def run_case(self, case: MatchCase, candidate_root: Path):
            self.call_count += 1
            replay = self.root / f"{case.candidate_id}-{case.role}-{case.seed}.json"
            replay.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
            won = case.seed != 11
            return MatchResult(
                case=case,
                status="complete",
                result="win" if won else "loss",
                points=1.0 if won else 0.0,
                score_margin=10.0 if won else -10.0,
                rounds=4,
                payload={"terminal_base_hp": (50.0, 0.0) if won else (0.0, 50.0)},
                replay_path=replay,
            )

    run = build_offline_run(
        tmp_path,
        DevelopmentWinsCertificationLoses(tmp_path),
        GoalThatWritesBaseline(),
    )
    run.execute_until("first_match_finalized")
    run._ensure_baseline_metrics()
    certification = run.certify_champion()
    assert certification.passed is False
    certification_event = run._events("CertificationFinalized")[-1]
    assert certification_event.payload["failed_count"] == 2
    assert "failed_cases" not in certification_event.payload
    assert "incomplete_cases" not in certification_event.payload

    result = run.advance_one_iteration()

    assert result.version_id == "v001"
    assert result.target_id == "rank20"
    assert len(run._events("RunCompleted")) == 0


def test_non_winning_frontier_grows_across_iterations_without_v0_rollback(
    tmp_path: Path,
) -> None:
    arena = FixtureArena(tmp_path)
    goal = GoalThatWritesBaseline()
    run = build_offline_run(tmp_path, arena, goal)
    run.execute_until("first_match_finalized")

    first = run.advance_one_iteration()
    second = run.advance_one_iteration()

    assert first.version_id == "v001"
    assert first.selection == "frontier"
    assert run.candidates.state.versions["v001"].parent_id == "v000"
    assert second.version_id == "v002"
    assert run.candidates.state.versions["v002"].parent_id == "v001"
    assert run.candidates.state.champion_id == "v000"
    assert run.candidates.state.frontier_id == "v002"
    assert run.candidates.state.exploration_debt == 2
    assert len(run._events("ExperienceRecorded")) == 7
    v001_roles = {
        event.payload["role"]
        for event in run._events("ExperienceRecorded")
        if event.payload.get("candidate_id") == "v001"
    }
    assert v001_roles == {"P0", "P1"}


def test_curriculum_uses_the_selected_frontier_for_promotion_progress(
    tmp_path: Path,
) -> None:
    class WinningArena(FixtureArena):
        def run_case(self, case: MatchCase, candidate_root: Path):
            self.call_count += 1
            replay = self.root / f"{case.candidate_id}-{case.role}-{case.seed}.json"
            replay.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
            return MatchResult(
                case=case,
                status="complete",
                result="win" if case.candidate_id != "v000" and case.role == "P0" else "loss",
                points=1.0 if case.candidate_id != "v000" and case.role == "P0" else 0.0,
                score_margin=10.0,
                rounds=4,
                payload={
                    "terminal_base_hp": (50.0, 0.0)
                    if case.candidate_id != "v000" and case.role == "P0"
                    else (0.0, 50.0)
                },
                replay_path=replay,
            )

    run = build_offline_run(
        tmp_path,
        WinningArena(tmp_path),
        GoalThatWritesBaseline(),
    )
    run.required_win_rate = 0.5
    run.execute_until("first_match_finalized")
    result = run.advance_one_iteration()

    assert result.selection == "frontier"
    assert run.candidates.state.champion_id == "v000"
    assert run.candidates.state.frontier_id == "v001"
    assert run._curriculum().status().by_opponent["rank20"].state == "solved"


def test_curriculum_never_combines_wins_from_different_candidate_versions(
    tmp_path: Path,
) -> None:
    run = build_offline_run(
        tmp_path,
        FixtureArena(tmp_path),
        GoalThatWritesBaseline(),
    )
    run.execute_until("first_match_finalized")
    run.event_store.append(
        FinalizedEvent.create(
            "EvaluationCaseCompleted",
            {
                "version_id": "v001",
                "opponent_id": "rank20",
                "role": "P1",
                "seed": 1,
                "status": "complete",
                "result": "win",
            },
            idempotency_key="fixture:v001:rank20:P1:1",
        )
    )

    status = run._curriculum().status()

    assert run.candidates.state.champion_id == "v000"
    assert status.by_opponent["rank20"].state == "unsolved"
    assert status.locked_regression_ids == ()


def test_iteration_repairs_failed_candidate_before_sealing(tmp_path: Path) -> None:
    class FailFirstRevision:
        def __init__(self) -> None:
            self.failed_revision = False

        def __call__(self, workspace: Path) -> ValidationResult:
            proposal = workspace / ".agentbench/proposal.json"
            if proposal.exists() and not self.failed_revision:
                self.failed_revision = True
                return ValidationResult("failed", "fixture import error", {})
            return ValidationResult("complete", None, {"roles": ["P0", "P1"]})

    goal = GoalThatWritesBaseline()
    validator = FailFirstRevision()
    run = build_offline_run(
        tmp_path,
        FixtureArena(tmp_path),
        goal,
        candidate_validator=validator,
    )
    run.execute_until("first_match_finalized")

    result = run.advance_one_iteration()

    validations = [
        event.payload
        for event in run._events("CandidateValidated")
        if event.payload.get("candidate_id") == "v001"
    ]
    assert result.version_id == "v001"
    assert [item["status"] for item in validations] == ["failed", "complete"]
    assert goal.turn_count == 3

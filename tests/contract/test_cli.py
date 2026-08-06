from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.application.curriculum_service import CurriculumComplete
from agentbench_hl.application.iteration_service import choose_iteration_parent
from agentbench_hl.cli.main import _load_env_file, main
from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.lineage import CandidateVersion, LineageState


def test_cli_missing_run_is_json_and_redacts_key(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ABHL_API_KEY", "sk-abcdefghijk")

    exit_code = main(["run", "status", "--run-root", str(tmp_path / "missing")])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert "sk-" not in captured.out
    assert "sk-" not in captured.err


def test_cli_status_reports_integer_iteration(capsys, tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run-manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "research_iteration": 0}),
        encoding="utf-8",
    )
    (run_root / "events.jsonl").write_text("", encoding="utf-8")

    exit_code = main(["run", "status", "--run-root", str(run_root)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "initialized"
    assert payload["research_iteration"] == 0


def test_cli_status_reports_formal_matches_lineage_and_experience(capsys, tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run-manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "research_iteration": 0}),
        encoding="utf-8",
    )
    (run_root / "checkpoint.json").write_text("{}", encoding="utf-8")
    store = JsonlEventStore(run_root / "events.jsonl")
    rows = (
        ("CandidateSealed", {"version_id": "v000"}),
        ("CandidatePromoted", {"version_id": "v000"}),
        ("FrontierSelected", {"version_id": "v001"}),
        ("MatchFinalized", {"status": "complete"}),
        ("EvaluationCaseCompleted", {"version_id": "v000"}),
        ("EvaluationCaseCompleted", {"version_id": "v001"}),
        ("ExperienceRecorded", {"experience_id": "exp-1"}),
        (
            "IterationMetricsFinalized",
            {"research_iteration": 1, "candidate_id": "v001"},
        ),
    )
    for index, (event_type, payload) in enumerate(rows):
        store.append(
            FinalizedEvent.create(
                event_type,
                payload,
                f"status:{index}",
                occurred_at=datetime(2026, 8, 5, index, tzinfo=UTC),
            )
        )

    exit_code = main(["run", "status", "--run-root", str(run_root)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["research_iteration"] == 1
    assert payload["smoke_match_count"] == 1
    assert payload["formal_match_count"] == 2
    assert payload["match_count"] == 3
    assert payload["experience_count"] == 1
    assert payload["champion_id"] == "v000"
    assert payload["frontier_id"] == "v001"


def test_lineage_rollback_selects_archived_version_as_next_parent(capsys, tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    store = JsonlEventStore(run_root / "events.jsonl")
    versions = (
        CandidateVersion(
            version_id="v007",
            parent_id=None,
            workspace_id="workspace-v007",
            content_hash="7" * 64,
            object_path=run_root / "candidates" / ("7" * 64),
            source_hashes={"ai.py": "7" * 64},
            reason="stable qualification parent",
        ),
        CandidateVersion(
            version_id="v008",
            parent_id="v007",
            workspace_id="workspace-v008",
            content_hash="8" * 64,
            object_path=run_root / "candidates" / ("8" * 64),
            source_hashes={"ai.py": "8" * 64},
            reason="refuted intervention",
        ),
    )
    for version in versions:
        store.append(
            FinalizedEvent.create(
                "CandidateSealed",
                version.to_payload(),
                f"candidate-sealed:{version.version_id}",
            )
        )
    store.append(
        FinalizedEvent.create(
            "CandidatePromoted",
            {"version_id": "v007"},
            "candidate-promoted:v007",
        )
    )
    store.append(
        FinalizedEvent.create(
            "FrontierSelected",
            {"version_id": "v008", "rationale": "test risky branch"},
            "frontier-selected:v008:1",
        )
    )

    exit_code = main(
        [
            "lineage",
            "rollback",
            "--run-root",
            str(run_root),
            "--version",
            "v007",
            "--reason",
            "v008 hypothesis was refuted by target evaluation",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    lineage = LineageState.replay(store.read_all())
    assert exit_code == 0
    assert payload == {
        "frontier_id": "v007",
        "status": "checkpoint",
    }
    assert choose_iteration_parent(lineage) == "v007"


def test_local_env_loader_does_not_overwrite_process_environment(monkeypatch, tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "ABHL_API_KEY='fixture-secret'\nAGENTBENCH_ROOT=/frozen/bench\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    monkeypatch.setenv("AGENTBENCH_ROOT", "/explicit/bench")
    monkeypatch.delenv("ABHL_API_KEY", raising=False)

    _load_env_file(path)

    assert os.environ["ABHL_API_KEY"] == "fixture-secret"
    assert os.environ["AGENTBENCH_ROOT"] == "/explicit/bench"


def test_local_env_loader_rejects_group_or_world_access(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text("ABHL_API_KEY=fixture-secret\n", encoding="utf-8")
    path.chmod(0o644)

    with pytest.raises(ValueError, match="permissions"):
        _load_env_file(path)


def test_run_audit_reports_complete_from_scratch_checkpoint(capsys, tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run-manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "origin": "from_scratch"}),
        encoding="utf-8",
    )
    (run_root / "checkpoint.json").write_text(
        json.dumps({"thread_id": "thread-fixture"}), encoding="utf-8"
    )
    store = JsonlEventStore(run_root / "events.jsonl")
    for index, (event_type, payload) in enumerate(
        (
            ("CandidateSealed", {"version_id": "v000"}),
            ("MatchFinalized", {"status": "complete", "match_id": "match-1"}),
            ("ReplayDecoded", {"match_id": "match-1"}),
            ("ExperienceRecorded", {"experience_id": "exp-1"}),
        )
    ):
        store.append(
            FinalizedEvent.create(
                event_type,
                payload,
                f"fixture:{index}",
                occurred_at=datetime(2026, 8, 5, index, tzinfo=UTC),
            )
        )

    exit_code = main(["run", "audit", "--run-root", str(run_root)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "candidate_origin": "from_scratch",
        "credential_leaks": 0,
        "experience_records": 1,
        "matches_complete": 1,
        "reference_policy_leaks": 0,
        "resumable": True,
        "semantic_replays": 1,
        "status": "complete",
    }


def test_run_audit_fails_when_artifact_contains_credential(capsys, tmp_path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "run-manifest.json").write_text(
        json.dumps({"schema_version": "1.0", "origin": "from_scratch"}),
        encoding="utf-8",
    )
    (run_root / "leak.txt").write_text("sk-abcdefghijk", encoding="utf-8")

    exit_code = main(["run", "audit", "--run-root", str(run_root)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["credential_leaks"] == 1
    assert "sk-" not in json.dumps(payload)


def test_run_resume_executes_requested_acts_under_one_lease(capsys, monkeypatch, tmp_path) -> None:
    repository = tmp_path / "repository"
    config = repository / "configs/experiments/antwar2.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("fixture", encoding="utf-8")

    class Runtime:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Run:
        def __init__(self) -> None:
            self.root = tmp_path / "run"
            self.root.mkdir()
            self.runtime = Runtime()
            self.calls = 0

        def advance_one_iteration(self):
            self.calls += 1
            return SimpleNamespace(
                version_id=f"v{self.calls:03d}",
                parent_id=f"v{self.calls - 1:03d}",
                target_id="rank20",
                selection="frontier",
                metrics=SimpleNamespace(research_iteration=self.calls),
            )

    run = Run()
    monkeypatch.setattr(
        "agentbench_hl.application.live_run.resume_live_run",
        lambda _config, *, run_id: run,
    )

    exit_code = main(
        [
            "run",
            "resume",
            "--config",
            str(config),
            "--run-id",
            "fixture-run",
            "--acts",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "checkpoint"
    assert payload["acts_completed"] == 2
    assert payload["candidate_id"] == "v002"
    assert payload["research_iteration"] == 2
    assert run.calls == 2
    assert run.runtime.closed is True


def test_run_init_creates_the_from_scratch_checkpoint(capsys, monkeypatch, tmp_path) -> None:
    repository = tmp_path / "repository"
    config = repository / "configs/experiments/antwar2.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("fixture", encoding="utf-8")

    class Runtime:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Result:
        root = tmp_path / "run"
        match_id = "v000-rank20-p0-s1"
        metrics = SimpleNamespace(candidate_id="v000", research_iteration=0)

        @staticmethod
        def event_count(event_type: str) -> int:
            return {"MatchFinalized": 1, "ExperienceRecorded": 1}[event_type]

    class Run:
        def __init__(self) -> None:
            self.root = Result.root
            self.root.mkdir()
            self.runtime = Runtime()

        def execute_until(self, checkpoint: str):
            assert checkpoint == "first_match_finalized"
            (self.root / "checkpoint.json").write_text("{}", encoding="utf-8")
            return Result()

    run = Run()
    monkeypatch.setattr(
        "agentbench_hl.application.live_run.build_live_run",
        lambda _config, *, run_id: run,
    )

    exit_code = main(
        [
            "run",
            "init",
            "--config",
            str(config),
            "--run-id",
            "fixture-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["candidate_id"] == "v000"
    assert payload["research_iteration"] == 0
    assert payload["resumable"] is True
    assert run.runtime.closed is True


def test_run_init_resumes_an_existing_incomplete_v000_checkpoint(
    capsys, monkeypatch, tmp_path
) -> None:
    repository = tmp_path / "repository"
    config = repository / "configs/experiments/antwar2.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("fixture", encoding="utf-8")

    class Runtime:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Result:
        root = tmp_path / "run"
        match_id = "v000-rank20-p0-s1"
        metrics = SimpleNamespace(candidate_id="v000", research_iteration=0)

        @staticmethod
        def event_count(event_type: str) -> int:
            return {"MatchFinalized": 1, "ExperienceRecorded": 1}[event_type]

    class Run:
        root = Result.root
        runtime = Runtime()

        def execute_until(self, checkpoint: str):
            assert checkpoint == "first_match_finalized"
            return Result()

    resumed = Run()
    monkeypatch.setattr(
        "agentbench_hl.application.live_run.build_live_run",
        lambda _config, *, run_id: (_ for _ in ()).throw(
            ValueError(f"run already exists: {run_id}")
        ),
    )
    monkeypatch.setattr(
        "agentbench_hl.application.live_run.resume_live_run",
        lambda _config, *, run_id: resumed,
    )

    exit_code = main(
        [
            "run",
            "init",
            "--config",
            str(config),
            "--run-id",
            "fixture-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "checkpoint"
    assert payload["candidate_id"] == "v000"
    assert resumed.runtime.closed is True


def test_run_resume_automatically_certifies_when_public_curriculum_is_solved(
    capsys, monkeypatch, tmp_path
) -> None:
    repository = tmp_path / "repository"
    config = repository / "configs/experiments/antwar2.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("fixture", encoding="utf-8")

    class Runtime:
        def close(self) -> None:
            pass

    class Run:
        root = tmp_path / "run"
        runtime = Runtime()

        def advance_one_iteration(self):
            raise CurriculumComplete("all runnable human opponents are solved")

        def certify_champion(self):
            return SimpleNamespace(
                passed=True,
                champion_id="v009",
                total_cases=120,
                wins=120,
                incomplete_cases=(),
                failed_cases=(),
            )

    Run.root.mkdir()
    monkeypatch.setattr(
        "agentbench_hl.application.live_run.resume_live_run",
        lambda _config, *, run_id: Run(),
    )

    exit_code = main(
        [
            "run",
            "resume",
            "--config",
            str(config),
            "--run-id",
            "fixture-run",
            "--acts",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "complete"
    assert payload["champion_id"] == "v009"
    assert payload["certification_wins"] == 120


def test_run_pursue_keeps_advancing_until_certification_passes(
    capsys, monkeypatch, tmp_path
) -> None:
    repository = tmp_path / "repository"
    config = repository / "configs/experiments/antwar2.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("fixture", encoding="utf-8")

    class Runtime:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Run:
        root = tmp_path / "run"
        runtime = Runtime()

        def __init__(self) -> None:
            self.calls = 0

        def advance_one_iteration(self):
            self.calls += 1
            if self.calls == 2:
                raise CurriculumComplete("public curriculum solved")
            return SimpleNamespace(
                version_id="v001",
                parent_id="v000",
                target_id="rank20",
                selection="promoted",
                metrics=SimpleNamespace(research_iteration=1),
            )

        def certify_champion(self):
            return SimpleNamespace(
                passed=True,
                champion_id="v001",
                total_cases=108,
                wins=108,
                incomplete_cases=(),
                failed_cases=(),
            )

    Run.root.mkdir()
    run = Run()
    monkeypatch.setattr(
        "agentbench_hl.application.live_run.resume_live_run",
        lambda _config, *, run_id: run,
    )

    exit_code = main(
        [
            "run",
            "pursue",
            "--config",
            str(config),
            "--run-id",
            "fixture-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "complete"
    assert payload["acts_completed"] == 1
    assert payload["champion_id"] == "v001"
    assert payload["certification_wins"] == 108
    assert run.runtime.closed is True

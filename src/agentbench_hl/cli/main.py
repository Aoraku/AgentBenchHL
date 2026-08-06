"""Stable, machine-readable AgentBench HL command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.lineage import LineageState

_CREDENTIAL = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_REFERENCE_MARKERS = (
    b"handoff_next_agent",
    b"reference_policy",
    b"candidate-v239",
    b"candidate-v22",
)


def _redact(value: object) -> object:
    if isinstance(value, str):
        return _CREDENTIAL.sub("[REDACTED]", value)
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    if path.stat().st_mode & 0o077:
        raise ValueError(f".env permissions must deny group/world access: {path}")
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"invalid .env entry at line {line_number}")
        name, value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid .env name at line {line_number}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def _status(run_root: Path) -> dict[str, object]:
    manifest_path = run_root / "run-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = JsonlEventStore(run_root / "events.jsonl").read_all()
    types = [item.event_type for item in events]
    status = (
        "checkpoint"
        if "CheckpointCreated" in types or (run_root / "checkpoint.json").is_file()
        else "initialized"
    )
    if "RunCompleted" in types:
        status = "complete"
    metrics = [item for item in events if item.event_type == "IterationMetricsFinalized"]
    iteration = (
        int(metrics[-1].payload["research_iteration"])
        if metrics
        else int(manifest.get("research_iteration", 0))
    )

    def latest_version(event_type: str) -> str | None:
        return next(
            (
                str(event.payload["version_id"])
                for event in reversed(events)
                if event.event_type == event_type
                and isinstance(event.payload.get("version_id"), str)
            ),
            None,
        )

    smoke_matches = sum(
        event.event_type == "MatchFinalized" and event.payload.get("status") == "complete"
        for event in events
    )
    formal_matches = types.count("EvaluationCaseCompleted")
    return {
        "status": status,
        "research_iteration": iteration,
        "event_count": len(events),
        "candidate_count": types.count("CandidateSealed"),
        "smoke_match_count": smoke_matches,
        "formal_match_count": formal_matches,
        "match_count": smoke_matches + formal_matches,
        "experience_count": types.count("ExperienceRecorded"),
        "champion_id": latest_version("CandidatePromoted"),
        "frontier_id": latest_version("FrontierSelected"),
        "resumable": (run_root / "checkpoint.json").is_file(),
    }


def _audit(run_root: Path) -> dict[str, object]:
    manifest_path = run_root / "run-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = JsonlEventStore(run_root / "events.jsonl").read_all()
    credential_leaks = 0
    reference_leaks = 0
    for path in run_root.rglob("*"):
        if not path.is_file() or path.name == ".env":
            continue
        content = path.read_bytes()
        credential_leaks += bool(_CREDENTIAL.search(content.decode("latin1")))
        lowered = content.lower()
        reference_leaks += any(marker in lowered for marker in _REFERENCE_MARKERS)
    matches = sum(
        event.event_type == "MatchFinalized" and event.payload.get("status") == "complete"
        for event in events
    )
    semantic_replays = sum(event.event_type == "ReplayDecoded" for event in events)
    experience = sum(event.event_type == "ExperienceRecorded" for event in events)
    candidates = sum(event.event_type == "CandidateSealed" for event in events)
    resumable = (run_root / "checkpoint.json").is_file()
    origin = manifest.get("origin")
    complete = all(
        (
            origin == "from_scratch",
            credential_leaks == 0,
            reference_leaks == 0,
            candidates >= 1,
            matches >= 1,
            semantic_replays >= 1,
            experience >= 1,
            resumable,
        )
    )
    return {
        "status": "complete" if complete else "failed",
        "credential_leaks": credential_leaks,
        "reference_policy_leaks": reference_leaks,
        "candidate_origin": origin,
        "matches_complete": matches,
        "semantic_replays": semantic_replays,
        "experience_records": experience,
        "resumable": resumable,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abhl")
    groups = parser.add_subparsers(dest="group", required=True)
    run = groups.add_parser("run")
    commands = run.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.add_argument("--run-root", type=Path, required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--run-id", required=True)
    initialize = commands.add_parser("init")
    initialize.add_argument("--config", type=Path, required=True)
    initialize.add_argument("--run-id", required=True)
    resume = commands.add_parser("resume")
    resume.add_argument("--config", type=Path, required=True)
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--acts", type=int, default=1)
    pursue = commands.add_parser("pursue")
    pursue.add_argument("--config", type=Path, required=True)
    pursue.add_argument("--run-id", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--run-root", type=Path, required=True)
    goal_led = groups.add_parser("goal-led")
    goal_led_commands = goal_led.add_subparsers(dest="command", required=True)
    for name in ("start", "continue"):
        command = goal_led_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--run-id", required=True)
    lineage = groups.add_parser("lineage")
    lineage_commands = lineage.add_subparsers(dest="command", required=True)
    rollback = lineage_commands.add_parser("rollback")
    rollback.add_argument("--run-root", type=Path, required=True)
    rollback.add_argument("--version", required=True)
    rollback.add_argument("--reason", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
        if arguments.group == "goal-led":
            from agentbench_hl.application.live_run import build_goal_led_service
            from agentbench_hl.application.run_lease import RunLease

            repository_root = arguments.config.resolve().parents[2]
            _load_env_file(repository_root / ".env")
            service = build_goal_led_service(
                arguments.config.resolve(),
                run_id=arguments.run_id,
                resume=arguments.command == "continue",
            )
            try:
                with RunLease(service.root):
                    outcome = service.start() if arguments.command == "start" else service.advance()
            finally:
                close = getattr(service.runtime, "close", None)
                if callable(close):
                    close()
            payload = {
                "status": "checkpoint",
                "run_root": str(service.root),
                "thread_id": outcome.thread_id,
                "workspace": str(outcome.workspace),
                "request_id": outcome.request_id,
                "match_count": outcome.match_count,
            }
        elif arguments.group == "lineage" and arguments.command == "rollback":
            from agentbench_hl.application.run_lease import RunLease

            run_root = arguments.run_root.resolve()
            store = JsonlEventStore(run_root / "events.jsonl")
            with RunLease(run_root):
                events = store.read_all()
                lineage = LineageState.replay(events)
                if lineage.frontier_id != arguments.version:
                    lineage.choose_frontier(arguments.version, arguments.reason)
                    store.append(
                        FinalizedEvent.create(
                            "FrontierSelected",
                            {
                                "version_id": arguments.version,
                                "rationale": arguments.reason,
                            },
                            (f"frontier-rollback:{arguments.version}:{len(events) + 1}"),
                        )
                    )
            payload = {
                "status": "checkpoint",
                "frontier_id": arguments.version,
            }
        elif arguments.group == "run" and arguments.command == "status":
            payload = _status(arguments.run_root.resolve())
        elif arguments.group == "run" and arguments.command in {"smoke", "init"}:
            from agentbench_hl.application.live_run import (
                build_live_run,
                resume_live_run,
            )
            from agentbench_hl.application.run_lease import RunLease

            repository_root = arguments.config.resolve().parents[2]
            _load_env_file(repository_root / ".env")
            try:
                run = build_live_run(arguments.config.resolve(), run_id=arguments.run_id)
            except ValueError as error:
                if arguments.command != "init" or not str(error).startswith("run already exists:"):
                    raise
                run = resume_live_run(arguments.config.resolve(), run_id=arguments.run_id)
            try:
                with RunLease(run.root):
                    result = run.execute_until("first_match_finalized")
            finally:
                close = getattr(run.runtime, "close", None)
                if callable(close):
                    close()
            payload = {
                "status": "checkpoint",
                "run_root": str(result.root),
                "candidate_id": result.metrics.candidate_id,
                "research_iteration": result.metrics.research_iteration,
                "match_id": result.match_id,
                "match_count": result.event_count("MatchFinalized"),
                "experience_count": result.event_count("ExperienceRecorded"),
                "resumable": (result.root / "checkpoint.json").is_file(),
            }
        elif arguments.group == "run" and arguments.command == "resume":
            from agentbench_hl.application.curriculum_service import CurriculumComplete
            from agentbench_hl.application.live_run import resume_live_run
            from agentbench_hl.application.run_lease import RunLease
            from agentbench_hl.application.run_service import advance_run

            if arguments.acts < 1:
                raise ValueError("--acts must be a positive integer")
            repository_root = arguments.config.resolve().parents[2]
            _load_env_file(repository_root / ".env")
            resumed = resume_live_run(arguments.config.resolve(), run_id=arguments.run_id)
            certification = None
            try:
                with RunLease(resumed.root):
                    try:
                        results = advance_run(resumed, acts=arguments.acts)
                    except CurriculumComplete:
                        results = ()
                        certification = resumed.certify_champion()
            finally:
                close = getattr(resumed.runtime, "close", None)
                if callable(close):
                    close()
            if certification is not None:
                payload = {
                    "status": "complete" if certification.passed else "checkpoint",
                    "run_root": str(resumed.root),
                    "acts_completed": 0,
                    "champion_id": certification.champion_id,
                    "certification_passed": certification.passed,
                    "certification_cases": certification.total_cases,
                    "certification_wins": certification.wins,
                    "certification_incomplete_count": len(certification.incomplete_cases),
                    "certification_failed_count": len(certification.failed_cases),
                    "resumable": not certification.passed,
                }
            else:
                last = results[-1]
                payload = {
                    "status": "checkpoint",
                    "run_root": str(resumed.root),
                    "acts_completed": len(results),
                    "candidate_id": last.version_id,
                    "parent_id": last.parent_id,
                    "target_id": last.target_id,
                    "selection": last.selection,
                    "research_iteration": last.metrics.research_iteration,
                    "resumable": (resumed.root / "checkpoint.json").is_file(),
                }
        elif arguments.group == "run" and arguments.command == "pursue":
            from agentbench_hl.application.curriculum_service import CurriculumComplete
            from agentbench_hl.application.live_run import resume_live_run
            from agentbench_hl.application.run_lease import RunLease

            repository_root = arguments.config.resolve().parents[2]
            _load_env_file(repository_root / ".env")
            pursued = resume_live_run(arguments.config.resolve(), run_id=arguments.run_id)
            acts_completed = 0
            certification = None
            try:
                with RunLease(pursued.root):
                    while certification is None or not certification.passed:
                        try:
                            pursued.advance_one_iteration()
                            acts_completed += 1
                        except CurriculumComplete:
                            certification = pursued.certify_champion()
            finally:
                close = getattr(pursued.runtime, "close", None)
                if callable(close):
                    close()
            payload = {
                "status": "complete",
                "run_root": str(pursued.root),
                "acts_completed": acts_completed,
                "champion_id": certification.champion_id,
                "certification_passed": certification.passed,
                "certification_cases": certification.total_cases,
                "certification_wins": certification.wins,
                "resumable": False,
            }
        elif arguments.group == "run" and arguments.command == "audit":
            payload = _audit(arguments.run_root.resolve())
        else:
            raise ValueError("unsupported command")
        print(json.dumps(_redact(payload), ensure_ascii=False, sort_keys=True))
        return 1 if payload.get("status") == "failed" else 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        payload = {"status": "error", "error": str(exc)}
        print(json.dumps(_redact(payload), ensure_ascii=False, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())

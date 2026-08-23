"""Stable, machine-readable AgentBench HL command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.config import repository_root_for
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
    drive = goal_led_commands.add_parser(
        "run", help="连续推进 N 轮迭代（服务化用；可中断可 resume）"
    )
    drive.add_argument("--config", type=Path, required=True)
    drive.add_argument("--run-id", required=True)
    drive.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "跑多少轮。不给则用配置里的 runtime.max_iterations；"
            "两者都不给 = 不限轮数（跑到 budget.tokens / budget.wall_seconds 耗尽）"
        ),
    )
    pool = groups.add_parser("pool")
    pool_commands = pool.add_subparsers(dest="command", required=True)
    pool_audit = pool_commands.add_parser("audit", help="审计某游戏选手池的可运行性")
    pool_audit.add_argument("game")
    pool_audit.add_argument("--agentbench-root", type=Path, default=None)
    pool_audit.add_argument(
        "--verify",
        action="store_true",
        help="实际打一局 self-play smoke 验证（写 players/runnable.json）",
    )
    pool_audit.add_argument("--parallel", type=int, default=4)
    pool_audit.add_argument("--attempts", type=int, default=2, help="任一次成功即判可用")
    pool_audit.add_argument("--cpus-per-match", type=int, default=4)
    pool_audit.add_argument(
        "--all", action="store_true", help="验证全部可运行选手（默认只验证带 Elo 的）"
    )
    pool_audit.add_argument("--work-root", type=Path, default=None)
    pool_audit.add_argument("--isolation", default="auto")
    ladder = groups.add_parser("ladder")
    ladder_commands = ladder.add_subparsers(dest="command", required=True)
    ladder_eval = ladder_commands.add_parser(
        "eval", help="全池实测评分：稀疏配对对战 + BT 拟合（可中断续跑）"
    )
    ladder_eval.add_argument("game")
    ladder_eval.add_argument("--agentbench-root", type=Path, default=None)
    ladder_eval.add_argument(
        "--scope",
        default="verified",
        choices=("verified", "ranked", "all"),
        help="参赛口径：审计通过 / 有 rank / 全部可运行",
    )
    ladder_eval.add_argument("--degree", type=int, default=6, help="每个选手大致对手数")
    ladder_eval.add_argument("--seeds", default="7", help="逗号分隔的 seed 列表")
    ladder_eval.add_argument("--parallel", type=int, default=4)
    ladder_eval.add_argument("--cpus-per-match", type=int, default=3)
    ladder_eval.add_argument("--timeout", type=float, default=900.0)
    ladder_eval.add_argument("--work-root", type=Path, default=None)
    ladder_eval.add_argument("--isolation", default="auto")
    ladder_eval.add_argument(
        "--plan-only", action="store_true", help="只打印赛程规模与成本估计，不跑对局"
    )
    metrics = groups.add_parser("metrics")
    metrics_commands = metrics.add_subparsers(dest="command", required=True)
    metrics_export = metrics_commands.add_parser(
        "export", help="导出 run 的迭代指标（供服务端/报告消费）"
    )
    metrics_export.add_argument("--run-root", type=Path, required=True)
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

            repository_root = repository_root_for(arguments.config)
            _load_env_file(repository_root / ".env")
            resume = arguments.command != "start"
            if arguments.command == "run":
                # 连续多轮：run 目录已存在则按 resume 装配。
                config_path = arguments.config.resolve()
                from agentbench_hl.config import ExperimentConfig

                config = ExperimentConfig.load(config_path)
                resume = (config.paths.runs_root / arguments.run_id / "run-manifest.json").is_file()
            service = build_goal_led_service(
                arguments.config.resolve(),
                run_id=arguments.run_id,
                resume=resume,
            )
            try:
                with RunLease(service.root):
                    if arguments.command == "run":
                        from agentbench_hl.application.goal_led_driver import drive

                        # 慢评测：evaluation.background_pool 为真时，起一个**独立**
                        # worker 进程，每 N 轮把中间版本拉去打完整个冻结人类池，
                        # 算它的真实池内 Elo 与名次。
                        #
                        # 为什么放在这里而不是 driver 里：worker 只写 pool-elo/，
                        # 主账本 events.jsonl 由迭代进程独占（两个进程追加同一个
                        # 账本会让对局记录交错，而这从曲线上看不出来）；
                        # 而且它自带 CPU 水位控制，忙时自己停下等，不拖慢迭代。
                        #
                        # 这个配置项以前**只被解析、没人消费**：写 true 什么也不会
                        # 发生，慢评测一直靠人手动敲 scripts/pool_elo_worker.py，
                        # 漏做的表现是 Elo 面板里少一条实测曲线（不报错）。
                        slow_eval = None
                        if config.evaluation.background_pool:
                            from agentbench_hl.application import slow_eval as slow

                            try:
                                plan = slow.build_plan(
                                    run_root=service.root,
                                    agentbench_root=config.paths.agentbench_root,
                                    game=config.game,
                                    repository_root=repository_root,
                                    seeds=config.evaluation.pool_seeds,
                                    stride=config.evaluation.pool_stride,
                                    challenger_track=config.evaluation.challenger_track,
                                )
                                slow_eval = slow.spawn(plan)
                                print(
                                    json.dumps(
                                        {
                                            "status": (
                                                "slow_eval_started"
                                                if slow_eval is not None
                                                # 已经有 worker 在跑（例如之前用
                                                # attach_slow_eval.sh 手动挂过）。
                                                # 不重复起：两个 worker 并发写同一个
                                                # pool-elo/ 会让对局重复调度、
                                                # 结果文件互相覆盖，且不报错。
                                                else "slow_eval_already_running"
                                            ),
                                            "pid": (
                                                slow_eval.pid
                                                if slow_eval is not None
                                                else slow.already_running(service.root)
                                            ),
                                            **plan.as_dict(),
                                        },
                                        ensure_ascii=False,
                                    ),
                                    file=sys.stderr,
                                )
                            except (OSError, FileNotFoundError) as error:
                                # 慢评测起不来**不能**打死迭代：它只是观测通道。
                                # 但必须吼出来，否则就变成"图上静静地少一条线"。
                                print(
                                    json.dumps(
                                        {
                                            "status": "slow_eval_failed",
                                            "error": f"{type(error).__name__}: {error}",
                                            "impact": "迭代继续，但不会有全池实测 Elo 曲线",
                                        },
                                        ensure_ascii=False,
                                    ),
                                    file=sys.stderr,
                                )

                        # 轮数优先级：CLI --iterations > config runtime.max_iterations >
                        # 无限（None）。这样 max_iterations 才真正有消费点——它以前
                        # 只被解析、没人读，配了也不生效。
                        # None = 不限轮数，跑到预算耗尽为止（实验 2 需要）。
                        iterations = arguments.iterations
                        if iterations is None:
                            iterations = config.runtime.max_iterations
                        try:
                            result = drive(service, iterations=iterations)
                        finally:
                            if slow_eval is not None and slow_eval.poll() is None:
                                # 迭代结束后给慢评测一段时间把队列里剩下的版本测完
                                # （它落后于迭代是常态），超时再收掉。
                                # 不等的话最后几个版本会永远缺数据，而那几个恰好是
                                # 最强的版本——曲线的尾巴断在最有价值的地方。
                                try:
                                    slow_eval.wait(timeout=1800)
                                except subprocess.TimeoutExpired:
                                    slow_eval.terminate()
                        payload = {
                            "status": "finished",
                            "run_root": str(service.root),
                            **result.as_dict(),
                        }
                        outcome = result.last
                    else:
                        outcome = (
                            service.start()
                            if arguments.command == "start"
                            else service.advance()
                        )
                        payload = {
                            "status": "checkpoint",
                            "run_root": str(service.root),
                            "thread_id": outcome.thread_id,
                            "workspace": str(outcome.workspace),
                            "request_id": outcome.request_id,
                            "match_count": outcome.match_count,
                        }
            finally:
                close = getattr(service.runtime, "close", None)
                if callable(close):
                    close()
        elif arguments.group == "pool" and arguments.command == "audit":
            from agentbench_hl.adapters.contract.factory import (
                _supports_compiled_players,
                game_roles,
            )
            from agentbench_hl.adapters.contract.pool import (
                load_pool,
                ranked_ladder,
                runnable_players,
            )

            root = (
                arguments.agentbench_root.resolve()
                if arguments.agentbench_root is not None
                else Path(os.environ.get("AGENTBENCH_ROOT", ".")).resolve()
            )
            players = load_pool(
                root,
                arguments.game,
                supports_compiled=_supports_compiled_players(root, arguments.game),
            )
            runnable = runnable_players(players)
            ladder = ranked_ladder(players)
            payload = {
                "status": "ok",
                "game": arguments.game,
                "roles": list(game_roles(root, arguments.game)),
                "players_total": len(players),
                "players_runnable": len(runnable),
                "players_ranked_runnable": len(ladder),
                "top_runnable": [
                    {"player_id": item.player_id, "rank": item.rank, "elo": item.elo}
                    for item in ladder[:10]
                ],
                "unrunnable_examples": [
                    {"player_id": item.player_id, "reason": item.exclusion_diagnostic}
                    for item in players
                    if not item.runnable
                ][:5],
            }
            if arguments.verify:
                from agentbench_hl.application.pool_audit import audit_pool

                work_root = (
                    arguments.work_root.resolve()
                    if arguments.work_root is not None
                    else Path("/tmp") / f"abhl-pool-audit-{arguments.game}"
                )
                report = audit_pool(
                    arguments.game,
                    root,
                    work_root=work_root,
                    ranked_only=not arguments.all,
                    parallel=max(1, arguments.parallel),
                    attempts=max(1, arguments.attempts),
                    cpus_per_match=max(2, arguments.cpus_per_match),
                    isolation_backend=arguments.isolation,
                )
                payload["audit"] = {
                    "scope": report["scope"],
                    "checked": report["checked"],
                    "verified": report["verified"],
                    "written_to": report.get("written_to"),
                    "rejected": [
                        {"player_id": row["player_id"], "reason": row["diagnostic"]}
                        for row in report["rows"]  # type: ignore[union-attr]
                        if isinstance(row, dict) and not row.get("verified")
                    ],
                }
        elif arguments.group == "ladder" and arguments.command == "eval":
            from agentbench_hl.adapters.contract.factory import (
                _supports_compiled_players,
                game_roles,
            )
            from agentbench_hl.adapters.contract.pool import load_pool
            from agentbench_hl.application.ladder_eval import (
                _select_players,
                build_plan,
                run_ladder,
            )

            root = (
                arguments.agentbench_root.resolve()
                if arguments.agentbench_root is not None
                else Path(os.environ.get("AGENTBENCH_ROOT", ".")).resolve()
            )
            seeds = tuple(
                int(item) for item in str(arguments.seeds).split(",") if item.strip()
            ) or (7,)
            work_root = (
                arguments.work_root.resolve()
                if arguments.work_root is not None
                else Path("/tmp") / f"abhl-ladder-{arguments.game}"
            )
            if arguments.plan_only:
                pool_players = load_pool(
                    root,
                    arguments.game,
                    supports_compiled=_supports_compiled_players(root, arguments.game),
                )
                ids, note = _select_players(root, arguments.game, pool_players, arguments.scope)
                plan = build_plan(
                    arguments.game,
                    ids,
                    game_roles(root, arguments.game),
                    degree=arguments.degree,
                    seeds=seeds,
                )
                payload = {
                    "status": "ok",
                    "game": arguments.game,
                    "scope": arguments.scope,
                    "scope_note": note,
                    "players": len(plan.players),
                    "planned_matches": plan.total,
                    "seeds": list(seeds),
                    "degree": arguments.degree,
                }
            else:
                report = run_ladder(
                    arguments.game,
                    root,
                    work_root=work_root,
                    scope=arguments.scope,
                    degree=max(1, arguments.degree),
                    seeds=seeds,
                    parallel=max(1, arguments.parallel),
                    cpus_per_match=max(2, arguments.cpus_per_match),
                    timeout_s=arguments.timeout,
                    isolation_backend=arguments.isolation,
                )
                payload = {
                    "status": "ok",
                    "game": arguments.game,
                    "scope": report["scope"],
                    "players": report["players"],
                    "planned_matches": report["planned_matches"],
                    "played_matches": report["played_matches"],
                    "rated_players": report["rated_players"],
                    "anchor_alignment": report["anchor_alignment"],
                    "written_to": report.get("written_to"),
                    "top": [
                        row
                        for row in report["ratings"][:10]  # type: ignore[index]
                    ],
                }
        elif arguments.group == "metrics" and arguments.command == "export":
            run_root = arguments.run_root.resolve()
            store = JsonlEventStore(run_root / "events.jsonl")
            rows = [
                dict(event.payload)
                for event in store.read_all()
                if event.event_type == "IterationMetricsFinalized"
            ]
            payload = {
                "status": "ok",
                "run_root": str(run_root),
                "schema_version": "1.1",
                "iterations": rows,
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

            repository_root = repository_root_for(arguments.config)
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
            repository_root = repository_root_for(arguments.config)
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

            repository_root = repository_root_for(arguments.config)
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

"""Assemble the real frozen AntWar2 arena and isolated Codex Goal runtime.

This is the AntWar2 composition root.  All AntWar2-specific wiring lives here so
the framework core (``application`` / ``registry``) never imports this adapter
directly; it is reached only through the game registry.
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import yaml

from agentbench_hl.adapters.antwar2.arena import AntWar2Arena
from agentbench_hl.adapters.antwar2.policy_probe import probe_policy_episode
from agentbench_hl.adapters.antwar2.replay import decode_replay
from agentbench_hl.adapters.antwar2.runtime import (
    AntWar2Layout,
    Opponent,
    audit_human_pool,
    build_backend,
    materialize_bootstrap,
    sha256_file,
    tree_sha256,
)
from agentbench_hl.adapters.antwar2.smoke import verify_smoke
from agentbench_hl.adapters.codex_goal.app_server import CodexGoalRuntime
from agentbench_hl.adapters.codex_goal.read_isolation import (
    write_candidate_isolation_profile,
)
from agentbench_hl.adapters.filesystem.artifact_store import FilesystemArtifactStore
from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.application.live_run import (
    codex_goal_runtime,
    goal_app_server_command,
    probe_codex_installation,
    read_json_object,
    use_responses_compat_proxy,
    verify_codex_identity,
)
from agentbench_hl.application.run_service import RunService
from agentbench_hl.config import EvaluatorConfig, ExperimentConfig
from agentbench_hl.ports.arena import ProcessSpec

GAME = "antwar2"


def official_human_ratings(opponents: tuple[Opponent, ...]) -> dict[str, float]:
    runnable = tuple(item for item in opponents if item.runnable)
    missing = tuple(item.opponent_id for item in runnable if item.score is None)
    if missing:
        raise ValueError(f"frozen ladder has no score for runnable opponents: {missing}")
    return {item.opponent_id: float(item.score) for item in runnable if item.score is not None}


def _gamepack_sdk_hash(gamepack_root: Path) -> str:
    manifest = yaml.safe_load((gamepack_root / "manifest.yaml").read_text(encoding="utf-8"))
    value = manifest.get("public_sdk_tree_sha256") if isinstance(manifest, dict) else None
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("GamePack has no frozen public SDK hash")
    return value


def _certification_config_path(gamepack_root: Path) -> Path:
    return gamepack_root / "evaluator" / "certification.yaml"


class AntWar2AdapterFactory:
    """Game registry factory that assembles AntWar2 live runs."""

    game = GAME

    def build_run(self, config_path: str | Path, *, run_id: str) -> RunService:
        return build_live_run(config_path, run_id=run_id)

    def resume_run(self, config_path: str | Path, *, run_id: str) -> RunService:
        return resume_live_run(config_path, run_id=run_id)


def build_live_run(
    config_path: str | Path,
    *,
    run_id: str,
) -> RunService:
    """Build one fresh run without exposing a credential or hidden human root."""

    source = Path(config_path).resolve()
    config = ExperimentConfig.load(source)
    secret = config.secret_environment()[config.provider.api_key_env]
    repository_root = source.parents[2]
    gamepack_root = repository_root / "gamepacks" / config.game
    evaluator_config_path = _certification_config_path(gamepack_root)
    evaluator_root = evaluator_config_path.parent
    evaluator = EvaluatorConfig.load(evaluator_config_path)
    run_root = config.paths.runs_root / run_id
    if (run_root / "run-manifest.json").exists():
        raise ValueError(f"run already exists: {run_id}")
    run_root.mkdir(parents=True, exist_ok=True)

    codex_probe = probe_codex_installation(
        config.runtime.codex_binary,
        run_root / "provider-preflight",
    )

    layout = AntWar2Layout.from_root(
        config.paths.agentbench_root,
        run_root / "frozen-build",
    )
    layout.validate(expected_sdk_sha256=_gamepack_sdk_hash(gamepack_root))
    backend = build_backend(layout)
    pool = audit_human_pool(layout)
    runnable = tuple(item for item in pool if item.runnable)
    if not runnable:
        raise ValueError("frozen human pool has no runnable opponent")
    target = max(runnable, key=lambda item: item.rank)

    bootstrap = run_root / "bootstrap"
    materialize_bootstrap(
        layout,
        gamepack_root / "candidate_support",
        bootstrap,
    )
    opponents = {
        item.opponent_id: ProcessSpec(item.entry_command, item.package_root)
        for item in runnable
        if item.entry_command is not None
    }
    candidate_profile = write_candidate_isolation_profile(
        run_root / "candidate-runtime.sb",
        denied_read_roots=(
            layout.human_manifest.parent,
            evaluator_root,
            repository_root / ".env",
            run_root / "codex-home",
            run_root / "hidden-certification",
        ),
    )
    candidate_command_prefix = (
        "/usr/bin/sandbox-exec",
        "-f",
        str(candidate_profile),
    )
    arena = AntWar2Arena(
        game=ProcessSpec((str(backend.executable),), backend.executable.parents[1]),
        opponents=opponents,
        artifact_root=run_root / "official-matches",
        timeout_s=180.0,
        candidate_command_prefix=candidate_command_prefix,
    )
    certification_arena = AntWar2Arena(
        game=ProcessSpec((str(backend.executable),), backend.executable.parents[1]),
        opponents=opponents,
        artifact_root=run_root / "hidden-certification/matches",
        timeout_s=180.0,
        candidate_command_prefix=candidate_command_prefix,
    )
    runtime = codex_goal_runtime(config, secret, codex_home=run_root / "codex-home")
    frozen_config = config.frozen_dict()
    (run_root / "run-config.json").write_text(
        json.dumps(frozen_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_root / "resource-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "backend_archive_sha256": backend.archive_sha256,
                "backend_executable_sha256": backend.executable_sha256,
                "public_sdk_tree_sha256": tree_sha256(layout.public_sdk_root),
                "codex_version": codex_probe.version,
                "codex_goals_enabled": codex_probe.goals_enabled,
                "codex_schema_tree_sha256": codex_probe.schema_tree_sha256,
                "codex_schema_hash_kind": "canonical-json-v1",
                "codex_schema_file_count": codex_probe.schema_file_count,
                "certification_config_sha256": sha256_file(evaluator_config_path),
                "target": target.opponent_id,
                "human_pool": [
                    {
                        "opponent_id": item.opponent_id,
                        "rank": item.rank,
                        "archive_sha256": item.archive_sha256,
                        "runnable": item.runnable,
                        "exclusion_diagnostic": item.exclusion_diagnostic,
                    }
                    for item in pool
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    human_ratings = official_human_ratings(pool)
    return RunService(
        run_root=run_root,
        bootstrap_root=bootstrap,
        gamepack_root=gamepack_root,
        human_pool_root=layout.human_extracted_root,
        evaluator_root=evaluator_root,
        event_store=JsonlEventStore(run_root / "events.jsonl"),
        certification_event_store=JsonlEventStore(run_root / "hidden-certification/events.jsonl"),
        artifact_store=FilesystemArtifactStore(run_root / "candidates"),
        runtime=runtime,
        arena=arena,
        certification_arena=certification_arena,
        opponent_id=target.opponent_id,
        role="P0",
        seed=config.curriculum.development_seeds[0],
        human_ratings=human_ratings,
        epsilon=config.measurement.epsilon,
        model=config.provider.model,
        model_provider="OpenAI",
        candidate_validator=partial(
            verify_smoke,
            command_prefix=candidate_command_prefix,
        ),
        opponents=pool,
        development_roles=("P0", "P1"),
        development_seeds=config.curriculum.development_seeds,
        backend_hash=backend.executable_sha256,
        opponent_hashes={item.opponent_id: item.archive_sha256 for item in runnable},
        policy_probe=partial(
            probe_policy_episode,
            command_prefix=candidate_command_prefix,
        ),
        replay_decoder=decode_replay,
        certification_roles=tuple(evaluator.roles),
        certification_seeds=evaluator.certification_seeds,
        required_win_rate=0.5,
    )


def resume_live_run(
    config_path: str | Path,
    *,
    run_id: str,
) -> RunService:
    """Reassemble frozen external resources and resume one existing Goal run."""

    source = Path(config_path).resolve()
    config = ExperimentConfig.load(source)
    repository_root = source.parents[2]
    gamepack_root = repository_root / "gamepacks" / config.game
    evaluator_config_path = _certification_config_path(gamepack_root)
    evaluator_root = evaluator_config_path.parent
    run_root = config.paths.runs_root / run_id
    if not (run_root / "run-manifest.json").is_file():
        raise FileNotFoundError(f"run manifest not found: {run_root / 'run-manifest.json'}")
    evaluator = EvaluatorConfig.load(evaluator_config_path)
    frozen_config = read_json_object(run_root / "run-config.json", "run config")
    if frozen_config != config.frozen_dict():
        raise ValueError("current experiment config differs from frozen run config")
    resources = read_json_object(run_root / "resource-manifest.json", "resource manifest")

    codex_probe = probe_codex_installation(
        config.runtime.codex_binary,
        run_root / "provider-resume-preflight",
    )
    if not verify_codex_identity(resources, codex_probe):
        raise ValueError("Codex App Server identity differs from frozen run")

    layout = AntWar2Layout.from_root(
        config.paths.agentbench_root,
        run_root / "frozen-build",
    )
    layout.validate(expected_sdk_sha256=_gamepack_sdk_hash(gamepack_root))
    backend = build_backend(layout)
    pool = audit_human_pool(layout)
    runnable = tuple(item for item in pool if item.runnable)
    frozen_pool = [
        {
            "opponent_id": item.opponent_id,
            "rank": item.rank,
            "archive_sha256": item.archive_sha256,
            "runnable": item.runnable,
            "exclusion_diagnostic": item.exclusion_diagnostic,
        }
        for item in pool
    ]
    resource_identity = {
        "backend_archive_sha256": backend.archive_sha256,
        "backend_executable_sha256": backend.executable_sha256,
        "public_sdk_tree_sha256": tree_sha256(layout.public_sdk_root),
        "human_pool": frozen_pool,
        "certification_config_sha256": sha256_file(evaluator_config_path),
    }
    if any(resources.get(key) != value for key, value in resource_identity.items()):
        raise ValueError("AntWar2 resource identity differs from frozen run")

    opponents = {
        item.opponent_id: ProcessSpec(item.entry_command, item.package_root)
        for item in runnable
        if item.entry_command is not None
    }
    candidate_profile = write_candidate_isolation_profile(
        run_root / "candidate-runtime.sb",
        denied_read_roots=(
            layout.human_manifest.parent,
            evaluator_root,
            repository_root / ".env",
            run_root / "codex-home",
            run_root / "hidden-certification",
        ),
    )
    candidate_command_prefix = (
        "/usr/bin/sandbox-exec",
        "-f",
        str(candidate_profile),
    )
    arena = AntWar2Arena(
        game=ProcessSpec((str(backend.executable),), backend.executable.parents[1]),
        opponents=opponents,
        artifact_root=run_root / "official-matches",
        timeout_s=180.0,
        candidate_command_prefix=candidate_command_prefix,
    )
    certification_arena = AntWar2Arena(
        game=ProcessSpec((str(backend.executable),), backend.executable.parents[1]),
        opponents=opponents,
        artifact_root=run_root / "hidden-certification/matches",
        timeout_s=180.0,
        candidate_command_prefix=candidate_command_prefix,
    )
    secret = config.secret_environment()[config.provider.api_key_env]
    runtime = codex_goal_runtime(config, secret, codex_home=run_root / "codex-home")
    return RunService.resume(
        run_root,
        runtime=runtime,
        arena=arena,
        certification_arena=certification_arena,
        candidate_validator=partial(
            verify_smoke,
            command_prefix=candidate_command_prefix,
        ),
        opponents=pool,
        policy_probe=partial(
            probe_policy_episode,
            command_prefix=candidate_command_prefix,
        ),
        replay_decoder=decode_replay,
        certification_roles=tuple(evaluator.roles),
        certification_seeds=evaluator.certification_seeds,
        required_win_rate=0.5,
    )

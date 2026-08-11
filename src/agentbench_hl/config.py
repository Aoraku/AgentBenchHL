"""Typed, secret-safe experiment configuration."""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import yaml

_ENV_SCALAR = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _seeds(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{name} must contain integers")
    seeds = tuple(value)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must be unique")
    return seeds


def _path(value: object, name: str, *, base: Path, env: Mapping[str, str]) -> Path:
    raw = _text(value, name)
    match = _ENV_SCALAR.fullmatch(raw)
    if match:
        variable = match.group(1)
        if variable not in env or not env[variable]:
            raise ValueError(f"missing environment variable {variable} for {name}")
        raw = env[variable]
    elif "${" in raw:
        raise ValueError(f"{name} only supports a complete ${{NAME}} scalar")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class ProviderConfig:
    model: str
    reasoning_effort: str
    base_url: str
    api_key_env: str
    disable_response_storage: bool


@dataclass(frozen=True)
class RuntimeConfig:
    codex_binary: str
    branch_width: int
    max_iterations: int | None
    network_access: Literal["disabled"]


@dataclass(frozen=True)
class PathConfig:
    agentbench_root: Path
    runs_root: Path


@dataclass(frozen=True)
class CurriculumConfig:
    order: Literal["lowest_rank_first"]
    development_seeds: tuple[int, ...]


@dataclass(frozen=True)
class MeasurementConfig:
    epsilon: float


@dataclass(frozen=True)
class EvaluatorConfig:
    certification_seeds: tuple[int, ...]
    roles: tuple[str, ...]

    @classmethod
    def load(cls, path: Path) -> EvaluatorConfig:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = _mapping(value, "evaluator config")
        if root.get("schema_version") != "1.0":
            raise ValueError("evaluator schema_version must be 1.0")
        raw_roles = root.get("roles")
        if (
            not isinstance(raw_roles, list)
            or not raw_roles
            or any(not isinstance(role, str) or not role for role in raw_roles)
            or len(set(raw_roles)) != len(raw_roles)
        ):
            raise ValueError("roles must be unique non-empty strings")
        return cls(
            certification_seeds=_seeds(root.get("certification_seeds"), "certification_seeds"),
            roles=tuple(str(role) for role in raw_roles),
        )


_SAFE_GAME = re.compile(r"[a-z0-9][a-z0-9_]{0,63}\Z")


def _gamepacks_root(config_path: Path) -> Path:
    """Locate the repository ``gamepacks/`` directory from a config file path.

    Experiment configs live under ``configs/experiments/<name>.yaml`` inside the
    repository, so the repository root is three levels up from the config file.
    """

    return config_path.resolve().parents[2] / "gamepacks"


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: Literal["1.0"]
    game: str
    origin: Literal["from_scratch"]
    provider: ProviderConfig
    runtime: RuntimeConfig
    paths: PathConfig
    curriculum: CurriculumConfig
    measurement: MeasurementConfig
    _environment: Mapping[str, str] = field(repr=False, compare=False)

    @classmethod
    def load(
        cls,
        path: Path,
        env: Mapping[str, str] | None = None,
        *,
        gamepacks_root: Path | None = None,
    ) -> ExperimentConfig:
        environment = dict(os.environ if env is None else env)
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = _mapping(value, "experiment config")
        if root.get("origin") != "from_scratch":
            raise ValueError("origin must be from_scratch")
        if root.get("schema_version") != "1.0":
            raise ValueError("schema_version must be 1.0")
        game = _text(root.get("game"), "game")
        if not _SAFE_GAME.fullmatch(game):
            raise ValueError("game must be a lowercase [a-z0-9_] identifier")
        packs_root = gamepacks_root if gamepacks_root is not None else _gamepacks_root(path)
        if not (packs_root / game).is_dir():
            raise ValueError(f"no GamePack registered for game {game!r} under {packs_root}")

        provider = _mapping(root.get("provider"), "provider")
        runtime = _mapping(root.get("runtime"), "runtime")
        paths = _mapping(root.get("paths"), "paths")
        curriculum = _mapping(root.get("curriculum"), "curriculum")
        measurement = _mapping(root.get("measurement"), "measurement")

        branch_width = runtime.get("branch_width")
        if isinstance(branch_width, bool) or not isinstance(branch_width, int):
            raise ValueError("runtime.branch_width must be an integer")
        if branch_width < 1:
            raise ValueError("runtime.branch_width must be positive")
        max_iterations = runtime.get("max_iterations")
        if max_iterations is not None and (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("runtime.max_iterations must be null or positive")
        if runtime.get("network_access") != "disabled":
            raise ValueError("runtime.network_access must be disabled")
        if curriculum.get("order") != "lowest_rank_first":
            raise ValueError("curriculum.order must be lowest_rank_first")
        epsilon = measurement.get("epsilon")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError("measurement.epsilon must be numeric")
        if not 0 < float(epsilon) < 1:
            raise ValueError("measurement.epsilon must be between zero and one")

        return cls(
            schema_version="1.0",
            game=game,
            origin="from_scratch",
            provider=ProviderConfig(
                model=_text(provider.get("model"), "provider.model"),
                reasoning_effort=_text(
                    provider.get("reasoning_effort"), "provider.reasoning_effort"
                ),
                base_url=_text(provider.get("base_url"), "provider.base_url"),
                api_key_env=_text(provider.get("api_key_env"), "provider.api_key_env"),
                disable_response_storage=_bool(
                    provider.get("disable_response_storage"),
                    "provider.disable_response_storage",
                ),
            ),
            runtime=RuntimeConfig(
                codex_binary=_text(runtime.get("codex_binary"), "runtime.codex_binary"),
                branch_width=branch_width,
                max_iterations=max_iterations,
                network_access="disabled",
            ),
            paths=PathConfig(
                agentbench_root=_path(
                    paths.get("agentbench_root"),
                    "paths.agentbench_root",
                    base=path.parent,
                    env=environment,
                ),
                runs_root=_path(
                    paths.get("runs_root"),
                    "paths.runs_root",
                    base=path.parent,
                    env=environment,
                ),
            ),
            curriculum=CurriculumConfig(
                order="lowest_rank_first",
                development_seeds=_seeds(
                    curriculum.get("development_seeds"),
                    "curriculum.development_seeds",
                ),
            ),
            measurement=MeasurementConfig(epsilon=float(epsilon)),
            _environment=environment,
        )

    def secret_environment(self) -> dict[str, str]:
        name = self.provider.api_key_env
        value = self._environment.get(name)
        if not value:
            raise ValueError(f"missing API key environment variable {name}")
        return {name: value}

    def frozen_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "game": self.game,
            "origin": self.origin,
            "provider": dataclasses.asdict(self.provider),
            "runtime": dataclasses.asdict(self.runtime),
            "paths": {
                "agentbench_root": str(self.paths.agentbench_root),
                "runs_root": str(self.paths.runs_root),
            },
            "curriculum": {
                "order": self.curriculum.order,
                "development_seeds": list(self.curriculum.development_seeds),
            },
            "measurement": dataclasses.asdict(self.measurement),
        }

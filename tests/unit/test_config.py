from pathlib import Path

import pytest

from agentbench_hl.config import EvaluatorConfig, ExperimentConfig

VALID_CONFIG = """
schema_version: "1.0"
game: antwar2
origin: from_scratch
provider:
  model: gpt-5.5
  reasoning_effort: xhigh
  base_url: https://example.invalid/responses
  api_key_env: ABHL_API_KEY
  disable_response_storage: true
runtime:
  codex_binary: codex
  branch_width: 1
  max_iterations: null
  network_access: disabled
paths:
  agentbench_root: ${AB_ROOT}
  runs_root: ./runs
curriculum:
  order: lowest_rank_first
  development_seeds: [1, 2]
measurement:
  epsilon: 0.01
"""


def test_config_expands_public_paths_but_never_serializes_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(VALID_CONFIG, encoding="utf-8")

    config = ExperimentConfig.load(
        path,
        env={"AB_ROOT": "/bench", "ABHL_API_KEY": "secret-value"},
    )

    assert config.paths.agentbench_root == Path("/bench")
    assert config.paths.runs_root == (tmp_path / "runs").resolve()
    assert config.runtime.max_iterations is None
    assert config.secret_environment() == {"ABHL_API_KEY": "secret-value"}
    assert "secret-value" not in repr(config.frozen_dict())


def test_config_rejects_non_from_scratch_origin(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(VALID_CONFIG.replace("from_scratch", "v239"), encoding="utf-8")

    with pytest.raises(ValueError, match="from_scratch"):
        ExperimentConfig.load(path, env={"AB_ROOT": "/bench"})


def test_evaluator_config_is_separate_from_goal_visible_config(
    tmp_path: Path,
) -> None:
    path = tmp_path / "certification.yaml"
    path.write_text(
        "schema_version: '1.0'\ncertification_seeds: [11, 12, 13]\nroles: [P0, P1]\n",
        encoding="utf-8",
    )

    config = EvaluatorConfig.load(path)

    assert config.certification_seeds == (11, 12, 13)
    assert config.roles == ("P0", "P1")

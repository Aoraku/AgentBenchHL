from pathlib import Path

import pytest

from agentbench_hl.config import (
    DEFAULT_MAX_ITERATIONS,
    UNBOUNDED_MAX_ITERATIONS,
    EvaluatorConfig,
    ExperimentConfig,
)

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


def _gamepacks_with(tmp_path: Path, *games: str) -> Path:
    packs = tmp_path / "gamepacks"
    for game in games:
        (packs / game).mkdir(parents=True, exist_ok=True)
    return packs


def test_config_expands_public_paths_but_never_serializes_key(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(VALID_CONFIG, encoding="utf-8")

    config = ExperimentConfig.load(
        path,
        env={"AB_ROOT": "/bench", "ABHL_API_KEY": "secret-value"},
        gamepacks_root=_gamepacks_with(tmp_path, "antwar2"),
    )

    assert config.game == "antwar2"
    assert config.paths.agentbench_root == Path("/bench")
    assert config.paths.runs_root == (tmp_path / "runs").resolve()
    # max_iterations: null 不再表示"无限"。无限跑靠的是 token 预算，而 token
    # 记账修好之前那个守卫是失效的（见 goal_led_service._token_total 详注），
    # 于是"不设上限"实际等于"跑到人手动杀掉"，不同 run 的轮数不可比。
    # 现在 null = 32 轮，"不设上限"是一个显式的、可复现的数字（128）。
    assert config.runtime.max_iterations == DEFAULT_MAX_ITERATIONS
    assert config.secret_environment() == {"ABHL_API_KEY": "secret-value"}
    assert "secret-value" not in repr(config.frozen_dict())


def test_unbounded_iterations_mean_a_concrete_number(tmp_path: Path) -> None:
    """"不设上限"必须落成一个具体数字，否则实验不可比也不可复现。"""

    path = tmp_path / "experiment.yaml"
    path.write_text(
        VALID_CONFIG.replace("max_iterations: null", "max_iterations: unbounded"),
        encoding="utf-8",
    )

    config = ExperimentConfig.load(
        path,
        env={"AB_ROOT": "/bench", "ABHL_API_KEY": "secret-value"},
        gamepacks_root=_gamepacks_with(tmp_path, "antwar2"),
    )

    assert config.runtime.max_iterations == UNBOUNDED_MAX_ITERATIONS == 128


def test_explicit_iteration_count_wins(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        VALID_CONFIG.replace("max_iterations: null", "max_iterations: 8"), encoding="utf-8"
    )

    config = ExperimentConfig.load(
        path,
        env={"AB_ROOT": "/bench", "ABHL_API_KEY": "secret-value"},
        gamepacks_root=_gamepacks_with(tmp_path, "antwar2"),
    )

    assert config.runtime.max_iterations == 8


def test_defaults_are_k1_batch4_and_no_information_gain(tmp_path: Path) -> None:
    """默认配置的四件事：k=1、b=4、progress 课程、IG 全关。"""

    path = tmp_path / "experiment.yaml"
    path.write_text(VALID_CONFIG, encoding="utf-8")

    config = ExperimentConfig.load(
        path,
        env={"AB_ROOT": "/bench", "ABHL_API_KEY": "secret-value"},
        gamepacks_root=_gamepacks_with(tmp_path, "antwar2"),
    )

    assert config.runtime.rollout_k == 1
    assert config.curriculum.batch == 4
    assert config.curriculum.opponent_policy == "progress"
    assert config.measurement.information_gain is False
    assert config.measurement.behavioral_ig_cases == 0


def test_model_profile_supplies_endpoint_and_key_env(tmp_path: Path) -> None:
    """模型配置独立成文件：实验里只写一个 profile 名字。

    7 个模型的中转站与 key 各不相同，散在每个实验配置里必然漏字段——
    漏掉 context_window 的后果是 codex 用兜底元数据、压缩在看不见的点触发
    并打死整个 run。
    """

    profiles = tmp_path / "models"
    profiles.mkdir()
    (profiles / "glm-5.3.yaml").write_text(
        "model: glm-5.3\n"
        "base_url: https://relay.invalid/sub2api\n"
        "api_key_env: ABHL_KEY_GLM\n"
        "reasoning_effort: high\n"
        "context_window: 204800\n"
        "disable_response_storage: true\n",
        encoding="utf-8",
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(
        VALID_CONFIG.replace(
            "provider:\n"
            "  model: gpt-5.5\n"
            "  reasoning_effort: xhigh\n"
            "  base_url: https://example.invalid/responses\n"
            "  api_key_env: ABHL_API_KEY\n"
            "  disable_response_storage: true\n",
            "provider:\n  model_profile: glm-5.3\n",
        ),
        encoding="utf-8",
    )

    config = ExperimentConfig.load(
        path,
        env={"AB_ROOT": "/bench", "ABHL_KEY_GLM": "glm-secret"},
        gamepacks_root=_gamepacks_with(tmp_path, "antwar2"),
        model_profiles_root=profiles,
    )

    assert config.provider.model == "glm-5.3"
    assert config.provider.base_url == "https://relay.invalid/sub2api"
    assert config.provider.context_window == 204800
    assert config.secret_environment() == {"ABHL_KEY_GLM": "glm-secret"}


def test_unknown_model_profile_lists_the_available_ones(tmp_path: Path) -> None:
    profiles = tmp_path / "models"
    profiles.mkdir()
    (profiles / "glm-5.3.yaml").write_text(
        "model: glm-5.3\nbase_url: https://relay.invalid\napi_key_env: K\n", encoding="utf-8"
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(
        VALID_CONFIG.replace("  model: gpt-5.5", "  model_profile: no-such-model"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="glm-5.3"):
        ExperimentConfig.load(
            path,
            env={"AB_ROOT": "/bench", "ABHL_API_KEY": "x"},
            gamepacks_root=_gamepacks_with(tmp_path, "antwar2"),
            model_profiles_root=profiles,
        )


def test_config_accepts_any_registered_gamepack(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(VALID_CONFIG.replace("game: antwar2", "game: chess"), encoding="utf-8")

    config = ExperimentConfig.load(
        path,
        env={"AB_ROOT": "/bench", "ABHL_API_KEY": "secret-value"},
        gamepacks_root=_gamepacks_with(tmp_path, "chess"),
    )

    assert config.game == "chess"


def test_config_rejects_game_without_gamepack(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(VALID_CONFIG, encoding="utf-8")

    with pytest.raises(ValueError, match="no GamePack registered"):
        ExperimentConfig.load(
            path,
            env={"AB_ROOT": "/bench", "ABHL_API_KEY": "secret-value"},
            gamepacks_root=_gamepacks_with(tmp_path, "snakego"),
        )


def test_config_rejects_non_from_scratch_origin(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(VALID_CONFIG.replace("from_scratch", "v239"), encoding="utf-8")

    with pytest.raises(ValueError, match="from_scratch"):
        ExperimentConfig.load(
            path,
            env={"AB_ROOT": "/bench"},
            gamepacks_root=_gamepacks_with(tmp_path, "antwar2"),
        )


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

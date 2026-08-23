"""续跑时哪些配置改动可以放行。

这道校验本身是对的：中途改模型、改 k、改对手策略会让前后轮次不可比，
而事故只会在事后画图时才显现。

但它原来把**观测通道**和**实验变量**一视同仁，于是"打开慢评测"这种完全不碰
迭代的改动也会让续跑直接失败：

    ContractFactoryError: current experiment config differs from frozen run config

报错还不说哪个字段变了，只能靠人 diff 两份 JSON。实测 random 组因此无法续跑
最后 6 轮 —— 而它死于第 26 轮的 checkpoint 超时，本该一条命令救回来。

判据：**这个字段会不会改变 agent 看到的东西或它的对局？**
"""

from __future__ import annotations

from agentbench_hl.adapters.contract.factory import _significant_config_changes


def _config(**overrides: object) -> dict:
    base = {
        "game": "antwar2",
        "provider": {"model": "gpt-5.6-sol", "reasoning_effort": "high"},
        "runtime": {
            "rollout_k": 1,
            "max_iterations": 32,
            "match_parallelism": 8,
            "match_timeout_s": 1800,
        },
        "curriculum": {"opponent_policy": "progress", "batch": 4},
        "goal": {"history_mode": "full"},
        "evaluation": {"background_pool": False, "pool_stride": 3},
        "budget": {"tokens": None},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


def test_identical_config_resumes() -> None:
    assert _significant_config_changes(_config(), _config()) == []


def test_turning_on_slow_eval_is_allowed() -> None:
    """慢评测是**观测通道**：另一个进程、另建对局、只写 pool-elo/。

    它不改变 agent 看到的任何东西，所以打开它必须能续跑。
    这正是 random 组卡住的那个改动。
    """

    after = _config(evaluation={"background_pool": True, "pool_stride": 3})

    assert _significant_config_changes(_config(), after) == []


def test_extending_the_iteration_budget_is_allowed() -> None:
    """"再跑几轮"就是续跑的定义本身（32 → 64）。"""

    after = _config(
        runtime={
            "rollout_k": 1,
            "max_iterations": 64,
            "match_parallelism": 8,
            "match_timeout_s": 1800,
        }
    )

    assert _significant_config_changes(_config(), after) == []


def test_relaxing_timeouts_and_parallelism_is_allowed() -> None:
    """并发度与超时是机时调度参数，不改变对局规则与对手。"""

    after = _config(
        runtime={
            "rollout_k": 1,
            "max_iterations": 32,
            "match_parallelism": 16,
            "match_timeout_s": 3600,
        }
    )

    assert _significant_config_changes(_config(), after) == []


def test_changing_the_model_is_rejected() -> None:
    after = _config(provider={"model": "glm-5.2", "reasoning_effort": "high"})

    assert _significant_config_changes(_config(), after) == ["provider"]


def test_changing_the_opponent_policy_is_rejected() -> None:
    """这是 ablation 的自变量 —— 中途改它会让整个 run 的数据失去意义。"""

    after = _config(curriculum={"opponent_policy": "fix", "batch": 4})

    assert _significant_config_changes(_config(), after) == ["curriculum"]


def test_changing_rollout_k_is_rejected() -> None:
    """k 在 runtime 段里，但它**不在**放行名单内。"""

    after = _config(
        runtime={
            "rollout_k": 4,
            "max_iterations": 32,
            "match_parallelism": 8,
            "match_timeout_s": 1800,
        }
    )

    assert _significant_config_changes(_config(), after) == ["runtime.rollout_k"]


def test_changing_history_mode_is_rejected() -> None:
    after = _config(goal={"history_mode": "last_only"})

    assert _significant_config_changes(_config(), after) == ["goal"]


def test_the_error_names_every_offending_field() -> None:
    """报错必须说出到底哪里变了，否则调用方只能自己 diff JSON 去猜。"""

    after = _config(
        provider={"model": "glm-5.2", "reasoning_effort": "high"},
        runtime={
            "rollout_k": 4,
            "max_iterations": 64,  # 这个是允许的
            "match_parallelism": 8,
            "match_timeout_s": 1800,
        },
    )

    changed = _significant_config_changes(_config(), after)

    assert changed == ["provider", "runtime.rollout_k"], (
        "只应报告真正不允许的字段，max_iterations 的变化不该出现在里面"
    )

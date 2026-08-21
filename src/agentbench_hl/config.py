"""Typed, secret-safe experiment configuration.

Schema 1.0 保持完全兼容；1.1 追加**可选**小节，让服务端表单的每个字段都有真实
消费点（不再有"只校验不生效"的死字段）：

- ``provider.harness``      : codex | cc（Claude Code）
- ``runtime.rollout_k``     : 每轮候选数（真正注入 Goal 指令并在 action 校验）
- ``runtime.match_parallelism``: 一轮内对局并行度（32 核服务器的关键）
- ``isolation.*``           : 隔离后端 + 对手代码可见性（消融）
- ``budget.*``              : token / wall 预算守卫
- ``curriculum.opponent_policy`` / ``opponent_rank`` / ``seed_mode``
- ``goal.*``                : prompt 覆盖、经验 skills 开关、策略语言约束、种子策略
- ``evaluation.*``          : 后台全池评测（慢通道）参数

缺省值保证：老的 1.0 配置逐字段行为不变。
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml

from agentbench_hl.adapters.codex_goal.app_server import AGENT_SANDBOX_MODES
from agentbench_hl.adapters.transcript.coupling import (
    COUPLING_COMMON_RANDOM,
    COUPLING_MODES,
)
from agentbench_hl.application.goal_led_service import BEHAVIORAL_IG_PROBES

_ENV_SCALAR = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")

SCHEMA_VERSIONS = ("1.0", "1.1")
HARNESSES = ("codex", "cc")
ORIGINS = ("from_scratch", "seeded")
OPPONENT_POLICIES = (
    "self_decide",
    "fixed_top",
    "fixed_rank",
    "random",
    "k_random",
    "ladder_up",
    "ladder_down",
    "k_diverse",
    "adaptive",
)
SEED_MODES = ("fixed", "generalize")
CODE_CONSTRAINTS = ("any", "if_else")
ITERATION_MODES = ("lockstep", "goal_autonomous")
# 实验 4 的消融：agent 能看到多少自己的历史。
#   full        : 常驻会话 + 经验文档 + 历次版本（默认）
#   no_notes    : 只关掉经验文档（保留会话与历史代码）
#   memoryless  : 每轮全新会话，只给当前父版本代码 + 上一轮反馈（无经验、无对话历史）
HISTORY_MODES = ("full", "no_notes", "memoryless")
# 与 adapters/contract/pool.py 保持一致：crawled 是正名，official 是弃用别名。
LADDER_SCOPES = ("auto", "crawled", "official", "reference", "measured")
ISOLATION_BACKENDS = ("auto", "seatbelt", "bubblewrap", "docker", "disabled")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _optional_mapping(value: object, name: str) -> Mapping[str, object]:
    if value is None:
        return {}
    return _mapping(value, name)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _text(value, name)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_bool(value: object, name: str, default: bool) -> bool:
    if value is None:
        return default
    return _bool(value, name)


def _positive_int(value: object, name: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _non_negative_int(value: object, name: str, default: int) -> int:
    """非负整数。0 是有意义的取值（"关闭这项测量"），所以不能复用 _positive_int。"""

    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be zero or positive")
    return value


def _positive_float(value: object, name: str, default: float) -> float:
    """正浮点（秒数一类）。整数写法也接受，省得配置里必须写 1800.0。"""

    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def _choice(value: object, name: str, allowed: tuple[str, ...], default: str) -> str:
    if value is None:
        return default
    text = _text(value, name)
    if text not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {text!r}")
    return text


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
    harness: str = "codex"
    # harness 本地的模型目录只认识自家模型（`codex debug models` 只返回 gpt-5.6-sol），
    # 对 glm-5.2 这类中转模型会打印 "Unknown model … will use fallback model metadata"
    # 并套用兜底参数。这两个字段把真实参数配准；留 None = 不写、保持 harness 兜底。
    #
    # context_window：模型的最大上下文 token 数。
    # auto_compact_token_limit：对话历史涨到多少 token 就自动压缩。实测一个 2 轮的 run
    #   input token 从 5888 单调涨到 219571（膨胀全部来自 agent 自己的工具调用历史，
    #   初始语料只占 2.7%），期间从未触发压缩——长跑必须把这个阈值配上。
    context_window: int | None = None
    auto_compact_token_limit: int | None = None
    # 厂商官方 model catalog 的名字（当前支持 ``zhipu``），None = 不用。
    #
    # 给了 catalog 就以它为准，``context_window`` 不再写进 config.toml。
    # 这不是可选的美化：不给 catalog 时 codex 用**兜底**模型元数据，压缩会在一个
    # 我们没设过也看不见的点触发（实测 antwar2 在 97k 触发并死掉，而当时我们
    # 声明的是 200000）。catalog 让"压缩线是多少"重新变成我们能回答的问题。
    model_catalog: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    codex_binary: str
    branch_width: int
    max_iterations: int | None
    network_access: str
    rollout_k: int = 4
    match_parallelism: int = 1
    # 单局墙钟上限。默认 1800s，而不是原来硬编码的 420s——
    # 各游戏单局长度差一个数量级：miracle 一局 7s，snakego 一局实测 246s，
    # 而全池评分时 snakego 还有 19 局连 900s 都没跑完。420s 会让长局被超时判负，
    # 主表上看起来就是"这个模型不会玩 snakego"，属于把基建问题读成实验结论。
    match_timeout_s: float = 1800.0
    agent_binary: str | None = None
    # 会话轮转阈值（当前 thread 的上下文 token 数）。None = 不轮转。
    #
    # 为什么需要它：codex 0.147 的自动压缩走的是 **remote compaction v2**
    # （POST /responses/compact），它要求响应里恰好一个 compaction output item。
    # glm-5.2 在任何 reasoning_effort 下都会返回 [reasoning, message] 两个 item，
    # 于是压缩必定失败并把整个 turn 打成 status=failed：
    #   "expected exactly one compaction output item, got 0 from 2 output items"
    # 实测两个 run 分别死在 135k 与 90k 上下文处，都不是本地阈值触发点——
    # 也就是说"把 auto_compact_token_limit 调低让主动压缩先触发"这条路是错的：
    # 主动压缩用的是同一条 remote 通道，一触发就死。
    #
    # 所以框架必须保证**压缩永远不被触发**：在上下文触到 codex 阈值之前主动换一个
    # 新 thread。这不是消融变量——工作区侧的历史（历史候选代码、全部反馈、
    # EXPERIENCE.md）一份都不动，断掉的只有 harness 的对话记忆，
    # 而 HL 本来就要求"经验必须落到 research/ 才算学到"。
    thread_rotate_context_tokens: int | None = None
    # 每轮无条件换 thread（不看上下文大小）。
    #
    # 为什么"看阈值"不够：``thread_rotate_context_tokens`` 只能在**turn 之前**检查，
    # 而一个 turn 内部 agent 会连续发几十个请求（读 8 份 replay.md、写 k 个候选），
    # 上下文能在这一个 turn 里从 90k 冲到压缩线。实测代价：antwar2 死在 97k，
    # 低于当时 110k 的阈值——检查时机根本没来得及。
    # 每轮从零开始，则单轮增量（实测 antwar2 ≈ 90k）就是上下文的上界。
    thread_rotate_each_iteration: bool = False
    # lockstep：轮数由框架决定（Goal 自称 complete 也继续推进，只记事件）；
    # goal_autonomous：尊重 Goal 的提前收敛（对照实验用）。
    iteration_mode: str = "lockstep"

    @property
    def harness_binary(self) -> str:
        """harness 可执行文件：``agent_binary`` 优先，回落 ``codex_binary``。"""

        return self.agent_binary or self.codex_binary


@dataclass(frozen=True)
class IsolationConfig:
    #: 候选**对局**的隔离后端（bwrap / seatbelt / docker）。科学上必须成立的那一层：
    #: 它把人类选手池与评测器 tmpfs 掉，决定"候选有没有偷看过答案"。
    backend: str = "auto"
    rival_code_visible: bool = False
    docker_image: str | None = None
    #: harness（codex）自带的 OS 级沙箱。默认 ``danger-full-access`` = 关掉。
    #: 实测 codex 0.147 的 linux_sandbox 在服务器上会拒掉每次 exec_command
    #: （连 PATH 里的 venv 都读不到），agent 因此永远交不出 action.json。
    #: 这一层与上面的 ``backend`` 是**两件事**，别混。
    agent_sandbox: str = "danger-full-access"


@dataclass(frozen=True)
class BudgetConfig:
    tokens: int | None = None
    wall_seconds: int | None = None


@dataclass(frozen=True)
class GoalConfig:
    prompt_override: str | None = None
    experience_skills: bool = True
    code_constraint: str = "any"
    seed_policy_path: Path | None = None
    # 实验 4：历史可见性消融（见 HISTORY_MODES）。
    history_mode: str = "full"


@dataclass(frozen=True)
class EvaluationConfig:
    background_pool: bool = False
    pool_sample: int = 16
    pool_seeds: tuple[int, ...] = (7,)


@dataclass(frozen=True)
class PathConfig:
    agentbench_root: Path
    runs_root: Path


@dataclass(frozen=True)
class CurriculumConfig:
    order: str
    development_seeds: tuple[int, ...]
    opponent_policy: str = "self_decide"
    opponent_rank: int | None = None
    seed_mode: str = "fixed"
    # 有序征服课程（ladder_up / ladder_down）的起点名次与"打赢了才换人"的判据。
    opponent_start_rank: int | None = None
    advance_min_matches: int = 2
    advance_win_rate: float = 0.6
    advance_streak: int = 1
    # 对手榜单口径：auto = measured → reference → official 逐选手回落。
    # 默认 auto（measured → reference → crawled 逐选手回落）。
    # 绝不能默认成 crawled：那只有第一批爬取的几十个人有名次，
    # 榜单会从 229 人缩到 20 人，对手课程直接缺斤少两。
    ladder_scope: str = "auto"


@dataclass(frozen=True)
class MeasurementConfig:
    epsilon: float
    # 逐轮信息增益（配对影子对局）。关掉可省每轮 N 局基线对局，但曲线会变 null。
    information_gain: bool = True
    # 决策级行为信息增益：每轮拿几个配对 case 做线协议录制 + 冻结重放。
    # 每个 case 的成本 = 2 局录制对局 + 2 次本地重放；0 = 关闭（曲线记 null + 原因）。
    behavioral_ig_cases: int = 1
    # 单次本地重放的墙钟上限。重放要走完整局的观测流，所以不能比单局超时小太多。
    behavioral_ig_timeout_s: float = 900.0
    # 随机流耦合口径：``common_random_seed``（默认）让父/子版本在同一条随机流下比较，
    # 否则任何调 random 的选手都会被判"非确定"而记 null。口径会随数写进事件。
    behavioral_ig_coupling: str = COUPLING_COMMON_RANDOM
    # 行为 IG 用哪种探针。默认 transcript_replay（8 游戏 / 9 轨都做过零点校准与灵敏度
    # 验收）。**一个 run 只用一种**：两种探针的 |A| 与动作口径不同，逐轮自动回落会让
    # 同一条曲线的相邻两点口径不同，斜率失去意义。
    behavioral_ig_probe: str = "transcript_replay"


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
    schema_version: str
    game: str
    origin: str
    provider: ProviderConfig
    runtime: RuntimeConfig
    paths: PathConfig
    curriculum: CurriculumConfig
    measurement: MeasurementConfig
    isolation: IsolationConfig = field(default_factory=IsolationConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    _environment: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

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
        origin = _choice(root.get("origin"), "origin", ORIGINS, "from_scratch")
        schema_version = _text(root.get("schema_version"), "schema_version")
        if schema_version not in SCHEMA_VERSIONS:
            raise ValueError(f"schema_version must be one of {SCHEMA_VERSIONS}")
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
        isolation = _optional_mapping(root.get("isolation"), "isolation")
        budget = _optional_mapping(root.get("budget"), "budget")
        goal = _optional_mapping(root.get("goal"), "goal")
        evaluation = _optional_mapping(root.get("evaluation"), "evaluation")

        branch_width = _positive_int(runtime.get("branch_width"), "runtime.branch_width")
        max_iterations = _optional_positive_int(
            runtime.get("max_iterations"), "runtime.max_iterations"
        )
        if runtime.get("network_access") != "disabled":
            raise ValueError("runtime.network_access must be disabled")
        if curriculum.get("order") != "lowest_rank_first":
            raise ValueError("curriculum.order must be lowest_rank_first")
        epsilon = measurement.get("epsilon")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError("measurement.epsilon must be numeric")
        if not 0 < float(epsilon) < 1:
            raise ValueError("measurement.epsilon must be between zero and one")

        opponent_policy = _choice(
            curriculum.get("opponent_policy"),
            "curriculum.opponent_policy",
            OPPONENT_POLICIES,
            "self_decide",
        )
        opponent_rank = _optional_positive_int(
            curriculum.get("opponent_rank"), "curriculum.opponent_rank"
        )
        if opponent_policy == "fixed_rank" and opponent_rank is None:
            raise ValueError("curriculum.opponent_rank is required when opponent_policy=fixed_rank")
        advance_rate = curriculum.get("advance_win_rate")
        if advance_rate is None:
            advance_win_rate = 0.6
        elif isinstance(advance_rate, bool) or not isinstance(advance_rate, (int, float)):
            raise ValueError("curriculum.advance_win_rate must be numeric")
        else:
            advance_win_rate = float(advance_rate)
            if not 0 < advance_win_rate <= 1:
                raise ValueError("curriculum.advance_win_rate must be within (0, 1]")

        seed_policy = goal.get("seed_policy_path")
        if origin == "seeded" and seed_policy is None:
            raise ValueError("goal.seed_policy_path is required when origin=seeded")

        return cls(
            schema_version=schema_version,
            game=game,
            origin=origin,
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
                context_window=_optional_positive_int(
                    provider.get("context_window"), "provider.context_window"
                ),
                auto_compact_token_limit=_optional_positive_int(
                    provider.get("auto_compact_token_limit"),
                    "provider.auto_compact_token_limit",
                ),
                model_catalog=_optional_text(
                    provider.get("model_catalog"), "provider.model_catalog"
                ),
                harness=_choice(provider.get("harness"), "provider.harness", HARNESSES, "codex"),
            ),
            runtime=RuntimeConfig(
                codex_binary=_text(runtime.get("codex_binary"), "runtime.codex_binary"),
                branch_width=branch_width,
                max_iterations=max_iterations,
                network_access="disabled",
                rollout_k=_positive_int(runtime.get("rollout_k"), "runtime.rollout_k", 4),
                match_parallelism=_positive_int(
                    runtime.get("match_parallelism"), "runtime.match_parallelism", 1
                ),
                match_timeout_s=_positive_float(
                    runtime.get("match_timeout_s"), "runtime.match_timeout_s", 1800.0
                ),
                agent_binary=_optional_text(runtime.get("agent_binary"), "runtime.agent_binary"),
                thread_rotate_context_tokens=_optional_positive_int(
                    runtime.get("thread_rotate_context_tokens"),
                    "runtime.thread_rotate_context_tokens",
                ),
                thread_rotate_each_iteration=_optional_bool(
                    runtime.get("thread_rotate_each_iteration"),
                    "runtime.thread_rotate_each_iteration",
                    False,
                ),
                iteration_mode=_choice(
                    runtime.get("iteration_mode"),
                    "runtime.iteration_mode",
                    ITERATION_MODES,
                    "lockstep",
                ),
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
                opponent_policy=opponent_policy,
                opponent_rank=opponent_rank,
                seed_mode=_choice(
                    curriculum.get("seed_mode"), "curriculum.seed_mode", SEED_MODES, "fixed"
                ),
                opponent_start_rank=_optional_positive_int(
                    curriculum.get("opponent_start_rank"), "curriculum.opponent_start_rank"
                ),
                advance_min_matches=_positive_int(
                    curriculum.get("advance_min_matches"), "curriculum.advance_min_matches", 2
                ),
                advance_win_rate=advance_win_rate,
                advance_streak=_positive_int(
                    curriculum.get("advance_streak"), "curriculum.advance_streak", 1
                ),
                ladder_scope=_choice(
                    curriculum.get("ladder_scope"),
                    "curriculum.ladder_scope",
                    LADDER_SCOPES,
                    "auto",
                ),
            ),
            measurement=MeasurementConfig(
                epsilon=float(epsilon),
                information_gain=_optional_bool(
                    measurement.get("information_gain"), "measurement.information_gain", True
                ),
                behavioral_ig_cases=_non_negative_int(
                    measurement.get("behavioral_ig_cases"), "measurement.behavioral_ig_cases", 1
                ),
                behavioral_ig_timeout_s=_positive_float(
                    measurement.get("behavioral_ig_timeout_s"),
                    "measurement.behavioral_ig_timeout_s",
                    900.0,
                ),
                behavioral_ig_coupling=_choice(
                    measurement.get("behavioral_ig_coupling"),
                    "measurement.behavioral_ig_coupling",
                    COUPLING_MODES,
                    COUPLING_COMMON_RANDOM,
                ),
                behavioral_ig_probe=_choice(
                    measurement.get("behavioral_ig_probe"),
                    "measurement.behavioral_ig_probe",
                    BEHAVIORAL_IG_PROBES,
                    "transcript_replay",
                ),
            ),
            isolation=IsolationConfig(
                backend=_choice(
                    isolation.get("backend"), "isolation.backend", ISOLATION_BACKENDS, "auto"
                ),
                rival_code_visible=_optional_bool(
                    isolation.get("rival_code_visible"), "isolation.rival_code_visible", False
                ),
                docker_image=_optional_text(
                    isolation.get("docker_image"), "isolation.docker_image"
                ),
                agent_sandbox=_choice(
                    isolation.get("agent_sandbox"),
                    "isolation.agent_sandbox",
                    AGENT_SANDBOX_MODES,
                    "danger-full-access",
                ),
            ),
            budget=BudgetConfig(
                tokens=_optional_positive_int(budget.get("tokens"), "budget.tokens"),
                wall_seconds=_optional_positive_int(
                    budget.get("wall_seconds"), "budget.wall_seconds"
                ),
            ),
            goal=GoalConfig(
                prompt_override=_optional_text(goal.get("prompt_override"), "goal.prompt_override"),
                experience_skills=_optional_bool(
                    goal.get("experience_skills"), "goal.experience_skills", True
                ),
                code_constraint=_choice(
                    goal.get("code_constraint"),
                    "goal.code_constraint",
                    CODE_CONSTRAINTS,
                    "any",
                ),
                history_mode=_choice(
                    goal.get("history_mode"), "goal.history_mode", HISTORY_MODES, "full"
                ),
                seed_policy_path=(
                    None
                    if seed_policy is None
                    else _path(
                        seed_policy, "goal.seed_policy_path", base=path.parent, env=environment
                    )
                ),
            ),
            evaluation=EvaluationConfig(
                background_pool=_optional_bool(
                    evaluation.get("background_pool"), "evaluation.background_pool", False
                ),
                pool_sample=_positive_int(
                    evaluation.get("pool_sample"), "evaluation.pool_sample", 16
                ),
                pool_seeds=(
                    _seeds(evaluation.get("pool_seeds"), "evaluation.pool_seeds")
                    if evaluation.get("pool_seeds") is not None
                    else (7,)
                ),
            ),
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
                "opponent_policy": self.curriculum.opponent_policy,
                "opponent_rank": self.curriculum.opponent_rank,
                "seed_mode": self.curriculum.seed_mode,
                "opponent_start_rank": self.curriculum.opponent_start_rank,
                "advance_min_matches": self.curriculum.advance_min_matches,
                "advance_win_rate": self.curriculum.advance_win_rate,
                "advance_streak": self.curriculum.advance_streak,
                "ladder_scope": self.curriculum.ladder_scope,
            },
            "measurement": dataclasses.asdict(self.measurement),
            "isolation": dataclasses.asdict(self.isolation),
            "budget": dataclasses.asdict(self.budget),
            "goal": {
                "prompt_override": self.goal.prompt_override,
                "experience_skills": self.goal.experience_skills,
                "code_constraint": self.goal.code_constraint,
                "history_mode": self.goal.history_mode,
                "seed_policy_path": (
                    None if self.goal.seed_policy_path is None else str(self.goal.seed_policy_path)
                ),
            },
            "evaluation": {
                "background_pool": self.evaluation.background_pool,
                "pool_sample": self.evaluation.pool_sample,
                "pool_seeds": list(self.evaluation.pool_seeds),
            },
        }

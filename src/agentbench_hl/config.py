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
# 四个主设置（都可消融）+ 三个历史别名（老配置与已完成 run 要能复现）。
OPPONENT_POLICIES = (
    "random",
    "self",
    "progress",
    "fix",
    # ---- 历史别名 ----
    "self_decide",
    "fixed_top",
    "ladder_up",
    "ladder_down",
    "fixed_rank",
    "k_diverse",
)
SEED_MODES = ("fixed", "generalize")
CODE_CONSTRAINTS = ("any", "if_else")
ITERATION_MODES = ("lockstep", "goal_autonomous")
# 历史可见性（超参数 2）。两个主设置：
#   full        : 能看到所有历史迭代（常驻会话 + 经验文档 + 历次版本代码）
#   last_only   : 只能看到上一版（每轮全新会话，工作区只留当前策略 + 最近一份反馈）
#
# ``last_only`` 下框架还会**要求策略写在单文件里**。这不是洁癖：
# 如果允许 ``v3.py import v2``，那"只能看到上一版"就被 import 链绕过了——
# agent 读一下 v2.py 就等于看到了历史，消融失效且从曲线上看不出来。
# 单文件约束让"上一版"这四个字有唯一含义：工作区里那一个 main.py 就是全部历史。
#
# ``no_notes`` / ``memoryless`` 是历史别名：前者只关经验文档，后者等价 last_only。
HISTORY_MODES = ("full", "last_only", "no_notes", "memoryless")
# 与 adapters/contract/pool.py 保持一致：crawled 是正名，official 是弃用别名。
LADDER_SCOPES = ("auto", "crawled", "official", "reference", "measured")
ISOLATION_BACKENDS = ("auto", "seatbelt", "bubblewrap", "docker", "disabled")

# 最大迭代轮数的默认值与"不设上限"的固定含义。
#
# 为什么"不设上限"要落成一个具体数字：``max_iterations: null`` 曾经表示"无限跑，
# 靠预算停"。但 token 预算在记账修好之前是失效的（见 goal_led_service._token_total
# 的详注），于是"不设上限"实际等于"跑到人手动杀掉"，不同 run 的轮数不可比。
# 现在固定：不写 = 32 轮；显式要"不设上限" = 128 轮。两者都是可复现的数字。
DEFAULT_MAX_ITERATIONS = 32
UNBOUNDED_MAX_ITERATIONS = 128


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
    #: JSON-RPC ``initialize`` 里报的客户端名字，会变成上游看到的
    #: ``originator`` 请求头。None = 用运行时默认（``agentbench-hl``）。
    #:
    #: 为什么要暴露成配置：**有些中转站按客户端白名单放行**。实测 sbtunnel
    #: 对 ``originator: agentbench-hl`` 返回 403
    #: ``This account only allows Codex official clients``，而同一个 key、
    #: 同一个端点、同一份 config.toml 换成 codex 官方 originator 就 200。
    #: 硬编码的话这类中转站整个不可用，且 403 完全指不回"是署名被拒"
    #: （我们为此把 config.toml 的每一项都单独试了一遍，全部无关）。
    client_name: str | None = None


# 模型档案库：``configs/models/<name>.yaml``。
#
# 为什么模型要独立成文件
# --------------------
# 主表要横向比 7 个模型（gpt5.6 / opus5 / deepseek-v4-pro / qwen3.8 /
# longcat-2.0 / glm-5.3 / kimi-k3），而它们的**中转站、api key 环境变量、
# 上下文窗口、model_catalog 全都不一样**。如果这些散在每个实验配置里，
# 换一个模型就要改 5 个字段，而且很容易漏掉一个 —— 漏掉 context_window
# 的后果是 codex 用兜底元数据、压缩在看不见的点触发并打死整个 run。
#
# 独立成档案之后，实验配置里只写 ``provider.model_profile: glm-5.3``，
# 剩下的从档案取。同一个模型换中转站只改一处，所有实验自动跟上。
MODEL_PROFILES_DIRNAME = "models"


def _model_profiles_root(config_path: Path) -> Path:
    """从配置文件位置**向上搜索**仓库里的 ``configs/models/``。

    和 :func:`_gamepacks_root` 同一个理由：写死"往上数 1 层"会让
    ``configs/experiments/ablation/x.yaml`` 去找 ``configs/experiments/models/``，
    然后报"未知模型档案"——错的是路径而不是档案名。
    """

    resolved = config_path.resolve()
    for parent in resolved.parents:
        candidate = parent / MODEL_PROFILES_DIRNAME
        if candidate.is_dir():
            return candidate
        nested = parent / "configs" / MODEL_PROFILES_DIRNAME
        if nested.is_dir():
            return nested
    return resolved.parents[1] / MODEL_PROFILES_DIRNAME


def _load_model_profile(name: str, root: Path) -> Mapping[str, object]:
    path = root / f"{name}.yaml"
    if not path.is_file():
        available = sorted(item.stem for item in root.glob("*.yaml")) if root.is_dir() else []
        raise ValueError(
            f"unknown provider.model_profile {name!r}: {path} not found"
            + (f"; available: {available}" if available else "")
        )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    profile = _mapping(document, f"model profile {name}")
    for required in ("model", "base_url", "api_key_env"):
        if not profile.get(required):
            raise ValueError(f"model profile {name} must define {required}")
    return profile


@dataclass(frozen=True)
class RuntimeConfig:
    codex_binary: str
    branch_width: int
    max_iterations: int | None
    network_access: str
    # 每轮候选数。默认 1：一轮只写一个策略，拿它去打 ``curriculum.batch`` 个对手。
    #
    # 默认 4，而且这个值**有实测依据**（antwar2，同一个 229 人池）：
    #
    #   k=4 + 单对手 progress → 第 3 轮进池内 #84，第 9 轮 #24，第 21 轮 #10
    #   k=1 + b=4 对手        → 第 30 轮才 #107
    #
    # 曾经改成 1，理由是"k=4 的多样性会退化成同一份代码改几个阈值"。
    # 那个判断被数据推翻了：旧 run 前 14 轮的 pairwise 行差异是 48~251 行
    # （阈值 15），全部判定 distinct —— 多样性是真的。
    #
    # k>1 的价值是**并行假设检验**：一轮把 k 条取胜路径同时下水，下一轮把胜出
    # 那条变成所有候选的共同底盘，再从那里分叉测增量。一轮拿到 k bit。
    # 我原先以为"b 个对手能提供探索广度"，这是错的：b 个对手只让**同一个策略**
    # 的评估更精确（降方差），不产生新的候选假设。广度必须来自策略侧。
    #
    # k=1 仍然是合法配置（消融维度之一），但它测的是"没有并行假设检验时
    # 能走多远"，不该用来代表框架能达到的水平。
    rollout_k: int = 4
    match_parallelism: int = 1
    # 单局墙钟上限。默认 1800s，而不是原来硬编码的 420s——
    # 各游戏单局长度差一个数量级：miracle 一局 7s，snakego 一局实测 246s，
    # 而全池评分时 snakego 还有 19 局连 900s 都没跑完。420s 会让长局被超时判负，
    # 主表上看起来就是"这个模型不会玩 snakego"，属于把基建问题读成实验结论。
    match_timeout_s: float = 1800.0
    #: **每一步**的思考上限（秒）。None = 不限（历史行为）。
    #:
    #: 为什么这是个一等实验变量，而不是一个运维旋钮
    #: --------------------------------------------
    #: saiblo 判题器对 AI 选手是**按步**计时的，不是按整局。已在 saiblo 上核对：
    #: lostspace/miracle 后端每步都重发 ``send_init(AI_TIME, length)``，
    #: 而 miracle 写得最直白::
    #:
    #:     AI_TIME = 3          # AI 选手每步 3 秒
    #:     PLAYER_TIME = 300    # 真人玩家每步 300 秒
    #:
    #: 我们的 arena 一直把这一帧里的 ``time`` **丢掉**，只保留整局墙钟。后果有两层：
    #:
    #: 1. 一名选手卡在某一步，saiblo 上只是那一步判超时、对局继续；我们这边是
    #:    整局耗尽墙钟后 TimeoutError，**整局作废**（0 回合 / loss / 无回放）。
    #: 2. 更要紧的是**有效性**：人类选手池是在"每步 3 秒"下写出来的（所以清一色
    #:    C++），而我们的候选此前享受每步无限时间。那样算出来的池内 Elo
    #:    不是同一个游戏的 Elo。
    #:
    #: 所以默认给一个**宽松但有限**的值：既拿回"一步卡住不毁整局"，又不至于把
    #: Python 候选按 3 秒直接打死。要做同条件对照就显式写 3。
    step_timeout_s: float | None = 30.0
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
    #: 是否自动起后台慢评测（每 N 轮把中间版本拉去打完整个冻结人类池）。
    #:
    #: ★ 这个开关**现在真的生效**了。它以前只被解析、没有任何消费点：
    #: 写 ``true`` 什么也不会发生，慢评测一直靠人手动敲
    #: ``scripts/pool_elo_worker.py``。漏做的表现是 Elo 面板里少一条实测曲线
    #: ——不报错，只是图上静静地少一条线。消费点见 ``cli/main.py``。
    background_pool: bool = False
    pool_sample: int = 16
    pool_seeds: tuple[int, ...] = (7,)
    #: 每几轮取一个版本做全池评测（只测各轮的最佳候选，即演进主线）。
    #:
    #: 3 是成本与分辨率的平衡点：32 轮 → 11 版 × 约 458 局 ≈ 5k 局，
    #: 与迭代并行跑得完。stride=1 会变成 32 版 ≈ 15k 局，慢评测永远追不上
    #: 迭代，图上只会有前几个点。相邻轮次的池内 Elo 差异远小于 ±50 标准误，
    #: 抽样不改变曲线形状。
    pool_stride: int = 3
    #: 分轨（非对称）游戏必填：挑战者**自己扮演**哪一轨（如 rollman / ghost）。
    #:
    #: 不给的话慢评测会出现同轨互殴（ghost 打 ghost），那种对局在协议层就没
    #: 意义——实测 rollman 的回放只有 2 行、IG 恒为常数，排查了很久才发现
    #: 根因是轨道没分。
    challenger_track: str | None = None


@dataclass(frozen=True)
class PathConfig:
    agentbench_root: Path
    runs_root: Path


@dataclass(frozen=True)
class CurriculumConfig:
    order: str
    development_seeds: tuple[int, ...]
    opponent_policy: str = "progress"
    # 一轮打几个对手（超参数 3 的 b）。默认 4，可消融。
    #
    # 这是 k=1 之后新的探索宽度旋钮：一轮只出一个策略，但让它面对 b 个不同的
    # 对手。b=1 时各策略退化成原来的单目标形态（progress 就是 ladder_up）。
    batch: int = 4
    opponent_rank: int | None = None
    seed_mode: str = "fixed"
    # 有序征服课程（progress / ladder_*）的起点名次与"打赢了才换人"的判据。
    opponent_start_rank: int | None = None
    advance_min_matches: int = 2
    advance_win_rate: float = 0.75
    advance_streak: int = 1
    # 对手榜单口径：auto = measured → reference → official 逐选手回落。
    # 默认 auto（measured → reference → crawled 逐选手回落）。
    # 绝不能默认成 crawled：那只有第一批爬取的几十个人有名次，
    # 榜单会从 229 人缩到 20 人，对手课程直接缺斤少两。
    ladder_scope: str = "auto"


@dataclass(frozen=True)
class MeasurementConfig:
    """测量口径。

    ⚠️ 信息增益（IG）已从主线指标中移除
    -----------------------------------
    ``information_gain`` 与 ``behavioral_ig_*`` 现在**默认全关**，曲线也不再画
    IG 面板（只剩胜率 / Elo / token 三组）。字段保留是为了让已完成的 run
    （antwar / antwar2 等带 IG 数据的）仍能被原样加载与复现，不是为了继续测。

    关掉之后每轮省下的成本是实打实的：``information_gain`` 每轮要跑一批配对
    影子对局，``behavioral_ig_cases`` 每个 case 还要 2 局录制 + 2 次本地重放，
    而这些对局对"策略变强了没有"这个问题不提供任何证据。
    """

    epsilon: float = 0.01
    # 逐轮结果分布信息增益（配对影子对局）。默认关闭。
    information_gain: bool = False
    # 决策级行为信息增益（线协议录制 + 冻结重放）。默认 0 = 关闭。
    behavioral_ig_cases: int = 0
    # 单次本地重放的墙钟上限。重放要走完整局的观测流，所以不能比单局超时小太多。
    behavioral_ig_timeout_s: float = 900.0
    # 随机流耦合口径（仅在显式开启 IG 时有意义）。
    behavioral_ig_coupling: str = COUPLING_COMMON_RANDOM
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


def repository_root_for(config_path: str | Path) -> Path:
    """从实验配置的位置定位**仓库根目录**。

    为什么需要这个函数（而不是各处写 ``parents[2]``）
    ----------------------------------------------
    "配置一定在 ``configs/experiments/`` 正下方，所以往上数 2 层就是仓库根"
    这个假设原先散在 6 个地方：``factory.build_goal_run`` 找 gamepack、
    4 处 CLI 找 ``.env``、以及 gamepacks/model-profiles 的根路径推导。

    一旦按用途给配置分子目录（``configs/experiments/ablation/`` 放一组
    只差一个字段的消融配置），这 6 处会**各自以不同方式**失败：

    * gamepack 报 ``GamePack manifest not found: …/configs/gamepacks/antwar2/…``
      —— 看着像少了文件，其实是路径少数了一层；
    * ``.env`` 更糟：找不到就**静默跳过**，于是 api key 没加载，
      run 起来之后才在第一次模型调用时报鉴权错，
      而那时错误信息只会说"401"，完全指不回真正的原因。

    改成"向上找第一个同时含 ``gamepacks/`` 与 ``configs/`` 的目录"：
    配置放在哪一层都对，也不再依赖任何目录深度约定。
    """

    resolved = Path(config_path).resolve()
    for parent in resolved.parents:
        if (parent / "gamepacks").is_dir() and (parent / "configs").is_dir():
            return parent
    # 兜底保留历史行为，让老布局下的错误信息与从前一致。
    return resolved.parents[2] if len(resolved.parents) > 2 else resolved.parent


def _gamepacks_root(config_path: Path) -> Path:
    """从配置文件位置**向上搜索**仓库里的 ``gamepacks/``。

    见 :func:`repository_root_for` 的详注：写死层数会在配置分子目录时
    报出指向错误方向的信息（"游戏没注册"而非"路径算错了"）。
    """

    resolved = config_path.resolve()
    for parent in resolved.parents:
        candidate = parent / "gamepacks"
        if candidate.is_dir():
            return candidate
    # 一个都没找到：保留原来的层数作为兜底，让错误信息与历史一致。
    return resolved.parents[2] / "gamepacks" if len(resolved.parents) > 2 else resolved.parent


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
        model_profiles_root: Path | None = None,
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

        provider = dict(_mapping(root.get("provider"), "provider"))
        # 模型档案：configs/models/<name>.yaml 提供 model / base_url / api_key_env /
        # context_window / model_catalog 等；实验配置里同名字段可以覆盖它
        # （便于临时试一个 reasoning_effort，而不必新建档案）。
        profile_name = _optional_text(provider.get("model_profile"), "provider.model_profile")
        if profile_name is not None:
            profiles_root = (
                model_profiles_root
                if model_profiles_root is not None
                else _model_profiles_root(path)
            )
            profile = _load_model_profile(profile_name, profiles_root)
            merged = dict(profile)
            merged.update({k: v for k, v in provider.items() if v is not None})
            provider = merged
        runtime = _mapping(root.get("runtime"), "runtime")
        paths = _mapping(root.get("paths"), "paths")
        curriculum = _mapping(root.get("curriculum"), "curriculum")
        measurement = _optional_mapping(root.get("measurement"), "measurement")
        isolation = _optional_mapping(root.get("isolation"), "isolation")
        budget = _optional_mapping(root.get("budget"), "budget")
        goal = _optional_mapping(root.get("goal"), "goal")
        evaluation = _optional_mapping(root.get("evaluation"), "evaluation")

        branch_width = _positive_int(runtime.get("branch_width"), "runtime.branch_width", 1)
        # 轮数上限（超参数 1）。三种写法：
        #   不写 / null      -> DEFAULT_MAX_ITERATIONS（32）
        #   "unbounded"      -> UNBOUNDED_MAX_ITERATIONS（128），"不设上限"的固定含义
        #   具体整数          -> 照用
        raw_iterations = runtime.get("max_iterations")
        if isinstance(raw_iterations, str):
            if raw_iterations.strip().lower() not in ("unbounded", "unlimited", "none"):
                raise ValueError(
                    "runtime.max_iterations must be an integer, null, or 'unbounded'"
                )
            max_iterations: int | None = UNBOUNDED_MAX_ITERATIONS
        elif raw_iterations is None:
            max_iterations = DEFAULT_MAX_ITERATIONS
        else:
            max_iterations = _positive_int(raw_iterations, "runtime.max_iterations")
        if runtime.get("network_access") not in (None, "disabled"):
            raise ValueError("runtime.network_access must be disabled")
        if curriculum.get("order") not in (None, "lowest_rank_first"):
            raise ValueError("curriculum.order must be lowest_rank_first")
        raw_epsilon = measurement.get("epsilon")
        if raw_epsilon is None:
            epsilon: float = 0.01
        elif isinstance(raw_epsilon, bool) or not isinstance(raw_epsilon, (int, float)):
            raise ValueError("measurement.epsilon must be numeric")
        else:
            epsilon = float(raw_epsilon)
            if not 0 < epsilon < 1:
                raise ValueError("measurement.epsilon must be between zero and one")

        opponent_policy = _choice(
            curriculum.get("opponent_policy"),
            "curriculum.opponent_policy",
            OPPONENT_POLICIES,
            "progress",
        )
        opponent_rank = _optional_positive_int(
            curriculum.get("opponent_rank"), "curriculum.opponent_rank"
        )
        if opponent_policy == "fixed_rank" and opponent_rank is None:
            raise ValueError("curriculum.opponent_rank is required when opponent_policy=fixed_rank")
        advance_rate = curriculum.get("advance_win_rate")
        if advance_rate is None:
            advance_win_rate = 0.75
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
                client_name=_optional_text(
                    provider.get("client_name"), "provider.client_name"
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
                # 显式写 null = 不限（回到历史行为）；不写 = 30s 宽松档。
                step_timeout_s=(
                    None
                    if "step_timeout_s" in runtime and runtime.get("step_timeout_s") is None
                    else _positive_float(
                        runtime.get("step_timeout_s"), "runtime.step_timeout_s", 30.0
                    )
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
                batch=_positive_int(curriculum.get("batch"), "curriculum.batch", 4),
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
                epsilon=epsilon,
                # IG 已从主线移除：默认关闭，只有配置里显式写 true 才测。
                information_gain=_optional_bool(
                    measurement.get("information_gain"), "measurement.information_gain", False
                ),
                behavioral_ig_cases=_non_negative_int(
                    measurement.get("behavioral_ig_cases"), "measurement.behavioral_ig_cases", 0
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
                pool_stride=_positive_int(
                    evaluation.get("pool_stride"), "evaluation.pool_stride", 3
                ),
                challenger_track=_optional_text(
                    evaluation.get("challenger_track"), "evaluation.challenger_track"
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
                "batch": self.curriculum.batch,
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

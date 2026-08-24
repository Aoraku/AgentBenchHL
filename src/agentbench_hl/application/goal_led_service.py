"""Thin bridge for a single Goal-led research run.

Goal 决定科学动作；本模块负责执行它写下的官方对局、把公开证据交还给同一个 Goal 线程，
并按实验配置施加**框架级约束**：

- 对手选择策略（`application/opponent_policy.py`，六个枚举）；
- 每轮候选数 K、座次、seed（``seed_mode`` 决定固定还是泛化）；
- 一轮内对局**并行**执行（32 核服务器的关键；默认 1 保持旧行为）；
- token / wall 预算守卫所需的度量事件（``IterationMetricsFinalized``）；
- Goal prompt 覆盖、经验文档开关、策略语言约束、对手代码可见性（消融）。

新增参数全部有默认值，默认行为与重构前一致。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.adapters.transcript.coupling import (
    COUPLING_COMMON_RANDOM,
    normalize_coupling,
)
from agentbench_hl.application.behavioral_ig import (
    BehavioralIgCase,
    BehavioralIgMeasurement,
    measure_behavioral_ig,
)
from agentbench_hl.application.candidate_diversity import (
    NEAR_DUPLICATE_LINES as DIVERSITY_THRESHOLD_LINES,
)
from agentbench_hl.application.candidate_diversity import feedback_note as diversity_note
from agentbench_hl.application.candidate_diversity import spread as candidate_spread
from agentbench_hl.application.candidate_preflight import check_candidate
from agentbench_hl.application.conquest import (
    AdvanceRule,
    ConquestState,
)
from agentbench_hl.application.conquest import evaluate as conquest_evaluate
from agentbench_hl.application.conquest import round_results as conquest_rounds
from agentbench_hl.application.container_guard import assert_sealed
from agentbench_hl.application.decision_space import load_information_gain_spec
from agentbench_hl.application.goal_led_protocol import MatchRequest
from agentbench_hl.application.info_gain import (
    outcome_counts,
    outcome_ig_nats,
    paired_margin_shift,
    replay_divergence,
)
from agentbench_hl.application.opponent_policy import (
    PROGRESS_WINDOW_ITERATIONS,
    PROGRESS_WINDOW_WINS,
    SELF_DECIDE,
    LadderEntry,
    OpponentHistory,
    build_policy,
    canonical_policy_name,
)
from agentbench_hl.application.replay_narration import NARRATION_FILENAME, narrate_case
from agentbench_hl.application.support_probe import provider_for
from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.pool_elo import estimate_pool_elo
from agentbench_hl.ports.agent_runtime import AgentRuntime, AgentSession, RunContext
from agentbench_hl.ports.arena import Arena, MatchCase, MatchResult
from agentbench_hl.ports.policy_probe import probe_availability

EXPERIENCE_FILE = "EXPERIENCE.md"

#: 行为 IG 的探针口径。一个 run 只能用一种，混用会让曲线的相邻两点口径不同。
BEHAVIORAL_IG_PROBES = ("transcript_replay", "in_process_first")


@dataclass(frozen=True)
class GoalLedOutcome:
    thread_id: str
    workspace: Path
    request_id: str
    match_count: int
    iteration: int = 0
    win_rate: float | None = None
    stop_reason: str | None = None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class GoalLedService:
    def __init__(
        self,
        *,
        run_root: str | Path,
        bootstrap_root: str | Path,
        gamepack_root: str | Path,
        runtime: AgentRuntime,
        arena: Arena,
        model: str,
        model_provider: str,
        runnable_opponent_ids: tuple[str, ...],
        public_leaderboard: tuple[dict[str, object], ...],
        game: str = "",
        roles: tuple[str, ...] = ("P0", "P1"),
        seeds: tuple[int, ...] = (1,),
        rollout_k: int = 1,
        opponent_policy: str = "progress",
        batch: int = 4,
        opponent_rank: int | None = None,
        opponent_start_rank: int | None = None,
        advance_min_matches: int = 2,
        advance_win_rate: float = 0.75,
        advance_streak: int = 1,
        #: progress 的晋级判据：最近 N 轮共 2N 局里至少 W 胜。
        #: 默认 2 轮 4 局 3 胜（见 opponent_policy.PROGRESS_WINDOW_*）。
        progress_window_iterations: int = PROGRESS_WINDOW_ITERATIONS,
        progress_window_wins: int = PROGRESS_WINDOW_WINS,
        match_parallelism: int = 1,
        prompt_override: str | None = None,
        experience_skills: bool = True,
        code_constraint: str = "any",
        history_mode: str = "full",
        rival_code_visible: bool = False,
        token_budget: int | None = None,
        wall_budget_s: int | None = None,
        epsilon: float = 0.02,
        measure_information_gain: bool = False,
        behavioral_ig_cases: int = 0,
        behavioral_ig_timeout_s: float = 900.0,
        behavioral_ig_coupling: str = COUPLING_COMMON_RANDOM,
        behavioral_ig_probe: str = "transcript_replay",
        agentbench_root: str | Path | None = None,
        iteration_mode: str = "lockstep",
        thread_rotate_context_tokens: int | None = None,
        thread_rotate_each_iteration: bool = False,
    ) -> None:
        self.root = Path(run_root).resolve()
        self.bootstrap_root = Path(bootstrap_root).resolve()
        self.gamepack_root = Path(gamepack_root).resolve()
        self.runtime = runtime
        self.arena = arena
        self.model = model
        self.model_provider = model_provider
        self.runnable_opponent_ids = frozenset(runnable_opponent_ids)
        self.public_leaderboard = tuple(
            sorted(public_leaderboard, key=lambda row: int(row["rank"]))
        )
        self.game = game
        self.roles = tuple(roles)
        self.seeds = tuple(seeds)
        self.rollout_k = max(1, int(rollout_k))
        # 一轮打几个对手（b）。k=1 之后这是主要的探索宽度旋钮。
        self.batch = max(1, int(batch))
        self.match_parallelism = max(1, int(match_parallelism))
        self.prompt_override = prompt_override
        self.experience_skills = experience_skills
        self.code_constraint = code_constraint
        # ``memoryless`` 是 ``last_only`` 的历史别名，统一归到新名字，
        # 免得工作区清理逻辑与提示词各认一半。
        if history_mode == "memoryless":
            history_mode = "last_only"
        if history_mode not in ("full", "last_only", "no_notes"):
            raise ValueError("history_mode must be full, last_only or no_notes")
        self.history_mode = history_mode
        # no_notes / last_only 都不写经验文档；last_only 还会每轮换新会话。
        if history_mode in ("no_notes", "last_only"):
            self.experience_skills = False
        self.rival_code_visible = rival_code_visible
        self.token_budget = token_budget
        self.wall_budget_s = wall_budget_s
        # ε 只用于经验分布的 KL 正则化（与 Plan II 的 measurement.epsilon 同一含义）。
        self.epsilon = float(epsilon)
        self.measure_information_gain = bool(measure_information_gain)
        # 决策级行为 IG：每轮拿几个配对 case 做线协议录制 + 冻结重放。
        # 0 = 关闭（此时 behavioral_ig 记 null 并写明是被关掉的，不是"测不出来"）。
        self.behavioral_ig_cases = max(0, int(behavioral_ig_cases))
        self.behavioral_ig_timeout_s = float(behavioral_ig_timeout_s)
        # 随机流耦合口径：非法值直接报错，不静默降级成"没耦合"。
        self.behavioral_ig_coupling = normalize_coupling(behavioral_ig_coupling)
        if behavioral_ig_probe not in BEHAVIORAL_IG_PROBES:
            raise ValueError(f"behavioral_ig_probe must be one of {BEHAVIORAL_IG_PROBES}")
        # 一个 run 只用一种探针，否则同一条曲线会混口径（见 _behavioral_ig 的说明）。
        self.behavioral_ig_probe = behavioral_ig_probe
        self.agentbench_root = (
            Path(agentbench_root).resolve() if agentbench_root is not None else None
        )
        if iteration_mode not in ("lockstep", "goal_autonomous"):
            raise ValueError("iteration_mode must be lockstep or goal_autonomous")
        self.iteration_mode = iteration_mode
        # 会话轮转阈值：见 config.RuntimeConfig.thread_rotate_context_tokens 的详注。
        # 目的只有一个——让 codex 的 remote compaction 永远不被触发。
        self.thread_rotate_context_tokens = (
            int(thread_rotate_context_tokens)
            if thread_rotate_context_tokens is not None and int(thread_rotate_context_tokens) > 0
            else None
        )
        self.thread_rotate_each_iteration = bool(thread_rotate_each_iteration)
        self.events = JsonlEventStore(self.root / "events.jsonl")
        self._usage_cursor = 0
        self.policy_name = canonical_policy_name(opponent_policy)
        self.advance_rule = AdvanceRule(
            min_matches=max(1, int(advance_min_matches)),
            win_rate=float(advance_win_rate),
            streak=max(1, int(advance_streak)),
        )
        self.policy = build_policy(
            opponent_policy,
            self._ladder(),
            target_rank=opponent_rank,
            start_rank=opponent_start_rank,
            seed=abs(hash(str(self.root))) % 10_000,
            # progress 的晋级判据：最近 N 轮共 2N 局至少 W 胜（默认 2 轮 4 局 3 胜）。
            # 与上面的 AdvanceRule 是**两套不同用途**：AdvanceRule 只服务
            # ladder_up/ladder_down 的"顺序征服进度"展示，而这里决定 progress
            # 的槽位什么时候往前挪。
            window_iterations=progress_window_iterations,
            window_wins=progress_window_wins,
        )
        # ★ b 的口径统一到**策略实际会打几个**，而不是配置里写了几个。
        #
        # 单目标策略（ladder_up / ladder_down / fixed_rank）无论 batch 写多少都
        # 只返回 1 个对手。exp2 主线就是 ladder_up 配着默认 batch: 4 在跑，
        # 于是事件账本、run-manifest.json、提示词里全写着 4，实际只打 1 个。
        #
        # 这不只是数字难看：提示词会跟 agent 说"这一版会被拿去打 4 个对手，
        # 你能从 4 份不同的回放里拿到证据"（它只会拿到 1 份），watch_runs 的
        # 对手数告警拿虚高的 b 当阈值因此失灵，而"一轮 = k × b × 座次"这个
        # 公式会让读账本的人以为丢了 3/4 的对局。
        self.batch = self.policy.effective_batch(self.batch)

    # -------------------------------------------------------- opponent history

    def _opponent_history(self) -> OpponentHistory:
        """迄今对每个对手的战绩（progress 晋级与 self 决策的依据）。

        从事件流重算而不是内存累加：断点续跑、事后复盘、换机器都要得到同一份
        进度。只算 ``status == complete`` 的局——一次沙箱故障不该被读成"打不赢"。

        除累计值外还给出**逐轮明细**：progress 的晋级判据是"最近 2 轮共 4 局
        至少 3 胜"，那是个滑动窗口，从累计值里还原不出来。
        """

        totals: dict[str, dict[str, float]] = {}
        rounds: dict[str, dict[int, dict[str, float]]] = {}
        for row in self._match_rows_from_events():
            if row.get("status") != "complete":
                continue
            opponent = row.get("opponent_id")
            if not isinstance(opponent, str):
                continue
            points = row.get("points")
            value = float(points) if isinstance(points, (int, float)) else 0.0
            entry = totals.setdefault(opponent, {"played": 0.0, "points": 0.0})
            entry["played"] += 1.0
            entry["points"] += value
            iteration = row.get("iteration")
            if isinstance(iteration, int):
                slot = rounds.setdefault(opponent, {}).setdefault(
                    iteration, {"played": 0.0, "points": 0.0}
                )
                slot["played"] += 1.0
                slot["points"] += value
        return OpponentHistory(totals, rounds)

    # ------------------------------------------------------------------ state

    def _ladder(self) -> tuple[LadderEntry, ...]:
        entries = [
            LadderEntry(
                opponent_id=str(row["opponent_id"]),
                rank=int(row["rank"]),
                score=(float(row["score"]) if row.get("score") is not None else None),
            )
            for row in self.public_leaderboard
            if row.get("opponent_id") is not None and row.get("rank") is not None
        ]
        if entries:
            return tuple(entries)
        # 没有 rank 锚点时（例如全匿名池）退化为可运行 id 顺序。
        return tuple(
            LadderEntry(opponent_id=item, rank=index + 1, score=None)
            for index, item in enumerate(sorted(self.runnable_opponent_ids))
        )

    @property
    def workspace(self) -> Path:
        return self.root / "workspace"

    @property
    def _state_path(self) -> Path:
        return self.root / "goal-led-state.json"

    def _append(self, event_type: str, payload: dict[str, object], key: str) -> None:
        self.events.append(FinalizedEvent.create(event_type, payload, key))

    def _append_telemetry(
        self, event_type: str, payload: dict[str, object], key: str
    ) -> None:
        """写**纯遥测**事件：撞键只记一条诊断，绝不打断迭代。

        科学事件（对局结果、指标、快照）必须严格幂等——撞键说明有真实的重复记账，
        应当立刻暴露。但遥测（token 用量之类）不参与任何结论：为了它把一次长跑
        终止是明显的代价错配。历史教训：一个 ``agent-usage`` 键冲突让 antwar
        在第 43 轮整个 run 崩掉，而丢失的信息只是几条 token 计数。
        """

        try:
            self._append(event_type, payload, key)
        except ValueError as error:
            with suppress(ValueError):
                self._append(
                    "TelemetryAppendSkipped",
                    {"event_type": event_type, "key": key, "error": str(error)},
                    f"telemetry-skipped:{key}",
                )

    # ----------------------------------------------------------- instructions

    def _rollout_instruction(self) -> str:
        """本轮要交几个候选，以及为什么。

        k=1（默认）与 k>1 是**两种不同的探索哲学**，提示词必须分开写：

        * k=1：把全部推理预算投进一个版本，广度由 b 个对手提供。这时要求
          "多个有差异的候选"是有害的——agent 会为了凑差异去改无关的阈值。
        * k>1：一轮探 k 个不同假设，此时才需要讲多样性下限与反例。
        """

        if self.rollout_k == 1:
            return (
                "本轮只交 **1 个候选**（k=1）。不要交多个版本、也不要为了"
                "凑差异去改无关的阈值——你的探索广度来自**对手**："
                f"这一个策略会被拿去打 {self.batch} 个对手，你能从 "
                f"{self.batch} 份不同的回放里拿到证据。"
                "所以请把全部思考投在这一个版本上：先读完上一轮每个对手的回放，"
                "找出**共性**的失败原因（对多个对手都吃亏的那条机制），"
                "优先修它；只对某一个对手成立的特判要谨慎，它会在别的对手身上亏回去。"
                "在 rationale 里写清：这一版改了什么、依据是哪个对手的哪段回放、"
                "预期在哪几个对手身上体现出来。"
                "候选目录规则：.agentbench/rollouts/<candidate_id>/ 会**叠加**到"
                "工作区当前版本之上，所以目录里必须放你改动过的 main.py"
                "（以及被改动的模块）；只放 README 会被判为无效候选并跳过。"
            )
        return (
            f"本轮提交 k={self.rollout_k} 个候选。"
            f"**k 个候选 = k 个不同的优化尝试，不是同一个尝试做 {self.rollout_k} 遍。**"
            "每个候选必须承载一个**独立的、可被单独证伪的假设**（不同的取胜路径 / "
            "不同的游戏机制 / 不同的资源分配哲学），并在 rationale 里写清"
            "「这个候选赌的是哪条机制、如果它错了会在回放里看到什么」。"
            "反例（会被判为伪多样性）：几个候选是同一份代码换了阈值、"
            "换了造兵顺序里的一个数字、或只改了一个常量——那样一轮只探到 1 个点，"
            f"却花掉了 {self.rollout_k} 倍的对局开销。"
            f"量化下限：任意两个候选之间的代码差异不少于 "
            f"{DIVERSITY_THRESHOLD_LINES} 行且落在不同的判断路径上；"
            "框架会逐轮度量 pairwise 差异并把结果写进反馈。"
            # ↓ 这段是 k>1 真正的用法，来自实测最好的那个 run（55 轮打进池内 #10）。
            # 缺了它，agent 会把 k 个候选当成"k 个独立的小改动"各自演进，
            # 于是每轮虽然有 k 份证据、却没人去合并它们——k 的复利效应完全丢失。
            "\n【怎么用好这 k 个候选：先分叉找到能赢的底盘，再在它上面分叉测增量】"
            "读上一轮反馈时，先分辨出**哪个候选赢了、它和输的那些在机制上差在哪**。"
            "那条差异就是已被对局证实的「底盘」。"
            "下一轮的 k 个候选应当**全部保留这条底盘**，然后各自测试一个独立增量；"
            "而不是让上一轮 4 个候选各自继续往下演进 4 条互不相干的线。"
            "换句话说：**分叉是为了合并**——一轮探 k 个方向，"
            "下一轮把胜出方向变成所有候选的共同前提，再从那里继续分叉。"
            "这样每一轮的 k 份证据都累积进同一条主线；"
            "各自演进则会让证据散开，k 轮之后你有 k 条半成品而不是一条强策略。"
            "如果上一轮**全部落败**，那说明当前底盘本身有问题："
            "这时该让 k 个候选走**更远**的方向（换取胜路径，而不是在同一路径上调参），"
            "先找到任何一个能赢的点。"
            "候选目录规则："
            ".agentbench/rollouts/<candidate_id>/ 会**叠加**到工作区当前版本之上，"
            "所以每个候选目录里必须放它自己的 main.py（以及被改动的模块）；"
            "只放 README 不算差异，会被判为无效候选并跳过。"
        )

    def _developer_instructions(self, *, iteration: int, cleared: int) -> str:
        """每轮下发的操作契约。

        这段话的第一职责不是"教它写代码"，而是**把容器契约讲清楚**：它只能读 6 样
        东西、只能通过一条通道交东西出去、对局完全不由它负责。历史上没讲清这件事的
        代价是实测的：一轮 850s 里约 530s 被 agent 拿去在容器内自己跑对局验证，
        而那件事既不该发生、也得不到任何关于人类对手的信息。
        """

        parts = [
            "【你在一个隔离容器里，没有网络】你只有一条对外通道：把 Action 写进 "
            ".agentbench/action.json。评测器（Evaluator）在容器之外，它持有人类选手池、"
            "负责跑完所有对局，然后把 Feedback 写回 feedback/ 交给你。",
            "**对局不由你负责，你也没有能力自己打比赛**：容器里没有对手代码、"
            "没有可运行的游戏后端、没有本地对战工具、没有训练脚本。"
            "不要写脚本去自评测、自对弈或估算胜率——那只会烧掉本轮的时间预算，"
            "而且得不到任何关于对手的信息（对手根本不在容器里）。"
            "你的策略强度**只由评测器回传的 Feedback 定义**。",
            "你可以读的东西只有这 6 样：gamepack/rules.md（规则）、"
            "gamepack/decision_space.yaml（决策空间）、"
            "gamepack/replay_skill.md 与 gamepack/replay_format.md（回放阅读指南）、"
            "leaderboard.json（人类排行榜，只有 id/rank/score）、"
            "你自己历次写的代码、以及 research/ 下你自己写的经验。"
            "另外还有一份格式示例与接口契约（CANDIDATE_CONTRACT.md / ai_example.py）。",
            "每轮你只需要做一件事：站在上面这些材料之上，把回放读透、想清楚，"
            "然后写出符合格式的新策略。",
            "通过 .agentbench/action.json 提交 Action（兼容 match_request.json）。"
            "字段：action_id、rollouts[{candidate_id}]、selected_rivals（对手 id 列表，"
            "写单个 selected_rival 也兼容）、roles、seeds、rationale。",
            self._rollout_instruction(),
            self.policy.instruction(
                iteration=iteration, batch=self.batch, history=self._opponent_history()
            ),
            f"本轮座次：{list(self.roles)}；seed：{list(self.seeds)}。"
            "座次名由游戏定义、由框架固定，**照抄这两个值**即可——"
            "写别的名字（例如把分轨游戏写成 P0/P1）不会改变实际对局，"
            "但会让 action.json 与真实赛程不一致，增加你自己复盘的难度。",
        ]
        if self.code_constraint == "if_else":
            parts.append(
                "策略语言约束：只允许可解释的 if-else / 规则表达式，禁止搜索、随机化与学习型组件。"
            )
        if self.experience_skills:
            parts.append(
                f"每轮必须更新 research/{EXPERIENCE_FILE}：记录本轮假设、证据、结论与下一步。"
                "**经验是这套框架里最值钱的东西**——对手的行为规律、你试过且失败的"
                "机制假设、以及「哪条游戏机制可以被用来超越对手」，都要写下来，"
                "否则下一轮等于从零开始。"
            )
        else:
            parts.append("本轮消融关闭了经验文档：不要维护 research/ 下的经验总结。")
        if self.rival_code_visible:
            parts.append(
                "本轮消融开启了『对手代码可见』：你可以阅读 rival-source/ 下提供的对手实现，"
                "可先蒸馏其有效行为再寻找弱点超越（distill_then_beat）。"
            )
        else:
            parts.append(
                "你看不到任何对手源码，只能从回放与排行榜推断对手行为。"
            )
        # ── HL 的核心理念。这段是这套框架区别于 RL 的全部所在，必须每轮重申 ──
        # 不写在这里的后果是实测过的：agent 会滑向"多加几条 if 试试"的随机游走，
        # 每轮都在动参数却说不出为什么，于是经验文档退化成 changelog、学习曲线走平。
        parts.append(
            "【怎么变强 —— 这套方法的核心】"
            "① **代码必须可解释**：每条规则都要能说出「在什么局面下、因为什么机制、所以做什么」。"
            "写不出理由的分支就是噪声，删掉它比留着更有价值。"
            "② **先模仿，再超越**：对手排名比你高，说明它的行为里有你还没有的东西。"
            "先从回放里把它的决策规律**蒸馏**出来（它开局造什么、在什么阈值上转攻、"
            "对你的动作如何反应），复制到能打平；然后回到 rules.md 找它**没有利用**的机制，"
            "用那条机制超过它。直接空想一个新战术、跳过模仿这一步，通常比抄它更慢。"
            "③ **经验是用来规划的，不是用来记账的**：research/ 里要写下"
            "「下一步该验证哪个假设、用什么证据判定它成立」，而不是罗列改了什么。"
            "每轮开始先读自己上轮写的下一步，先回答它。"
            "④ **一轮 = 一次有依据的改动**："
            + (
                (
                    f"这一轮只有 1 个候选，但它要面对 {self.batch} 个对手。"
                    "所以判断改对了没有的标准不是「赢了某一个」，而是"
                    "**在这批对手上的整体表现是否变好**（胜率、以及输的那几局分差是否收窄）。"
                    "改动幅度由证据决定：该改一条规则就改一条，该换整条战略就换整条；"
                    "但要避免为某一个对手写死特判——那会在其它对手身上亏回去。"
                )
                if self.rollout_k == 1
                else (
                    f"这一轮的 {self.rollout_k} 个候选之间要走**不同的取胜路径**"
                    "（例如：压制对手产能 / 抢经济后期成型 / 利用某条被对手忽视的机制），"
                    "彼此可以完全不像。"
                    "把同一份代码复制 k 份、各改一个阈值，是这套方法最常见也最贵的失效方式。"
                    "每个候选**改多少由证据决定**：从 rules.md 的机制和回放里读出"
                    "「哪里出了问题、为什么」，该改一条规则就改一条，该换整条战略就换整条。"
                )
            )
            + "框架对改动幅度没有偏好，只要求你在 rationale 里说清三件事："
            "依据的是哪条机制 / 哪段回放证据、赌的核心假设是什么、"
            "回放里出现什么现象就算它被证伪。"
            "⑤ **旧版本留得住**：把历史版本的策略留在文件里、保持可调用。"
            "这样 (a) 某条路线被证伪时，你能退回上一版的行为，而不必凭记忆重写；"
            "(b) 你能把两个历史版本**按局面组合**"
            "（例如「若第 20 回合我方仍无建筑就走旧节奏，否则走新节奏」）——"
            "这类组合要求旧版本没被覆盖掉。"
            "怎么实现随你：继承、组合/委托、可替换部件、显式参数表都行。"
            if self.history_mode == "full"
            else "⑤ **本轮只能看到上一版**（历史可见性消融）：见下面单文件约束一条。"
            "退回旧行为要靠你把它**写进当前这一个文件**（例如保留一个旧节奏的分支），"
            "而不是去 import 一个历史版本——历史版本已经不在工作区里了。"
            "⑥ **座次可以不对称**：P0 与 P1 的地形、先后手、对手行为都不同，"
            "最优策略也不必相同。你的入口类拿得到 player 参数，"
            "完全可以让 P0 和 P1 走**两条不同的策略**（例如内部持有两个委托对象，"
            "按 player 分派）。如果回放显示你在某个座次一直输、另一个座次能赢，"
            "那就该只改输的那个座次，而不是改一份共用代码去迁就两边。"
        )
        if self.history_mode == "last_only":
            # 单文件约束是 last_only 能成立的**前提**，不是风格建议。
            # 允许 v3.py import v2.py 的话，"只能看到上一版"就被 import 链绕过了：
            # agent 读一下 v2.py 等于看到了历史，消融失效而且从曲线上完全看不出来。
            parts.append(
                "【单文件约束（本轮消融的硬要求）】你只能看到**上一版**策略，"
                "所以策略代码必须全部写在一个文件里（main.py 加上游戏要求的入口即可），"
                "**不要**新建 v2.py / v3.py 之类的版本文件、也不要 import 任何历史版本模块。"
                "理由：如果新版本靠 import 旧版本来复用行为，那么『只能看到上一版』"
                "就名不副实了。你需要的旧行为请**抄进当前这一个文件**"
                "（可以留成一个分支或一个可切换的参数），这样它才随文件一起被继承下去。"
                "框架每轮只保留工作区里的当前策略与最近一份反馈，"
                "多写的版本文件会在下一轮被清掉，白费的是你自己的 token。"
            )
        return "".join(parts)

    def _context(self, prompt: str, *, iteration: int = 0, cleared: int = 0) -> RunContext:
        research = self.workspace / "research"
        objective = (
            self.prompt_override
            or "从规则出发，经公开回放持续研究并最终击败全部可运行人类选手（刷 SOTA Elo / 胜率）"
        )
        return RunContext(
            objective=objective,
            initial_prompt=prompt,
            base_instructions=(
                "你是容器里唯一的研究者。你的工作是：读规则与回放 → 想清楚 → 写策略代码。"
                "不要联网、不要找对手源码、不要自己跑对局或写评测脚本"
                "（容器里没有对手也没有后端，那么做只是白烧时间）、不要网格搜索。"
                "对局由容器外的评测器负责，你只通过 .agentbench/action.json 交货、"
                "通过 feedback/ 收货。"
            ),
            developer_instructions=self._developer_instructions(
                iteration=iteration, cleared=cleared
            ),
            cwd=self.workspace,
            candidate_root=self.workspace,
            gamepack_root=self.gamepack_root,
            research_root=research,
            human_pool_root=self.root / "hidden-human-pool",
            evaluator_root=self.root / "hidden-certification",
            runtime_workspace_roots=(self.workspace, self.gamepack_root, research),
            writable_workspace_roots=(self.workspace, research),
            model=self.model,
            model_provider=self.model_provider,
        )

    # ------------------------------------------------------------------ state

    def _write_state(self, thread_id: str, request_count: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        previous: dict[str, object] = {}
        if self._state_path.is_file():
            try:
                previous = json.loads(self._state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
        started_at = previous.get("started_at") or time.time()
        self._state_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "thread_id": thread_id,
                    "request_count": request_count,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        # 预算计时单独存，避免改变 goal-led-state.json 的既有 schema。
        (self.root / "goal-led-budget.json").write_text(
            json.dumps({"started_at": started_at}, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _load_state(self) -> tuple[str, int]:
        value = json.loads(self._state_path.read_text(encoding="utf-8"))
        thread_id = value.get("thread_id")
        count = value.get("request_count")
        if not isinstance(thread_id, str) or not thread_id or not isinstance(count, int):
            raise ValueError("goal-led state is invalid")
        return thread_id, count

    def _started_at(self) -> float:
        path = self.root / "goal-led-budget.json"
        if path.is_file():
            try:
                return float(json.loads(path.read_text(encoding="utf-8"))["started_at"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                return time.time()
        return time.time()

    # ------------------------------------------------------------ agent turns

    def _turn(
        self, session: AgentSession, prompt: str, *, iteration: int = 0, cleared: int = 0
    ) -> AgentSession:
        context = self._context(prompt, iteration=iteration, cleared=cleared)
        session = self.runtime.run_until_checkpoint(
            session,
            context,
            lambda event: getattr(event, "event_type", "") == "AgentTurnCompleted",
        )
        self._persist_agent_usage()
        return self._enforce_iteration_mode(session, iteration=iteration)

    def _enforce_iteration_mode(self, session: AgentSession, *, iteration: int) -> AgentSession:
        """迭代轮数由**框架**决定，而不是由 Goal 自己宣布"我做完了"。

        codex 的 Goal 可以在一个 turn 里把自己的状态置成 ``complete``；此时
        ``run_until_checkpoint`` 会提前返回，若不处理，后续轮次就跑不动了——
        "指定 N 轮"这个承诺会在统计上被 Goal 单方面打破。

        ``lockstep``（默认）：把这件事记成事件，并把 goal 重新 ``paused``，
        由驱动器继续推进剩余轮次。
        ``goal_autonomous``：尊重 Goal 的判断（提前收敛），用于对照实验。
        """

        status = str(getattr(session, "goal_status", "") or "")
        if status != "complete":
            return session
        self._append(
            "GoalDeclaredComplete",
            {"iteration": iteration, "iteration_mode": self.iteration_mode},
            f"goal-led-declared-complete:{iteration}",
        )
        if self.iteration_mode != "lockstep":
            return session
        pause = getattr(self.runtime, "pause", None)
        if callable(pause):
            session = pause(session)
        return session

    def _persist_agent_usage(self) -> None:
        """把 harness 内存里的用量事件落到 events.jsonl。

        harness（codex app-server / cc）只把映射后的事件放在内存列表里，而 token 预算
        守卫、token 曲线、成本核算都需要持久化数据。这里增量把带 ``total_tokens``
        的事件写进事件流；没有用量信息就什么都不写（绝不估算）。

        幂等键为什么**不能**用内存列表下标
        ----------------------------------
        曾经用的是 ``agent-usage:{内存下标}``。那个下标只在**当前进程**里单调，
        而账本是跨进程持久的，两者一旦错位就会撞键：

        * ``resume`` 起一个新进程，harness 的 ``events`` 从 0 重新计数，
          于是 ``agent-usage:0`` 会带着一份**不同的** payload 再写一次；
        * ``_append`` 撞上已存在的键且 payload 不同 ⇒ ``ValueError``，
          整个 run 被一个纯记账问题打死（实测 antwar 死在 ``agent-usage:64``）。

        改成按**账本里已有的 AgentTokenUsage 条数**发号：序号只依赖持久状态，
        重启后自然接着往下走，不会回退去抢占已用过的键。
        """

        events = getattr(self.runtime, "events", None)
        if not isinstance(events, list):
            return
        sequence = sum(
            1 for event in self.events.read_all() if event.event_type == "AgentTokenUsage"
        )
        for index in range(self._usage_cursor, len(events)):
            event = events[index]
            payload = getattr(event, "payload", None)
            if not isinstance(payload, Mapping):
                continue
            tokens = payload.get("total_tokens")
            if not isinstance(tokens, int):
                continue
            self._append_telemetry(
                "AgentTokenUsage",
                {
                    "harness": getattr(self.runtime, "harness", "codex"),
                    "total_tokens": tokens,
                    "input_tokens": payload.get("input_tokens"),
                    "output_tokens": payload.get("output_tokens"),
                    "source_event": getattr(event, "event_type", ""),
                },
                f"agent-usage:{sequence}",
            )
            sequence += 1
        self._usage_cursor = len(events)

    # --------------------------------------------------------------- requests

    def _request_path(self) -> Path | None:
        action = self.workspace / ".agentbench" / "action.json"
        if action.is_file():
            return action
        legacy = self.workspace / ".agentbench" / "match_request.json"
        if not legacy.is_file():
            return None
        try:
            request = MatchRequest.from_path(legacy)
        except ValueError:
            return legacy
        archived = (
            self.workspace
            / ".agentbench"
            / "processed-requests"
            / f"{request.request_id}.json"
        )
        return None if archived.is_file() else legacy

    def _consume_request(self) -> MatchRequest:
        path = self._request_path()
        if path is None:
            raise ValueError("Goal did not submit a new action.json")
        request = MatchRequest.from_path(path)
        if self.policy_name == SELF_DECIDE:
            unknown = [
                item
                for item in request.selected_opponents
                if item not in self.runnable_opponent_ids
            ]
            if unknown:
                raise ValueError(f"request names unknown or unrunnable opponent: {unknown}")
        archive = self.workspace / ".agentbench" / "processed-requests"
        archive.mkdir(parents=True, exist_ok=True)
        destination = archive / f"{request.request_id}.json"
        if destination.exists():
            raise ValueError(f"request_id was already consumed: {request.request_id}")
        path.replace(destination)
        return request

    def _assign_opponents(
        self, request: MatchRequest, *, iteration: int, cleared: int
    ) -> tuple[dict[str, tuple[str, ...]], str | None]:
        """返回 candidate_id -> 本轮要打的对手元组，以及"被框架覆盖"的说明。

        为什么值是**元组**而不是单个 id
        ------------------------------
        k=1 之后一轮的形状是"1 个策略 × b 个对手"：同一个候选要分别对 b 个对手
        各打一遍（再乘座次与 seed）。原实现的值是单个 id，做法是"第 i 个候选打
        第 i 个对手"，那在 k=1 时只会打到 b 个对手里的第一个 —— b 这个旋钮
        直接失效，而胜率曲线看起来毫无异常（只是恒定不动）。
        """

        if self.policy_name == SELF_DECIDE:
            chosen = tuple(
                item for item in request.selected_opponents if item in self.runnable_opponent_ids
            ) or (request.opponent_id,)
            return {cid: chosen for cid in request.candidate_ids}, None
        prescribed = self.policy.select(
            iteration=iteration,
            batch=self.batch,
            history=self._opponent_history(),
        )
        prescribed = tuple(item for item in prescribed if item in self.runnable_opponent_ids)
        if not prescribed:
            fallback = request.selected_opponents
            return {cid: fallback for cid in request.candidate_ids}, (
                f"policy {self.policy_name} produced no runnable opponent; kept Goal's choice"
            )
        mapping = {cid: prescribed for cid in request.candidate_ids}
        note = None
        if set(prescribed) != set(request.selected_opponents):
            note = (
                f"framework opponent policy {self.policy_name} overrode the Goal's "
                f"selection {list(request.selected_opponents)} with {list(prescribed)}"
            )
        return mapping, note

    def _effective_seeds(self, request: MatchRequest) -> tuple[int, ...]:
        if self.policy_name == SELF_DECIDE:
            return request.seeds
        return self.seeds or request.seeds

    def _effective_roles(self, request: MatchRequest) -> tuple[str, ...]:
        """本轮实际使用的座次。

        座次**名字**是框架级实验变量，由 A 仓 ``games/<game>/game.yaml`` 唯一定义；
        但"本轮打哪几个座次"是 agent 可以决定的（例如只想验证先手表现）。
        所以规则是：

        * 取 ``request.roles`` 与 ``self.roles`` 的交集 —— 保留 agent 的选择权；
        * 交集为空说明 agent 写的座次名**全都不合法**，此时退回 ``self.roles``
          （游戏定义的全部座次），而不是采纳它写的名字。

        为什么"交集为空时退回 self.roles"这件事很重要
        --------------------------------------------
        agent 会照抄 prompt 示例里的 ``P0`` / ``P1``。对 antwar 这类对称游戏恰好
        一致，但 rollman 的座次叫 ``rollman`` / ``ghost``——原实现在交集为空时
        回退到 ``request.roles``（即 agent 写的 P0/P1），于是每一局都以
        ``role P0 is not one of ('rollman', 'ghost')`` 失败。

        这个失败**极难排查**：指标上只显示"对局 0/N 完成"，看起来像对局跑不起来
        或候选有问题，完全看不出是座次名被 agent 的笔误带跑了。实测 rollman
        烟测连续两轮 8 局全灭，就栽在这里。

        ``self_decide`` 策略也不例外：它让 agent 自主选**对手**，不包括
        编造座次名。
        """

        allowed = set(self.roles)
        chosen = tuple(role for role in request.roles if role in allowed)
        return chosen or self.roles

    # ------------------------------------------------------------- snapshots

    _CODE_SUFFIXES = (".py", ".pyi")

    @property
    def candidate_interface(self) -> str | None:
        """GamePack 声明的候选入口契约（如 ``AI.choose_operations``）。

        物化后的 GamePack 里有 ``gamepack-manifest.json``；没有声明就返回 ``None``，
        前置校验只做"入口存在 + 语法 + 启动不崩"这些通用检查。
        """

        if getattr(self, "_candidate_interface_cache", "unset") != "unset":
            return self._candidate_interface_cache  # type: ignore[return-value]
        value: str | None = None
        manifest = self.gamepack_root / "gamepack-manifest.json"
        if manifest.is_file():
            try:
                document = json.loads(manifest.read_text(encoding="utf-8"))
                raw = document.get("candidate_interface")
                value = str(raw) if isinstance(raw, str) and raw.strip() else None
            except (json.JSONDecodeError, OSError):
                value = None
        self._candidate_interface_cache: str | None = value
        return value

    def _code_fingerprint(self, root: Path) -> str:
        """候选代码指纹（只看代码文件，忽略 README/笔记），用于识别"假多样性"。"""

        digest = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in self._CODE_SUFFIXES:
                continue
            relative = path.relative_to(root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()

    def _snapshot_root(self, request: MatchRequest, candidate_id: str) -> Path:
        """把一个候选物化成独立快照。

        语义（宽容且明确）：``.agentbench/rollouts/<candidate_id>/`` 是**叠加层**，
        覆盖在工作区当前版本之上。因此 Goal 只需为每个候选放差异文件；只放了
        README 的情况会被下面的指纹去重识别为"无代码差异"，并作为协议问题反馈回去，
        而不是让整个 run 崩掉。
        """

        overlay = self.workspace / ".agentbench" / "rollouts" / candidate_id
        destination = self.root / "snapshots" / candidate_id
        if destination.exists():
            raise ValueError(f"candidate snapshot already exists: {candidate_id}")
        ignored = shutil.ignore_patterns(
            "feedback", "research", "snapshots", "processed-requests", "rollouts", "runtime-tmp"
        )
        # 语义是**叠加**，与 k 无关：先铺工作区（含 candidate_support 平铺进来的
        # 运行时支撑：_bootstrap.py / common.py / 官方协议层 / SDK），再让 overlay
        # 覆盖同名文件。
        #
        # 曾经在这里加过一个"k=1 且 overlay 有 main.py 就只拷 overlay"的捷径，
        # 理由是"k=1 时 overlay 就是完整的一版"。那是错的：agent 的 overlay 只放
        # 它改动的文件（实测就是 main.py + ai.py 两个），只拷 overlay 会把
        # _bootstrap.py 落在外面，于是候选一启动就
        # ``ModuleNotFoundError: No module named '_bootstrap'`` ——
        # 而 main.py 顶部的 ``import _bootstrap`` 本来就是框架模板自己写的，
        # 候选完全无辜。预检把它判成 startup_crash，整轮 0 局对局，
        # 事件流里只留下一条"候选被拒"。
        shutil.copytree(self.workspace, destination, ignore=ignored)
        if overlay.is_dir():
            shutil.copytree(overlay, destination, dirs_exist_ok=True)
        if not (destination / "main.py").is_file():
            shutil.rmtree(destination, ignore_errors=True)
            raise ValueError(
                f"candidate {candidate_id} has no main.py after overlaying "
                f".agentbench/rollouts/{candidate_id}/ onto the workspace"
            )
        self._append(
            "GoalVersionSnapshot",
            {
                "candidate_id": candidate_id,
                "path": str(destination),
                "code_fingerprint": self._code_fingerprint(destination),
            },
            f"goal-led-snapshot:{candidate_id}",
        )
        return destination

    def _materialize_candidates(
        self, request: MatchRequest
    ) -> tuple[dict[str, Path], list[str]]:
        """物化全部候选，返回 (可评测候选, 协议问题说明)。

        - 缺 main.py 的候选被跳过并记录原因；
        - 与已有候选**代码完全相同**的候选被跳过（避免把"同一策略跑两遍"当成
          k 个候选，从而污染探索多样性的统计）。
        """

        usable: dict[str, Path] = {}
        notes: list[str] = []
        fingerprints: dict[str, str] = {}
        for candidate_id in request.candidate_ids:
            try:
                root = self._snapshot_root(request, candidate_id)
            except ValueError as error:
                notes.append(f"{candidate_id}: {error}")
                continue
            fingerprint = self._code_fingerprint(root)
            duplicate = next(
                (other for other, value in fingerprints.items() if value == fingerprint), None
            )
            if duplicate is not None:
                notes.append(
                    f"{candidate_id}: 与 {duplicate} 的代码完全相同（只有非代码文件不同），"
                    "已跳过；请把每个候选的差异化 main.py 放进 "
                    f".agentbench/rollouts/{candidate_id}/"
                )
                continue
            fingerprints[candidate_id] = fingerprint
            # ★ 前置校验：在烧掉 k×座次×seed 局对局之前，先确认这个包**能启动**。
            # 线上真实教训：缺一个 ai.py 让 5 轮迭代全部变成"0 回合判负"，
            # 曲线上是一条毫无信息的水平线。这里约 1 秒就能判掉，并给出可执行的原因。
            issues = check_candidate(
                candidate_id,
                root,
                candidate_interface=self.candidate_interface,
            )
            if issues:
                notes.extend(issue.as_note() for issue in issues)
                self._append(
                    "CandidatePreflightRejected",
                    {
                        "candidate_id": candidate_id,
                        "issues": [
                            {"kind": issue.kind, "detail": issue.detail} for issue in issues
                        ],
                    },
                    f"goal-led-preflight:{candidate_id}",
                )
                continue
            usable[candidate_id] = root
        return usable, notes

    # ----------------------------------------------------------- match rounds

    @staticmethod
    def _result_row(
        result: MatchResult,
        replay: Path | None,
        trace: Path | None,
        narration: Path | None = None,
        narration_note: str = "",
    ) -> dict[str, object]:
        payload = result.payload or {}
        # ★ 必须把对战器的诊断带进反馈：候选侧超时/非法/崩溃在契约里算"有效负局"
        # （status=complete, result=loss, rounds=0），如果反馈里只有"你输了"而没有
        # "因为第一帧格式非法"，Goal 就拿不到可执行信息，会在同一个坑里反复迭代。
        # 这是**自己**程序的失败原因，属于公开信息，不涉及对手代码。
        diagnostic = payload.get("game_error") or None
        return {
            "candidate_id": result.case.candidate_id,
            "opponent_id": result.case.opponent_id,
            "role": result.case.role,
            "seed": result.case.seed,
            "status": result.status,
            "result": result.result,
            "points": result.points,
            "score_margin": result.score_margin,
            "rounds": result.rounds,
            "replay_path": None if replay is None else str(replay),
            "trace_path": None if trace is None else str(trace),
            # 自然语言回放：agent 真正该读的那份（由 A 仓翻译）。
            "narration_path": None if narration is None else str(narration),
            "narration_note": narration_note,
            "error": result.error,
            "diagnostic": diagnostic,
            "evaluator_status": payload.get("evaluator_status"),
        }

    def _run_one(
        self,
        *,
        request: MatchRequest,
        candidate_id: str,
        candidate_root: Path,
        opponent_id: str,
        role: str,
        seed: int,
        feedback_root: Path,
    ) -> dict[str, object]:
        result = self.arena.run_case(
            MatchCase(candidate_id, opponent_id, role, seed), candidate_root
        )
        case_root = feedback_root / candidate_id / f"{role}-seed-{seed}"
        case_root.mkdir(parents=True, exist_ok=True)
        replay = None
        trace = None
        if result.replay_path is not None and Path(result.replay_path).is_file():
            replay = case_root / "replay.json"
            shutil.copy2(result.replay_path, replay)
        if result.trace_path is not None and Path(result.trace_path).is_file():
            trace = case_root / "public-trace.jsonl"
            shutil.copy2(result.trace_path, trace)
        # Feedback 通道的主体是**自然语言回放**，不是裸 JSON：
        # 裸 JSON 逼着 agent 每轮重写一遍解析代码（实测吃掉 63% 墙钟），
        # 而翻译只需在 A 仓做一次。裸 JSON 仍然保留在旁边，供它核对细节。
        payload = result.payload or {}
        narration, note = narrate_case(
            self.game,
            replay,
            case_root / NARRATION_FILENAME,
            match_id=f"{candidate_id}/{role}-seed-{seed}",
            perspective=role,
            opponent_id=opponent_id,
            official_winner=role if result.result == "win" else None,
            official_rounds=result.rounds,
            diagnostic=str(payload.get("game_error") or ""),
            agentbench_root=self.agentbench_root,
        )
        return self._result_row(result, replay, trace, narration, note)

    def _execute(
        self, request: MatchRequest, *, iteration: int, cleared: int
    ) -> tuple[Path, int, dict[str, object], list[dict[str, object]], dict[str, Path]]:
        feedback_root = self.workspace / "feedback" / request.request_id
        feedback_root.mkdir(parents=True, exist_ok=False)
        assignment, override_note = self._assign_opponents(
            request, iteration=iteration, cleared=cleared
        )
        roles = self._effective_roles(request)
        seeds = self._effective_seeds(request)
        self._append(
            "GoalMatchRequested",
            {
                "request_id": request.request_id,
                "candidate_ids": list(request.candidate_ids),
                "opponent_id": request.opponent_id,
                "opponent_assignment": {
                    cid: list(values) for cid, values in assignment.items()
                },
                "opponent_policy": self.policy_name,
                "policy_override": override_note,
                "roles": list(roles),
                "seeds": list(seeds),
                "rollout_k": self.rollout_k,
                "batch": self.batch,
                "rationale": request.rationale,
                # 轮次要写进事件：征服进度与曲线横坐标都靠它把逐局记录分组。
                "iteration": iteration,
            },
            f"goal-led-request:{request.request_id}",
        )

        jobs: list[dict[str, object]] = []
        usable, protocol_notes = self._materialize_candidates(request)
        for candidate_id, candidate_root in usable.items():
            # 一轮 = 候选 × 本轮 b 个对手 × 座次 × seed。
            # 对手维度是 k=1 改造的核心：胜率的分辨率就来自这里。
            for opponent_id in assignment.get(candidate_id, ()):
                for role in roles:
                    for seed in seeds:
                        jobs.append(
                            {
                                "candidate_id": candidate_id,
                                "candidate_root": candidate_root,
                                "opponent_id": opponent_id,
                                "role": role,
                                "seed": seed,
                            }
                        )
        if not jobs:
            # 协议问题（例如每个候选都没放 main.py）：不跑对局，但**不终止 run**，
            # 把可执行的修正说明写成反馈交回 Goal，让它下一轮自己改对。
            feedback = feedback_root / "feedback.json"
            summary = {
                "played": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "infra_errors": 0,
                "win_rate": None,
                "protocol_error": True,
                "protocol_notes": protocol_notes,
            }
            feedback.write_text(
                json.dumps(
                    {
                        "request_id": request.request_id,
                        "protocol_error": True,
                        "protocol_notes": protocol_notes,
                        "how_to_fix": (
                            "每个候选必须能独立跑起来：把该候选的 main.py（以及它依赖的模块）"
                            "放到 .agentbench/rollouts/<candidate_id>/；该目录会叠加到工作区之上。"
                            "候选之间必须有**代码**差异，只改 README 不算。"
                        ),
                        "summary": summary,
                        "matches": [],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            self._append(
                "GoalProtocolViolation",
                {"request_id": request.request_id, "notes": protocol_notes},
                f"goal-led-protocol:{request.request_id}",
            )
            return feedback, 0, summary, [], {}

        rows: list[dict[str, object]] = []
        if self.match_parallelism > 1 and len(jobs) > 1:
            # 并行前先串行预热后端构建（A 的 build cache 有 quarantine 竞态）。
            warmup = getattr(self.arena, "warmup", None)
            if callable(warmup):
                try:
                    warmup(Path(str(jobs[0]["candidate_root"])))
                except Exception:  # noqa: BLE001 - 预热失败交由正式对局报诊断
                    pass
            with ThreadPoolExecutor(max_workers=self.match_parallelism) as pool:
                futures = [
                    pool.submit(
                        self._run_one,
                        request=request,
                        feedback_root=feedback_root,
                        candidate_id=str(job["candidate_id"]),
                        candidate_root=Path(str(job["candidate_root"])),
                        opponent_id=str(job["opponent_id"]),
                        role=str(job["role"]),
                        seed=int(job["seed"]),  # type: ignore[arg-type]
                    )
                    for job in jobs
                ]
                rows = [future.result() for future in futures]
        else:
            for job in jobs:
                rows.append(
                    self._run_one(
                        request=request,
                        feedback_root=feedback_root,
                        candidate_id=str(job["candidate_id"]),
                        candidate_root=Path(str(job["candidate_root"])),
                        opponent_id=str(job["opponent_id"]),
                        role=str(job["role"]),
                        seed=int(job["seed"]),  # type: ignore[arg-type]
                    )
                )
        for row in rows:
            self._append(
                "GoalMatchCompleted",
                {"request_id": request.request_id, **row},
                # 幂等键必须带 opponent：k=1 之后同一个候选要在同一轮里打 b 个
                # 不同对手，只用 (候选, 座次, seed) 会让第 2..b 个对手的结果
                # 撞键，b>1 时对局悄悄丢掉 (b-1)/b。
                "goal-led-match:{rid}:{cid}:{opp}:{role}:{seed}".format(
                    rid=request.request_id,
                    cid=row["candidate_id"],
                    opp=row["opponent_id"],
                    role=row["role"],
                    seed=row["seed"],
                ),
            )

        summary = self._summarize(rows)
        if protocol_notes:
            summary["protocol_notes"] = protocol_notes
        feedback = feedback_root / "feedback.json"
        feedback.write_text(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "rationale": request.rationale,
                    "opponent_policy": self.policy_name,
                    "opponent_assignment": {
                        candidate: assignment[candidate] for candidate in usable
                    },
                    "policy_override": override_note,
                    "protocol_notes": protocol_notes,
                    "summary": summary,
                    "matches": rows,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_combined_narration(feedback_root, rows)
        return feedback, len(rows), summary, rows, dict(usable)

    def _write_combined_narration(
        self, feedback_root: Path, rows: Sequence[Mapping[str, object]]
    ) -> Path | None:
        """把本轮全部对局的自然语言复盘并成一份 ``all-replays.md``。

        为什么值得单独做一份：一轮的墙钟 ≈ **往返次数** × 单次延迟，而单次延迟
        几乎与上下文大小无关（实测同一个小问题在 reasoning_effort=high 下要 9.8s，
        输入只有 58 tokens；一轮 30~68 次往返里工具执行只占 1%）。也就是说
        agent 每多读一个文件，就要多付一次约 13 秒的"思考税"。
        antwar2 一轮 8 局、8 份 replay.md 分 8 次读，光这一项就是 ~100s。

        并成一份后 agent 一次读完：多出来的 prefill 只有几秒且有 95% 缓存命中，
        省下的是 7 次思考税。单局文件**照旧保留**——要核对某一局的具体数字时仍然
        用得上，也不动冻结产物的完整性。
        """

        sections: list[str] = []
        index: list[str] = []
        for position, row in enumerate(rows, start=1):
            label = (
                f"{row.get('candidate_id')} · {row.get('role')} · seed {row.get('seed')} "
                f"vs {row.get('opponent_id')}"
            )
            verdict = f"{row.get('status')}/{row.get('result')}"
            index.append(f"{position}. {label} — {verdict}")
            body: str
            path = row.get("narration_path")
            if isinstance(path, str) and Path(path).is_file():
                body = Path(path).read_text(encoding="utf-8", errors="replace").strip()
            else:
                # 没有复盘本身就是信息（评测器失败/崩溃），要如实写出来而不是留空。
                note = row.get("narration_note") or row.get("diagnostic") or row.get("error")
                body = f"（这一局没有自然语言复盘：{note or '原因未记录'}）"
            sections.append(f"## {position}. {label}\n\n判决：{verdict}\n\n{body}\n")

        if not sections:
            return None
        document = (
            "# 本轮全部对局复盘\n\n"
            "这一份是本轮每一局 `replay.md` 的合并版，**读这一份就够了**。\n"
            "需要核对某一局的具体数字时，单局文件仍在各自的 "
            "`<candidate>/<role>-seed-<seed>/replay.md`。\n\n"
            "## 目录\n\n" + "\n".join(index) + "\n\n---\n\n" + "\n---\n\n".join(sections)
        )
        combined = feedback_root / "all-replays.md"
        combined.write_text(document, encoding="utf-8")
        return combined

    # ------------------------------------------------------------ aggregation

    def _opponent_score(self, opponent_id: str) -> float | None:
        for row in self.public_leaderboard:
            if str(row.get("opponent_id")) == opponent_id and row.get("score") is not None:
                return float(row["score"])  # type: ignore[arg-type]
        return None

    def _pool_elo(self) -> dict[str, object] | None:
        """候选相对**固定人类池**的 Elo（累积锚定 MLE）。

        用该 run 迄今**全部** complete 官方对局（跨轮、跨对手），以每个对手在人类池
        里的 Elo 为固定锚点做一维 BT 极大似然估计。这是唯一跨轮/跨游戏可比的量：

        * ``elo_vs_opponent`` 只看当轮胜率、锚点还随对手切换而变，有序课程下
          "换更强的对手"会表现成假下降；
        * 这里换对手只是换锚点，尺子（人类池 Elo 刻度）始终不变。

        不额外烧机时——用的就是已经打完的局。详见 ``domain/pool_elo.py``。
        """

        rows = [row for row in self._match_rows_from_events() if row.get("status") == "complete"]
        if not rows:
            return None
        anchors = {
            str(row.get("opponent_id")): (
                float(row["score"]) if row.get("score") is not None else None  # type: ignore[arg-type]
            )
            for row in self.public_leaderboard
        }
        estimate = estimate_pool_elo(rows, anchors)
        return None if estimate is None else estimate.as_dict()

    def _summarize(self, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
        scored = [row for row in rows if row.get("status") == "complete"]
        infra = [row for row in rows if row.get("status") != "complete"]
        wins = sum(1 for row in scored if row.get("result") == "win")
        draws = sum(1 for row in scored if row.get("result") == "draw")
        losses = sum(1 for row in scored if row.get("result") == "loss")
        played = len(scored)
        win_rate = (wins + 0.5 * draws) / played if played else None

        per_candidate: dict[str, dict[str, float]] = {}
        for row in scored:
            entry = per_candidate.setdefault(
                str(row["candidate_id"]), {"played": 0.0, "points": 0.0, "margin": 0.0}
            )
            entry["played"] += 1
            entry["points"] += float(row.get("points") or 0.0)
            if isinstance(row.get("score_margin"), (int, float)):
                entry["margin"] += float(row["score_margin"])  # type: ignore[arg-type]
        best_candidate = None
        best_key: tuple[float, float] | None = None
        for candidate_id, entry in per_candidate.items():
            played_count = entry["played"] or 1.0
            # 排序键用 (得分率, 平均分差)：全败的一轮里所有候选得分率都是 0，
            # 此时只有分差能区分"差一点"和"被打爆"。不这样做的话 best_candidate
            # 退化成字典序，下一轮的基线就是随机挑的——那才是真正的 0 信号。
            key = (entry["points"] / played_count, entry["margin"] / played_count)
            if best_key is None or key > best_key:
                best_candidate, best_key = candidate_id, key
        best_rate = None if best_key is None else best_key[0]

        # 单对手 Elo 估计：以对手已知 Elo 为锚，按胜率做 logistic 反推。
        elo_estimate = None
        opponent_ids = sorted({str(row["opponent_id"]) for row in scored})

        # 逐对手战绩。k=1 × b 个对手之后这是最有信息量的一块：
        # 总胜率 0.5 可能是"打赢弱的两个、输给强的两个"（正常），
        # 也可能是"对每个都五五开"（完全不同的局面），只看总胜率分不出来。
        by_opponent: dict[str, dict[str, float]] = {}
        for row in scored:
            entry = by_opponent.setdefault(
                str(row["opponent_id"]), {"played": 0.0, "points": 0.0}
            )
            entry["played"] += 1.0
            entry["points"] += float(row.get("points") or 0.0)
        win_rate_by_opponent = {
            opponent: {
                "played": int(entry["played"]),
                "win_rate": round(entry["points"] / entry["played"], 4),
                "opponent_elo": self._opponent_score(opponent),
            }
            for opponent, entry in sorted(by_opponent.items())
            if entry["played"] > 0
        }

        # 在**全池刻度**上反解这一版的 Elo。
        #
        # 口径：拿"该候选这一轮真打过的那几局" + 冻结人类池的锚点，做锚定 BT/MLE
        # （``domain.pool_elo.estimate_pool_elo``，带 2 场虚拟平局的正则）。
        # 它回答的是"这一版插进全池会排在哪"，而不是"它对这 4 个人的胜率"。
        #
        # 为什么不再自己手写 logistic 反解（原实现）
        # -----------------------------------------
        # 原实现是逐对手 ``anchor + 400·log10(p/(1-p))`` 再取平均，并把 p 钳到
        # [0.02, 0.98] 以避免 log(0)。那个钳位会制造**假的平坦曲线**：
        # fix 组固定打榜单前 4 名、胜率恒 0，于是 p 恒被钳成 0.02，
        # 反解恒等于 2107.5 − 676 = 1431.4 —— 实测 14 轮一动不动全是 1431.37。
        # 看图会得出"这一组完全没在学"的结论，而同期分差从 −36.12 收窄到 −28.12。
        #
        # 正则 MLE 没有这个毛病：全败时 θ 由先验拉住而不是被硬钳，
        # 且估计值取决于**对手锚点的具体分布**，所以换了对手它就会动。
        # 另外它与慢通道（全池实测）用的是同一个估计器，两条 Elo 曲线才同尺度可比。
        estimate = estimate_pool_elo(
            [
                {"opponent_id": str(row["opponent_id"]), "points": row.get("points")}
                for row in scored
            ],
            {opponent: self._opponent_score(opponent) for opponent in by_opponent},
        )
        elo_estimate = None if estimate is None else estimate.elo
        elo_estimate_detail = None if estimate is None else estimate.as_dict()

        # 连续奖励：胜负是二值的，但**分差**不是。
        #
        # 为什么必须报它：只看胜率，一轮全败就等于 0 信号，agent 无从判断"这次改动
        # 是差一点还是差很远"。而 score_margin（终局分差）与 rounds（撑了多少回合）
        # 都是逐局连续量，早就落在事件里，只是从来没端到 agent 面前。实测 antwar2
        # 对 rank1 连续 15 轮胜率恒为 0，如果那 15 轮的分差在收窄，那就是**有**梯度的。
        #
        # 逐候选报，因为"哪个候选离赢最近"才是下一轮该继续推的方向。
        margins = [
            float(row["score_margin"])
            for row in scored
            if isinstance(row.get("score_margin"), (int, float))
        ]
        per_candidate_margin: dict[str, list[float]] = {}
        per_candidate_rounds: dict[str, list[int]] = {}
        for row in scored:
            candidate = str(row["candidate_id"])
            if isinstance(row.get("score_margin"), (int, float)):
                per_candidate_margin.setdefault(candidate, []).append(
                    float(row["score_margin"])  # type: ignore[arg-type]
                )
            if isinstance(row.get("rounds"), int):
                per_candidate_rounds.setdefault(candidate, []).append(int(row["rounds"]))
        # 逐候选战绩。**必须带胜率**，不能只有分差。
        #
        # k>1 的全部价值在于"分叉找到能赢的底盘 → 下一轮所有候选都保留它、
        # 再从那里分叉测增量"。要做这件事，agent 首先得知道**哪个候选赢了**。
        # 这个字段以前只有 margin，胜率得它自己按 candidate_id 分组数 matches
        # 才能算出来 —— 那一步很容易被跳过，于是 k 个候选各自独立往下演进，
        # k 轮之后是 k 条半成品而不是一条强策略。
        #
        # 实测最好的那个 run（55 轮进池内 #10）第 5 轮的 rationale 就是
        # "v003-cluster-storm 是首个 2/2 候选……本轮四个候选都保留这条底盘"——
        # 那正是这个字段该支撑的动作。
        #
        # 按 (胜率, 平均分差) 降序排：第一项就是本轮的底盘，不用再比对。
        margin_by_candidate = {
            candidate: {
                "win_rate": round(
                    per_candidate[candidate]["points"]
                    / (per_candidate[candidate]["played"] or 1.0),
                    4,
                ),
                "played": int(per_candidate[candidate]["played"]),
                "is_best": candidate == best_candidate,
                "mean": round(sum(values) / len(values), 2),
                "best": round(max(values), 2),
                "worst": round(min(values), 2),
                "rounds_mean": (
                    round(
                        sum(per_candidate_rounds[candidate])
                        / len(per_candidate_rounds[candidate]),
                        1,
                    )
                    if per_candidate_rounds.get(candidate)
                    else None
                ),
            }
            for candidate, values in sorted(
                per_candidate_margin.items(),
                key=lambda item: (
                    -(
                        per_candidate[item[0]]["points"]
                        / (per_candidate[item[0]]["played"] or 1.0)
                    ),
                    -(sum(item[1]) / len(item[1])),
                ),
            )
        }

        # 失败原因汇总：0 回合判负这种情况没有回放可看，诊断是唯一的可执行线索。
        diagnostics: dict[str, int] = {}
        for row in rows:
            detail = row.get("diagnostic")
            if isinstance(detail, str) and detail.strip():
                key = detail.strip()[:200]
                diagnostics[key] = diagnostics.get(key, 0) + 1
        zero_round_losses = sum(
            1
            for row in scored
            if row.get("result") == "loss" and int(row.get("rounds") or 0) == 0
        )

        return {
            "played": played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "infra_errors": len(infra),
            "win_rate": win_rate,
            "elo_vs_opponent": elo_estimate,
            # 估计的可信度线索：用了几局、其中几局有锚点、得分率、锚点均值、方法名。
            # 没有它就无法判断一个 Elo 数字该不该信（2 局与 200 局的估计画在
            # 同一条曲线上，看起来一模一样）。
            "elo_estimate": elo_estimate_detail,
            "opponent_ids": opponent_ids,
            "win_rate_by_opponent": win_rate_by_opponent,
            "best_candidate_id": best_candidate,
            "best_candidate_win_rate": best_rate,
            "zero_round_losses": zero_round_losses,
            # 连续量：全败的一轮里，这些才是唯一的梯度来源。
            "margin_mean": round(sum(margins) / len(margins), 2) if margins else None,
            "margin_best": round(max(margins), 2) if margins else None,
            "margin_by_candidate": margin_by_candidate,
            "diagnostics": [
                {"detail": detail, "count": count}
                for detail, count in sorted(
                    diagnostics.items(), key=lambda item: (-item[1], item[0])
                )
            ],
        }

    def _token_total(self) -> int | None:
        """全 run 累计 token 用量 = 每次模型请求的用量之和（账单口径）。

        为什么是"求和"而不是"取峰值"
        ----------------------------
        这里踩过一个把曲线彻底画错的坑，值得写清楚。``AgentTokenUsage`` 的
        ``total_tokens`` 来自 codex 的 ``thread/tokenUsage/updated``，而
        :mod:`agentbench_hl.adapters.codex_goal.event_mapper` 取的是
        ``tokenUsage.last`` —— **那一次请求**的用量，不是 thread 的累计量。

        原实现按"会话累计值"语义处理：以会话事件切段、段内取 max、跨段相加。
        语义反了，后果是实测的：

        * ``sota-antwar`` 第 2~11 轮 token 全报 137631，**连续 10 轮一动不动**；
        * 6 个新 run 逐轮精确翻倍（snakego4: 112526 → 260286 → 520572 →
          1041144），因为新增一个 rotate 边界就把同一份用量再叠一遍。

        真实口径很朴素：agent 每发一次模型请求就要重发整个上下文，那次请求的
        ``input + output`` 就是那次的花费，全 run 花费 = 逐次求和。实测 r4b
        4 轮 209 次请求合计 13.79M token，而原口径只报 0.99M（低估 14 倍）。

        ⚠️ 修好之后 ``budget.tokens`` 才第一次真正生效。老配置里写的 3M 是在
        低估 14 倍的口径下拍的，直接沿用会让 run 在第 1 轮就被预算掐死；
        新模板因此不设 token 预算，改用 ``max_iterations`` 收敛。
        """

        return self._token_event_sum()

    def _token_event_sum(self) -> int | None:
        """逐次请求用量求和。

        只认 ``AgentTokenUsage``：其它事件（例如指标事件自己回写的
        ``total_tokens``）也带同名字段，一起加进来会自我叠加。
        """

        total = 0
        seen = False
        for event in self.events.read_all():
            if event.event_type != "AgentTokenUsage":
                continue
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            value = payload.get("total_tokens")
            if isinstance(value, int):
                total += value
                seen = True
        return total if seen else None

    def _match_rows_from_events(self) -> list[dict[str, object]]:
        """从事件流还原逐局记录（带轮次），供征服判定与轨迹计数使用。"""

        rows: list[dict[str, object]] = []
        iteration = 0
        for event in self.events.read_all():
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            if event.event_type == "GoalMatchRequested":
                value = payload.get("iteration")
                iteration = int(value) if isinstance(value, int) else iteration + 1
                continue
            if event.event_type != "GoalMatchCompleted":
                continue
            rows.append(
                {
                    "iteration": iteration,
                    "opponent_id": payload.get("opponent_id"),
                    "status": payload.get("status"),
                    "points": payload.get("points"),
                    "rounds": payload.get("rounds"),
                    "replay_path": payload.get("replay_path"),
                }
            )
        return rows

    def _conquest_state(self) -> ConquestState | None:
        """有序课程（ladder_up / ladder_down）的当前进度；其它策略返回 None。

        只有单目标顺序课程才有"目标序列"这个概念。``progress`` 在 b>1 时是
        **b 个并行槽位**，每个槽位有自己的进度，压不成一个标量游标——它的进度
        由 ``win_rate_by_opponent`` 与策略自己的窗口计算表达。
        """

        sequence_of = getattr(self.policy, "target_sequence", None)
        if not callable(sequence_of):
            return None
        sequence = sequence_of()
        if not sequence:
            return None
        rows = self._match_rows_from_events()
        return conquest_evaluate(
            sequence, conquest_rounds(rows), rule=self.advance_rule
        )

    def _cleared_count(self) -> int:
        """课程推进用的"已征服对手数"。

        有序课程按目标序列判定（只有对**当前目标**稳定达标才 +1，见
        :mod:`application.conquest`）；其它策略退回"历史上得分率 > 0.5 的对手数"，
        它只用于生成指令文案，不影响对手选择。
        """

        state = self._conquest_state()
        if state is not None:
            return state.cleared

        outcomes: dict[str, list[float]] = {}
        for event in self.events.read_all():
            if event.event_type != "GoalMatchCompleted":
                continue
            payload = event.payload
            if payload.get("status") != "complete":
                continue
            opponent = str(payload.get("opponent_id"))
            outcomes.setdefault(opponent, []).append(float(payload.get("points") or 0.0))
        return sum(1 for points in outcomes.values() if points and sum(points) / len(points) > 0.5)

    def _trajectories_seen(self) -> int:
        """agent 已经能读到的**完整轨迹**数（实验 2 的横坐标之一）。

        判据：非影子对局、状态 complete、且回放文件真的落到了反馈目录里
        （0 回合判负没有回放，不算"看过一条轨迹"）。
        """

        total = 0
        for row in self._match_rows_from_events():
            if row.get("shadow") or row.get("status") != "complete":
                continue
            if row.get("replay_path"):
                total += 1
        return total

    def _iteration_index(self) -> int:
        return sum(
            1 for event in self.events.read_all() if event.event_type == "GoalFeedbackDelivered"
        )

    def _previous_win_rate(self) -> float | None:
        latest = None
        for event in self.events.read_all():
            if event.event_type != "IterationMetricsFinalized":
                continue
            value = event.payload.get("win_rate")
            if isinstance(value, (int, float)):
                latest = float(value)
        return latest

    # ------------------------------------------------------- information gain

    def _snapshot_paths(self) -> dict[str, Path]:
        """所有已物化候选快照（candidate_id → 路径），来自事件流。"""

        table: dict[str, Path] = {}
        for event in self.events.read_all():
            if event.event_type != "GoalVersionSnapshot":
                continue
            candidate = event.payload.get("candidate_id")
            path = event.payload.get("path")
            if isinstance(candidate, str) and isinstance(path, str):
                table[candidate] = Path(path)
        return table

    def _previous_champion(self) -> tuple[str, Path] | None:
        """上一轮最优候选（作为本轮的配对基线）。"""

        champion: str | None = None
        for event in self.events.read_all():
            if event.event_type != "IterationMetricsFinalized":
                continue
            value = event.payload.get("best_candidate_id")
            if isinstance(value, str) and value:
                champion = value
        if champion is None:
            return None
        path = self._snapshot_paths().get(champion)
        if path is None or not path.is_dir():
            return None
        return champion, path

    def _run_shadow_matches(
        self,
        *,
        request: MatchRequest,
        baseline_id: str,
        baseline_root: Path,
        cases: Sequence[Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """把基线策略放到**完全相同**的 case 上再跑一遍（不计入本轮胜率）。"""

        shadow_root = self.workspace / "feedback" / request.request_id / "shadow"
        jobs = [
            {
                "opponent_id": str(case["opponent_id"]),
                "role": str(case["role"]),
                "seed": int(case["seed"] or 0),  # type: ignore[arg-type]
            }
            for case in cases
        ]
        if not jobs:
            return []

        def run(job: Mapping[str, object]) -> dict[str, object]:
            return self._run_one(
                request=request,
                feedback_root=shadow_root,
                candidate_id=baseline_id,
                candidate_root=baseline_root,
                opponent_id=str(job["opponent_id"]),
                role=str(job["role"]),
                seed=int(job["seed"]),  # type: ignore[arg-type]
            )

        if self.match_parallelism > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=self.match_parallelism) as pool:
                rows = list(pool.map(run, jobs))
        else:
            rows = [run(job) for job in jobs]
        for row in rows:
            self._append(
                "ShadowMatchCompleted",
                {"request_id": request.request_id, "baseline_id": baseline_id, **row},
                "goal-led-shadow:{rid}:{role}:{seed}".format(
                    rid=request.request_id, role=row["role"], seed=row["seed"]
                ),
            )
        return rows

    # ------------------------------------------------ 决策级行为信息增益

    @property
    def behavioral_ig_root(self) -> Path:
        """录制/重放产物目录。``<root>/transcripts`` 需要对沙箱内选手进程可写。"""

        return self.root / "behavioral-ig"

    def _inprocess_behavioral_ig(
        self,
        *,
        baseline_root: Path,
        candidate_root: Path,
        pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    ) -> tuple[float | None, str]:
        """进程内策略探针（Plan II 的 ``ai.py`` 契约）：在公开回放上重驱动策略。

        它能拿到**精确**的每态合法支撑集，保真度高于通用线协议探针，所以优先用它；
        但它要求候选包能被 import 成一个策略对象，Plan I 的 ``main.py`` 进程契约不满足。
        """

        from agentbench_hl.domain.policy import (  # noqa: PLC0415 - 仅在需要时加载
            compare_decisions,
            compare_policy_episode,
        )

        binding, reason = probe_availability(self.game, candidate_root)
        if binding is None:
            return None, reason
        values: list[float] = []
        for _baseline_row, candidate_row in pairs:
            replay = candidate_row.get("replay_path")
            if not isinstance(replay, str) or not Path(replay).is_file():
                continue
            match_id = "{role}-seed-{seed}".format(
                role=candidate_row.get("role"), seed=candidate_row.get("seed")
            )
            role = str(candidate_row.get("role"))
            try:
                baseline_trace = binding.probe(
                    baseline_root, replay, match_id=match_id, role=role
                )
                candidate_trace = binding.probe(
                    candidate_root, replay, match_id=match_id, role=role
                )
            except Exception as error:  # noqa: BLE001 - 探针失败必须如实记录
                return None, f"in-process probe failed: {type(error).__name__}: {error}"
            samples = compare_policy_episode(baseline_trace, candidate_trace)
            comparison = compare_decisions(samples, epsilon=self.epsilon)
            if comparison.mean_kl_nats is not None:
                values.append(float(comparison.mean_kl_nats))
        if not values:
            return None, "in-process probe available but no paired replay produced decisions"
        return sum(values) / len(values), f"measured with {binding.schema}"

    def _wire_behavioral_ig(
        self,
        *,
        request: MatchRequest,
        iteration: int,
        baseline_id: str,
        baseline_root: Path,
        candidate_id: str,
        candidate_root: Path,
        pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    ) -> BehavioralIgMeasurement:
        """通用线协议探针（Plan I 的 ``main.py`` 进程契约）。

        口径来自 A 仓 ``games/<game>/decision_space.yaml`` 的 ``information_gain:`` 段；
        流程见 :mod:`agentbench_hl.application.behavioral_ig`。完整 trace 会落到
        ``<run>/behavioral-ig/trace-<iteration>.json``，曲线脚本读它。
        """

        spec, note = load_information_gain_spec(
            self.game, agentbench_root=self.agentbench_root
        )
        if spec is None:
            return BehavioralIgMeasurement(value=None, reason=note)
        cases = tuple(
            BehavioralIgCase(
                opponent_id=str(candidate_row.get("opponent_id")),
                role=str(candidate_row.get("role")),
                seed=int(candidate_row.get("seed") or 0),
            )
            for _baseline_row, candidate_row in pairs
        )
        feedback_root = self.workspace / "feedback" / request.request_id / "behavioral-ig"

        def run_match(
            player_id: str, root: Path, case: BehavioralIgCase
        ) -> Mapping[str, object]:
            return self._run_one(
                request=request,
                feedback_root=feedback_root,
                candidate_id=player_id,
                candidate_root=root,
                opponent_id=case.opponent_id,
                role=case.role,
                seed=case.seed,
            )

        measurement = measure_behavioral_ig(
            spec=spec,
            epsilon=self.epsilon,
            work_root=self.behavioral_ig_root,
            baseline_id=baseline_id,
            baseline_root=baseline_root,
            candidate_id=candidate_id,
            candidate_root=candidate_root,
            cases=cases,
            run_match=run_match,
            replay_timeout_s=self.behavioral_ig_timeout_s,
            max_cases=self.behavioral_ig_cases,
            coupling=self.behavioral_ig_coupling,
            # 逐决策点的真实 |A(s)|。有状态探针的游戏走精确枚举，
            # 其余游戏自动回落到字母表常量并在 support_mode 里如实标注。
            #
            # agentbench_root 必须传下去：rollman 这类候选包不自带后端 core/ 的
            # 游戏，探针要靠它去 A 仓定位合法动作枚举器。缺了它探针会直接失败，
            # 而失败会被当成"没有探针"静默回落到近似口径。
            support_provider=provider_for(
                self.game, agentbench_root=self.agentbench_root
            ),
        )
        trace_path = self.behavioral_ig_root / f"trace-{iteration:04d}.json"
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(
                json.dumps(measurement.trace_document(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            # 落盘失败不该让整轮迭代崩掉：数值已经在事件里了。
            pass
        return measurement

    def _behavioral_ig(
        self,
        *,
        request: MatchRequest,
        iteration: int,
        baseline_id: str,
        baseline_root: Path,
        candidate_id: str,
        candidate_root: Path,
        pairs: Sequence[tuple[Mapping[str, object], Mapping[str, object]]],
    ) -> tuple[float | None, str, dict[str, object]]:
        """决策级行为信息增益（nats/决策）。

        **一个 run 只用一种探针**（``behavioral_ig_probe``）。这不是洁癖：两种探针的
        |A| 与动作口径不同，如果按"先试进程内、失败退线协议"逐轮自动切换，同一条曲线上
        相邻两点就可能来自不同口径，斜率完全没有意义。实测 antwar2 的进程内探针每轮都
        ``TimeoutExpired``，正好会造出这种混口径曲线（还白烧几分钟墙钟）。

        默认 ``transcript_replay``：它对 8 个游戏 / 9 条角色轨都做过零点校准与灵敏度验收。
        ``in_process_first`` 保留给需要更高保真度、且确认该游戏探针可用的场合。
        测不出来就记 ``None`` + 原因——绝不用别的量顶替。
        """

        reason = ""
        if self.behavioral_ig_probe == "in_process_first":
            value, reason = self._inprocess_behavioral_ig(
                baseline_root=baseline_root, candidate_root=candidate_root, pairs=pairs
            )
            if value is not None:
                return value, reason, {"behavioral_ig_probe": "in_process_policy_probe"}
        if self.behavioral_ig_cases <= 0:
            note = "wire probe disabled (behavioral_ig_cases=0)"
            return None, f"{reason}; {note}" if reason else note, {}
        measurement = self._wire_behavioral_ig(
            request=request,
            iteration=iteration,
            baseline_id=baseline_id,
            baseline_root=baseline_root,
            candidate_id=candidate_id,
            candidate_root=candidate_root,
            pairs=pairs,
        )
        payload = measurement.payload()
        payload["behavioral_ig_probe"] = "transcript_replay"
        payload["behavioral_ig_reason"] = (
            f"{reason}; {measurement.reason}" if reason else measurement.reason
        )
        return measurement.value, str(payload["behavioral_ig_reason"]), payload

    def _measure_information_gain(
        self,
        *,
        request: MatchRequest,
        iteration: int,
        summary: Mapping[str, object],
        rows: Sequence[Mapping[str, object]],
        candidate_roots: Mapping[str, Path],
    ) -> dict[str, object]:
        """配对影子对局 + 三个层次的信息增益度量（见 :mod:`.info_gain` 模块说明）。"""

        empty: dict[str, object] = {
            "outcome_ig_nats": None,
            "behavioral_ig": None,
            "behavioral_ig_probe": None,
            "behavioral_action_disagreement": None,
            "behavioral_occupancy_shift": None,
            "behavioral_ig_decisions": 0,
            "behavior_divergence_frac": None,
            "behavior_identical": None,
            "paired_margin_shift": None,
            "paired_cases": 0,
            "baseline_candidate_id": None,
        }
        if not self.measure_information_gain:
            return {**empty, "behavioral_ig_reason": "information gain measurement disabled"}
        best = summary.get("best_candidate_id")
        if not isinstance(best, str) or not best:
            return {**empty, "behavioral_ig_reason": "no scored candidate this iteration"}
        baseline = self._previous_champion()
        if baseline is None:
            return {
                **empty,
                "behavioral_ig_reason": "no paired baseline yet (first scored iteration)",
            }
        baseline_id, baseline_root = baseline
        candidate_root = candidate_roots.get(best) or (self.root / "snapshots" / best)
        candidate_rows = [
            row
            for row in rows
            if row.get("candidate_id") == best and row.get("status") == "complete"
        ]
        if not candidate_rows:
            return {**empty, "behavioral_ig_reason": "best candidate has no complete match"}
        shadow_rows = self._run_shadow_matches(
            request=request,
            baseline_id=baseline_id,
            baseline_root=baseline_root,
            cases=candidate_rows,
        )
        key = lambda row: (  # noqa: E731 - 局部配对键
            str(row.get("opponent_id")),
            str(row.get("role")),
            int(row.get("seed") or 0),
        )
        shadow_by_case = {key(row): row for row in shadow_rows if row.get("status") == "complete"}
        pairs = [
            (shadow_by_case[key(row)], row) for row in candidate_rows if key(row) in shadow_by_case
        ]
        if not pairs:
            return {
                **empty,
                "baseline_candidate_id": baseline_id,
                "behavioral_ig_reason": "shadow matches produced no paired observation",
            }
        baseline_paired = [item[0] for item in pairs]
        candidate_paired = [item[1] for item in pairs]
        ig = outcome_ig_nats(
            outcome_counts(baseline_paired),
            outcome_counts(candidate_paired),
            epsilon=self.epsilon,
        )
        divergences: list[float] = []
        identical_flags: list[bool] = []
        for baseline_row, candidate_row in pairs:
            left = baseline_row.get("replay_path")
            right = candidate_row.get("replay_path")
            if not isinstance(left, str) or not isinstance(right, str):
                continue
            report = replay_divergence(left, right)
            if not report.get("available"):
                continue
            identical_flags.append(bool(report.get("identical")))
            value = report.get("divergence_frac")
            if isinstance(value, (int, float)):
                divergences.append(float(value))
        behavioral, reason, behavioral_payload = self._behavioral_ig(
            request=request,
            iteration=iteration,
            baseline_id=baseline_id,
            baseline_root=baseline_root,
            candidate_id=best,
            candidate_root=Path(candidate_root),
            pairs=pairs,
        )
        payload: dict[str, object] = {
            "outcome_ig_nats": round(ig, 6),
            **behavioral_payload,
            "behavioral_ig": None if behavioral is None else round(behavioral, 6),
            "behavioral_ig_reason": reason,
            "behavior_divergence_frac": (
                round(sum(divergences) / len(divergences), 6) if divergences else None
            ),
            "behavior_identical": (
                all(identical_flags) if identical_flags else None
            ),
            "paired_margin_shift": paired_margin_shift(baseline_paired, candidate_paired),
            "paired_cases": len(pairs),
            "baseline_candidate_id": baseline_id,
        }
        self._append(
            "InformationGainMeasured",
            {
                "request_id": request.request_id,
                "research_iteration": iteration,
                "candidate_id": best,
                "epsilon": self.epsilon,
                "baseline_outcomes": outcome_counts(baseline_paired),
                "candidate_outcomes": outcome_counts(candidate_paired),
                **payload,
            },
            f"goal-led-ig:{request.request_id}",
        )
        return payload

    def _emit_iteration_metrics(
        self,
        *,
        request: MatchRequest,
        iteration: int,
        summary: Mapping[str, object],
        rows: Sequence[Mapping[str, object]] = (),
        candidate_roots: Mapping[str, Path] | None = None,
        spread: Mapping[str, object] | None = None,
    ) -> None:
        previous = self._previous_win_rate()
        win_rate = summary.get("win_rate")
        outcome_shift = (
            abs(float(win_rate) - previous)
            if isinstance(win_rate, (int, float)) and previous is not None
            else None
        )
        gain = self._measure_information_gain(
            request=request,
            iteration=iteration,
            summary=summary,
            rows=rows,
            candidate_roots=candidate_roots or {},
        )
        tokens = self._token_total()
        pool_elo = self._pool_elo()
        payload: dict[str, object] = {
            "schema_version": "1.1",
            "research_iteration": iteration,
            "request_id": request.request_id,
            "game": self.game,
            "model": self.model,
            "harness": getattr(self.runtime, "harness", "codex"),
            "opponent_policy": self.policy_name,
            "opponent_ids": summary.get("opponent_ids"),
            "candidate_ids": list(request.candidate_ids),
            "rollout_k": self.rollout_k,
            # 本轮打了几个对手（b）。曲线要靠它解释胜率的分辨率：
            # b=1 时胜率只能取 {0, 0.5, 1}，b=4 时才有 0/0.25/…/1 九档。
            "batch": self.batch,
            "opponents_played": len(summary.get("opponent_ids") or []),
            # 逐对手战绩：k=1 × b 个对手之后，"对哪个对手赢了"比总胜率信息量大得多。
            # 慢评测之外的零成本 Elo 反解也用它（b 个不同强度的锚点一起拟合）。
            "win_rate_by_opponent": summary.get("win_rate_by_opponent"),
            # 探索多样性：pairwise 代码行差异（口径见 application/candidate_diversity.py）。
            # code_fingerprint 互异只能排除"逐字节相同"，排不掉"同一骨架换阈值"——
            # 后者让一轮 k 个候选退化成 1 个假设，是学习曲线走平最常见的原因。
            "candidate_spread": spread,
            "candidate_spread_verdict": (spread or {}).get("verdict"),
            "matches": summary.get("played"),
            "win_rate": win_rate,
            # 胜负平的**绝对计数**必须一起记：只有 win_rate 时，
            # 0.5 分不清是"2 胜 2 负"还是"4 平"，而这两种情况对
            # 下一轮该改什么的指示完全不同。
            "wins": summary.get("wins"),
            "draws": summary.get("draws"),
            "losses": summary.get("losses"),
            "infra_errors": summary.get("infra_errors"),
            "zero_round_losses": summary.get("zero_round_losses"),
            # 连续奖励：胜负是二值的，分差不是。全败的一轮里 win_rate 恒为 0，
            # 唯一能说明"哪个方向对"的就是分差与撑住的回合数
            # （见 docs/LESSONS_LEARNED.md G 条）。
            #
            # 这几个字段 _summarize() 一直在算，但从来没写进指标事件——
            # 于是逐轮表里 margin_mean 永远是 null，离线分析和出图都拿不到，
            # 等于那条修复只做了一半。
            "margin_mean": summary.get("margin_mean"),
            "margin_best": summary.get("margin_best"),
            "margin_by_candidate": summary.get("margin_by_candidate"),
            "best_candidate_win_rate": summary.get("best_candidate_win_rate"),
            "draw_rate": (
                summary.get("draws") / summary["played"]  # type: ignore[operator]
                if summary.get("played")
                else None
            ),
            "elo_vs_opponent": summary.get("elo_vs_opponent"),
            # 全池 Elo：用该 run 迄今全部对局做累积锚定 MLE（见 domain/pool_elo.py）。
            # 这是唯一跨轮、跨游戏可比的量——elo_vs_opponent 的锚点会随对手切换而跳，
            # 有序课程下"换更强对手"会画成假下降。
            # 仍保留 fixed_pool_elo 这个字段名给"真跑一遍全池"的慢通道；
            # 两者口径不同，所以分开记，不互相冒充。
            "pool_elo": (pool_elo or {}).get("elo"),
            "pool_elo_detail": pool_elo,
            "fixed_pool_elo": None,
            # 信息增益：配对影子对局给出结果分布 KL（nats/局，全游戏可得）；
            # 决策级行为 IG 走通用线协议探针（口径见 A 仓 decision_space.yaml 的
            # information_gain 段），测不出来就记 null + 原因，绝不用别的量顶替。
            "outcome_ig_nats": gain.get("outcome_ig_nats"),
            "behavioral_ig": gain.get("behavioral_ig"),
            "behavioral_ig_reason": gain.get("behavioral_ig_reason"),
            "behavioral_ig_probe": gain.get("behavioral_ig_probe"),
            # |A| 的口径必须跟着数一起走：读者要能看出 KL 用的支撑集是精确枚举
            # 还是"操作类型字母表"这个测量约定。
            "behavioral_ig_support_mode": gain.get("support_mode"),
            "behavioral_ig_support_cardinality": gain.get("support_cardinality"),
            # 精确覆盖率：|A(s)| 有多少个决策点是真枚举出来的、多少个退回了常量。
            # 只报 support_mode 不够——"mixed"这个标签本身不说明精确了多少。
            "behavioral_ig_support_exact_decisions": gain.get("support_exact_decisions"),
            "behavioral_ig_support_exact_fraction": gain.get("support_exact_fraction"),
            "behavioral_ig_support_notes": gain.get("support_alignment_notes"),
            # 随机流耦合口径：这个数是"公共随机流下的策略偏离"还是"严格确定性下的 KL"，
            # 必须跟着数一起出现，否则读者无从判断可比性。
            "behavioral_ig_coupling": gain.get("behavioral_ig_coupling"),
            "behavioral_ig_decisions": gain.get("behavioral_ig_decisions"),
            # 无测量假设的对照量：动作分歧率（KL 只是它在声明 |A| 下的单调重标度）。
            "behavioral_action_disagreement": gain.get("behavioral_action_disagreement"),
            # 状态占据位移：与 policy KL 分开报告，永不相加。
            "behavioral_occupancy_shift": gain.get("behavioral_occupancy_shift"),
            "behavior_divergence_frac": gain.get("behavior_divergence_frac"),
            "behavior_identical": gain.get("behavior_identical"),
            "paired_margin_shift": gain.get("paired_margin_shift"),
            "paired_cases": gain.get("paired_cases"),
            "baseline_candidate_id": gain.get("baseline_candidate_id"),
            "outcome_shift": outcome_shift,
            "best_candidate_id": summary.get("best_candidate_id"),
            "total_tokens": tokens,
            "token_events_sum": self._token_event_sum(),
            "total_wall_time_s": round(time.time() - self._started_at(), 3),
            # 实验 2 的第二个横坐标：agent 到目前为止真正能读到的完整轨迹数。
            "trajectories_seen": self._trajectories_seen(),
            # 实验 5：有序课程的征服进度（非有序策略为 null）。
            "conquest": (
                self._conquest_state().summary if self._conquest_state() is not None else None
            ),
            "ladder_size": len(self.public_leaderboard),
            "history_mode": self.history_mode,
            # 会话轮转：这一轮开始时 harness 的上下文占用，以及迄今换过几个 thread。
            # 记它是因为"agent 是否还记得上一轮"会影响读者对学习曲线的解释——
            # 轮转后经验只能来自 research/ 下的文件，而不是对话记忆。
            "thread_context_tokens": self._thread_context_tokens(),
            "thread_rotations": sum(
                1 for event in self.events.read_all() if event.event_type == "GoalSessionRotated"
            ),
        }
        self._append(
            "IterationMetricsFinalized", payload, f"goal-led-metrics:{request.request_id}"
        )

    # ---------------------------------------------------------------- budgets

    def budget_status(self) -> dict[str, object]:
        tokens = self._token_total()
        elapsed = time.time() - self._started_at()
        exhausted: str | None = None
        if self.token_budget is not None and tokens is not None and tokens >= self.token_budget:
            exhausted = "token_budget"
        elif self.wall_budget_s is not None and elapsed >= self.wall_budget_s:
            exhausted = "wall_budget"
        return {
            "tokens": tokens,
            "token_budget": self.token_budget,
            "elapsed_s": round(elapsed, 3),
            "wall_budget_s": self.wall_budget_s,
            "exhausted": exhausted,
        }

    # ------------------------------------------------------------- feedback

    def _pending_feedback(self) -> tuple[MatchRequest, Path, int] | None:
        delivered = {
            str(event.payload["request_id"])
            for event in self.events.read_all()
            if event.event_type == "GoalFeedbackDelivered"
            and isinstance(event.payload.get("request_id"), str)
        }
        for event in reversed(self.events.read_all()):
            if event.event_type != "GoalMatchRequested":
                continue
            request_id = event.payload.get("request_id")
            if not isinstance(request_id, str) or request_id in delivered:
                continue
            archived = self.workspace / ".agentbench" / "processed-requests" / f"{request_id}.json"
            feedback = self.workspace / "feedback" / request_id / "feedback.json"
            if not archived.is_file() or not feedback.is_file():
                continue
            request = MatchRequest.from_path(archived)
            match_count = sum(
                event.event_type == "GoalMatchCompleted"
                and event.payload.get("request_id") == request_id
                for event in self.events.read_all()
            )
            return request, feedback, match_count
        return None

    def _margin_note(self, summary: Mapping[str, object]) -> str:
        """逐候选分差段落。

        分差是连续量：胜负二值化之后信息就没了，但"差多少"一直在。全败的一轮
        尤其需要它——那时胜率恒为 0，唯一能说明"哪个方向对"的就是分差和
        撑住的回合数。实测 antwar2 对 rank1 连续 15 轮胜率为 0，如果只报胜率，
        那 15 轮对 agent 而言完全等价，等于白跑。
        """

        by_candidate = summary.get("margin_by_candidate")
        if not isinstance(by_candidate, Mapping) or not by_candidate:
            return ""
        ranked = sorted(
            by_candidate.items(),
            key=lambda item: -float(item[1].get("mean", 0.0)),  # type: ignore[union-attr]
        )
        parts = [
            "{name}: 平均分差 {mean:+g}（最好 {best:+g}，最差 {worst:+g}{rounds}）".format(
                name=name,
                mean=stats.get("mean"),  # type: ignore[union-attr]
                best=stats.get("best"),  # type: ignore[union-attr]
                worst=stats.get("worst"),  # type: ignore[union-attr]
                rounds=(
                    f"，均撑 {stats.get('rounds_mean')} 回合"  # type: ignore[union-attr]
                    if stats.get("rounds_mean") is not None  # type: ignore[union-attr]
                    else ""
                ),
            )
            for name, stats in ranked
        ]
        return (
            "\n\n**逐候选分差**（终局分差，正=你领先；胜负只是它过不过零，"
            "所以即使全败也要看这个数在往哪边走）：\n- " + "\n- ".join(parts) + "\n"
        )

    def _feedback_headline(self, summary: Mapping[str, object] | None) -> str:
        """下发给 agent 的成绩摘要。抽成独立方法是为了可单测。"""

        if summary is None:
            return ""
        if summary.get("protocol_error"):
            notes = "；".join(str(item) for item in (summary.get("protocol_notes") or []))
            return (
                "本轮**没有产生任何有效对局**，因为候选包不符合提交格式："
                f"{notes}。请先修好格式：每个候选的 main.py 放到 "
                ".agentbench/rollouts/<candidate_id>/（该目录叠加到工作区之上），"
                "候选之间必须有代码差异。修好后重新写 action.json。"
            )
        if summary.get("win_rate") is None:
            return ""
        headline = (
            f"本轮胜率 {float(summary['win_rate']):.2%}"  # type: ignore[arg-type]
            f"（{summary.get('wins')}胜/{summary.get('draws')}平/{summary.get('losses')}负）。"
        )
        return headline + self._margin_note(summary)

    def _deliver_feedback(
        self,
        session: AgentSession,
        request: MatchRequest,
        feedback: Path,
        match_count: int,
        request_count: int,
        summary: Mapping[str, object] | None = None,
        prompt_next: bool = True,
        spread: Mapping[str, object] | None = None,
        handoff: str = "",
    ) -> GoalLedOutcome:
        iteration = request_count + 1
        headline = self._feedback_headline(summary)
        if summary is not None and summary.get("win_rate") is not None:
            if summary.get("protocol_notes"):
                headline += (
                    "注意：部分候选被跳过 —— "
                    + "；".join(str(item) for item in summary["protocol_notes"])  # type: ignore[index]
                    + "。"
                )
            # 0 回合判负 = 你的程序在第一帧就被判死（超时/非法/崩溃）。
            # 这时没有回放可读，对战器诊断是唯一线索，必须直接说出来。
            zero_rounds = int(summary.get("zero_round_losses") or 0)
            diagnostics = summary.get("diagnostics") or []
            if zero_rounds and isinstance(diagnostics, list) and diagnostics:
                details = "；".join(
                    f"{item.get('detail')}（{item.get('count')} 局）"
                    for item in diagnostics[:3]
                    if isinstance(item, Mapping)
                )
                headline += (
                    f"其中 {zero_rounds} 局是**第 0 回合就判负**（没有回放可读），"
                    f"对战器诊断：{details}。"
                    "请先按诊断把协议/启动问题修好，再谈策略强度。"
                )
            elif isinstance(diagnostics, list) and diagnostics:
                details = "；".join(
                    f"{item.get('detail')}（{item.get('count')} 局）"
                    for item in diagnostics[:3]
                    if isinstance(item, Mapping)
                )
                headline += f"对战器诊断：{details}。"
        experience_hint = (
            f"更新 research/{EXPERIENCE_FILE} 中的成功经验与失败假设，"
            if self.experience_skills
            else ""
        )
        if prompt_next:
            # 这一次 turn 是全轮里最贵的（读 k 份回放 + 写 k 个候选），也是最容易把
            # 上下文推过压缩阈值的地方，所以进它之前必须再查一次轮转。
            if not handoff:
                session, handoff = self._rotate_if_needed(session, iteration=iteration)
            # 伪多样性是"这一轮白花了 k 倍对局"级别的浪费，所以它必须出现在
            # 下一轮提示词的**开头**，而不是埋在末尾的通用要求里。
            spread_note = diversity_note(spread)
            session = self._turn(
                session,
                (
                    handoff
                    + (f"{spread_note}\n" if spread_note else "")
                    + f"官方比赛反馈已写入 {feedback.relative_to(self.workspace)}。{headline}"
                    "**先读同目录下的 all-replays.md** —— 那是本轮每一局回放被翻译成的"
                    "自然语言复盘（判决、对手画像、你的浪费、逐回合时间线）合并成的**一份**文件，"
                    "一次读完即可，不要逐局去读单局的 replay.md（那样每多读一个文件"
                    "就多花一次模型往返；单局文件只在你要核对某个具体数字时才用）。"
                    "不需要自己写代码去解析 replay.json。汇总数字看 feedback.json。"
                    f"{experience_hint}修改策略，并在准备好下一步时写新的 action.json。"
                    f"常规探索请从共同 parent 产出 k={self.rollout_k} 个候选，"
                    "**每个候选一个不同的优化假设**（不同取胜路径/不同机制，"
                    f"任意两个之间代码差异 ≥ {DIVERSITY_THRESHOLD_LINES} 行），"
                    "各打一份诊断回放；不要把单一反事实或单点动作修改当作前置要求。"
                    "候选改多少由证据决定（从机制与回放里读出该怎么改），"
                    "但请把旧版本文件留着，以便回退与按局面组合；"
                    "提交前跑 selfcheck.py 确认每个候选都是合法代码。"
                ),
                iteration=iteration,
                cleared=self._cleared_count(),
            )
            self._append(
                "GoalFeedbackDelivered",
                {"request_id": request.request_id, "feedback_path": str(feedback)},
                f"goal-led-feedback:{request.request_id}",
            )
        else:
            # 末轮：反馈文件已经落盘（产物完整、可复盘），但不再驱动 agent 思考——
            # 那一轮的产出必然被丢弃，纯烧墙钟与 token。
            self._append(
                "GoalFeedbackWithheld",
                {
                    "request_id": request.request_id,
                    "feedback_path": str(feedback),
                    "reason": "final iteration; not prompting for work that would be discarded",
                },
                f"goal-led-feedback-withheld:{request.request_id}",
            )
        self._write_state(session.thread_id, request_count + 1)
        return GoalLedOutcome(
            thread_id=session.thread_id,
            workspace=self.workspace,
            request_id=request.request_id,
            match_count=match_count,
            iteration=iteration,
            win_rate=(
                float(summary["win_rate"])  # type: ignore[arg-type]
                if summary is not None and summary.get("win_rate") is not None
                else None
            ),
        )

    def _run_request(
        self, session: AgentSession, request_count: int, *, prompt_next: bool = True
    ) -> GoalLedOutcome:
        cleared = self._cleared_count()
        request = self._consume_request()
        if request_count == 0 and self.policy_name == SELF_DECIDE:
            top = self._ladder()[0].opponent_id
            if request.opponent_id != top:
                raise ValueError(
                    f"the first Goal-led official request must target the top-ranked opponent {top}"
                )
        feedback, match_count, summary, rows, candidate_roots = self._execute(
            request, iteration=request_count + 1, cleared=cleared
        )
        # 探索多样性：本轮 k 个候选到底是 k 个尝试还是一个尝试的 k 份复制。
        # 同一份度量走两条路——记进指标（可画曲线）+ 反馈给 agent（闭环纠正）。
        spread = candidate_spread(candidate_roots)
        self._emit_iteration_metrics(
            request=request,
            iteration=request_count + 1,
            summary=summary,
            rows=rows,
            candidate_roots=candidate_roots,
            spread=spread,
        )
        return self._deliver_feedback(
            session,
            request,
            feedback,
            match_count,
            request_count,
            summary,
            prompt_next=prompt_next,
            spread=spread,
        )

    # -------------------------------------------------------------- lifecycle

    def _harness_identity(self) -> dict[str, object]:
        """harness 二进制身份（版本换了，曲线就不可直接横比）。"""

        identity: dict[str, object] = {}
        capabilities = getattr(self.runtime, "_capabilities", None) or {}
        identity.update(
            {
                key: value
                for key, value in capabilities.items()
                if isinstance(value, (str, int, float, bool))
            }
        )
        # cc 侧 binary 是字符串；codex 侧是 command 序列（codex app-server …）。
        binary = getattr(self.runtime, "binary", None)
        command = getattr(self.runtime, "command", None)
        if isinstance(binary, str):
            identity.setdefault("binary", binary)
        elif isinstance(command, (list, tuple)) and command:
            identity.setdefault("binary", str(command[0]))
            identity.setdefault("command", [str(item) for item in command])
        if "version" not in identity and identity.get("binary"):
            try:
                probe = subprocess.run(
                    (str(identity["binary"]), "--version"),
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                line = (probe.stdout or probe.stderr or "").strip().splitlines()
                if line:
                    identity["version"] = line[0][:120]
            except (OSError, subprocess.SubprocessError):
                identity["version"] = None
        return identity

    def _append_reproducibility_manifest(self, session: AgentSession) -> None:
        """落盘"这一次 run 的结果由哪些输入决定"。

        LLM 采样本身不可逐 token 复现（中转站不暴露 seed/温度），所以我们能保证的是
        **统计可复现**：除模型采样外的一切输入都被冻结并留下摘要，任何两次 run 的
        差异都可以归因到"采样噪声"而不是"配置漂移"。这里记录：

        - prompt 三段（objective / base_instructions / developer_instructions）的摘要；
        - 冻结实验配置与 GamePack 摘要（run-manifest.json 已由 factory 写好）；
        - harness 身份（版本/二进制）、模型、隔离描述；
        - 框架级实验变量：seeds / roles / K / 对手策略 / 并行度 / ε / 迭代模式。
        """

        context = self._context("")

        def digest(value: str | None) -> str | None:
            if not value:
                return None
            return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

        manifest = self.root / "run-manifest.json"
        gamepack_digests: object = None
        if manifest.is_file():
            try:
                gamepack_digests = json.loads(manifest.read_text(encoding="utf-8")).get(
                    "gamepack_digests"
                )
            except json.JSONDecodeError:
                gamepack_digests = None
        self._append(
            "RunReproducibilityManifest",
            {
                "schema_version": "1.0",
                "thread_id": session.thread_id,
                "game": self.game,
                "model": self.model,
                "model_provider": self.model_provider,
                "harness": getattr(self.runtime, "harness", "codex"),
                "harness_identity": self._harness_identity(),
                "prompt_digests": {
                    "objective": digest(context.objective),
                    "base_instructions": digest(context.base_instructions),
                    "developer_instructions": digest(context.developer_instructions),
                    "prompt_override": digest(self.prompt_override),
                },
                "gamepack_digests": gamepack_digests,
                "experiment_variables": {
                    "seeds": list(self.seeds),
                    # 候选**实际坐的**座次。分轨游戏只有一个（换座次就是同轨互殴），
                    # 所以这里不等于 game.yaml 里的角色列表。
                    "roles": list(self.roles),
                    "rollout_k": self.rollout_k,
                    # b 以前**没记进可复现清单**，而"一轮 = k × b × 座次"要靠它。
                    # 这里是策略实际会打的个数（单目标策略恒为 1），不是配置里的值。
                    "batch": self.batch,
                    "opponent_policy": self.policy_name,
                    "match_parallelism": self.match_parallelism,
                    "code_constraint": self.code_constraint,
                    "experience_skills": self.experience_skills,
                    "rival_code_visible": self.rival_code_visible,
                    "epsilon": self.epsilon,
                    "information_gain": self.measure_information_gain,
                    "iteration_mode": self.iteration_mode,
                    "token_budget": self.token_budget,
                    "wall_budget_s": self.wall_budget_s,
                },
                "runnable_opponents": len(self.runnable_opponent_ids),
                "non_reproducible": [
                    "LLM sampling (no seed/temperature control on the relay endpoint)",
                    "wall-clock dependent match timeouts (mitigated by CPU leases)",
                ],
            },
            "goal-led-repro-manifest",
        )

    def start(self, *, prompt_next: bool = True) -> GoalLedOutcome:
        """起第一轮。``prompt_next=False`` 用于 ``iterations=1``：跑完就收，不白想一轮。"""

        if self._state_path.exists():
            raise ValueError("goal-led run already started")
        if not self.bootstrap_root.is_dir():
            raise ValueError("bootstrap root is unavailable")
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.bootstrap_root, self.workspace)
        research = self.workspace / "research"
        research.mkdir(exist_ok=True)
        if self.experience_skills and not (research / EXPERIENCE_FILE).exists():
            (research / EXPERIENCE_FILE).write_text(
                "# 迭代经验\n\n> 每轮追加：假设 / 证据（回放片段）/ 结论 / 下一步。\n",
                encoding="utf-8",
            )
        (self.workspace / "leaderboard.json").write_text(
            json.dumps(
                {"schema_version": "1.0", "opponents": list(self.public_leaderboard)},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        # 容器边界：装配完成、agent 还没开始思考之前，把工作区真的扫一遍。
        # 不通过就不开跑 —— 一次长跑几十小时，跑完才发现口径被污染的代价太大。
        assert_sealed(self.workspace)
        self._append(
            "ContainerBoundaryVerified",
            {
                "workspace": str(self.workspace),
                "violations": [],
                "checked": "local_match_runners, training_scripts, ready_made_strategies",
            },
            "goal-led-container-sealed",
        )
        session = self.runtime.start(self._context(""))
        self._append_reproducibility_manifest(session)
        self._append(
            "GoalLedStarted",
            {
                "thread_id": session.thread_id,
                "workspace": str(self.workspace),
                "opponent_policy": self.policy_name,
                "rollout_k": self.rollout_k,
                "seeds": list(self.seeds),
                "roles": list(self.roles),
                "match_parallelism": self.match_parallelism,
            },
            "goal-led-started",
        )
        # Persist the single thread before the first long Goal turn.  A
        # client-side timeout must never orphan research that is still stored
        # by the run-local App Server.
        self._write_state(session.thread_id, 0)
        first_opponent = self.policy.select(
            iteration=0, batch=self.batch, history=self._opponent_history()
        )
        target_hint = (
            f"本轮对手已由框架指定：{'、'.join(first_opponent)}。"
            if first_opponent
            else f"第一个 Action 必须挑战榜首 {self._ladder()[0].opponent_id}。"
        )
        rollout_hint = (
            (
                "本轮只交 **1 个候选**（k=1）。第 0 轮没有回放，所以这一版是你对"
                "这个游戏的**初始理解**：请从 rules.md 的机制出发，选定一条你认为"
                "站得住的开局哲学（例如先经济后成型 / 全程压制 / 围绕某条特定机制），"
                "把它写清楚、写对格式。"
                f"这一版会被拿去打 {self.batch} 个对手，"
                f"评测器回传的 {self.batch} 份回放就是你下一轮的全部信息来源。"
            )
            if self.rollout_k == 1
            else (
                f"本轮要交 k={self.rollout_k} 个候选，它们必须是 {self.rollout_k} 个"
                "**不同的开局哲学**（例如：先经济后成型 / 全程压制 / 围绕某条特定机制做文章），"
                f"而不是同一份 v000 换 {self.rollout_k} 个阈值——"
                "第 0 轮没有回放，唯一能获得信息的方式就是让这几个候选**互相远离**，"
                "这样评测器回传的 k 份回放才能告诉你哪条路线在这个游戏里站得住。"
                f"任意两个候选之间的代码差异不少于 {DIVERSITY_THRESHOLD_LINES} 行，"
                "框架会度量并在反馈里告诉你结果。"
            )
        )
        version_hint = (
            "每个候选的策略类请起**带版本号的名字**（如 V000EconomyAgent）并留在文件里："
            "以后每轮都会在已有版本上继续演进，旧版本留着才能回退与按局面组合，"
            "所以这一轮的类名就是整条演进链的根。\n"
            if self.history_mode == "full"
            else "策略请全部写在**一个文件**里（本轮消融只让你看到上一版，"
            "所以不要建多版本文件、也不要 import 历史版本）。\n"
        )
        session = self._turn(
            session,
            (
                "第 0 轮：你手上只有规则，没有任何回放。任务是**从规则出发**写出 v000。\n"
                "先读 gamepack/rules.md（规则）、gamepack/decision_space.yaml（你能做哪些动作）、"
                "CANDIDATE_CONTRACT.md（提交格式）、leaderboard.json（你要打的人是谁）。\n"
                "ai_example.py 只演示格式、强度为零；另存为 ai.py 后写你自己的策略，"
                "**不要**把它当成起点抄。\n"
                + rollout_hint
                + "\n"
                + version_hint
                + "写完后跑 selfcheck.py（容器内自带）确认每个候选都能导入、接口正确、"
                "能走完一帧；**不通过就不要提交** —— 格式错会让整轮反馈只告诉你「你崩了」。\n"
                "本轮的第一优先级是协议格式绝对正确（格式错=0 回合判负，"
                "整轮反馈只会告诉你「你崩了」，学不到任何策略信息），其次才是强度。\n"
                f"完成后在 .agentbench/action.json 写第一个官方 Action；{target_hint}"
                "不要等待框架替你选择对手。\n"
                "提醒：容器里没有对手、没有游戏后端、没有本地对战工具，"
                "所以本轮无法（也不需要）自我评测——写完就交，强度由评测器告诉你。"
            ),
            iteration=0,
            cleared=0,
        )
        return self._run_request(session, 0, prompt_next=prompt_next)

    def request_correction(self, problem: str) -> None:
        """把协议/格式问题交回 Goal 让它自我纠正（不消耗一轮迭代）。

        典型场景：没写 action.json、字段非法、候选目录里没有 main.py。研究上这属于
        "harness 与 agent 的协议对齐"，不是策略强度问题，因此不能算作一次迭代，
        更不该让整个 run 崩掉。
        """

        thread_id, _ = self._load_state()
        session = self.runtime.resume(thread_id, self._context(""))
        self._turn(
            session,
            (
                f"你上一步的提交无法执行：{problem}\n"
                "请修正后重新提交一份新的 .agentbench/action.json。要点：\n"
                "1) 必填字段：action_id（本 run 内唯一）、rollouts[{candidate_id}]、"
                "selected_rival（必须是 leaderboard.json 里的可运行 id）、"
                "roles、seeds、rationale；\n"
                "2) 每个候选的 main.py 必须放在 .agentbench/rollouts/<candidate_id>/ 下"
                "（该目录会叠加到工作区之上），候选之间要有代码差异；\n"
                "3) 不要复用已经用过的 action_id。"
            ),
        )
        self._append(
            "GoalCorrectionDelivered",
            {"problem": problem},
            f"goal-led-correction-delivered:{hashlib.sha256(problem.encode()).hexdigest()[:12]}",
        )

    def _apply_history_mode(self, *, iteration: int) -> None:
        """实验 4 的消融执行点：把"不该被看见的历史"从工作区里真正拿掉。

        只靠提示词说"别看历史"是无效的——历史就摆在工作区里。三档语义：

        * ``full``       : 什么都不动（默认，能看到所有历史迭代）。
        * ``no_notes``   : 删掉经验文档（历史代码与会话保留）。
        * ``last_only``  : 再删掉历史候选目录、**历次 action 原文**、以及除最近一份
          以外的反馈；配合 :meth:`_session_for_iteration` 每轮开新会话，
          agent 只剩"上一版代码 + 上一轮反馈"。提示词里还会加**单文件约束**
          （见 ``_developer_instructions``）：不许 import 历史版本，否则
          "只能看到上一版"会被 import 链绕过。

        当前策略文件（工作区根目录的 main.py 等）**永远保留**：消融的是"历史"，
        不是"起点"，否则每轮都从零重写，测的就不是同一件事了。

        ``last_only`` 下**必须**删 ``.agentbench/processed-requests/``：那里存着历次
        action 的原文（含 agent 自己写的 rationale 与选择的对手），是一份完整的
        决策日记。漏掉它的话这一档名义上"只看上一版"、实际上 agent 把上一轮的推理
        原样读回去，两组的差别就被抹平了——消融失效且**看不出来**。

        保留"最近一份反馈"是刻意的：不给任何对局结果，agent 连自己上一轮赢没赢
        都不知道，那消融掉的是**学习信号**而不是历史记忆，实验就换了题目。
        """

        if self.history_mode == "full":
            return
        removed: list[str] = []
        experience = self.workspace / "research" / EXPERIENCE_FILE
        if experience.is_file():
            experience.unlink()
            removed.append(f"research/{EXPERIENCE_FILE}")
        if self.history_mode == "last_only":
            for relative in (".agentbench/rollouts", ".agentbench/processed-requests"):
                target = self.workspace / relative
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                    removed.append(relative)
            feedback_root = self.workspace / "feedback"
            if feedback_root.is_dir():
                directories = sorted(
                    (item for item in feedback_root.iterdir() if item.is_dir()),
                    key=lambda item: item.stat().st_mtime,
                )
                for item in directories[:-1]:  # 只留最近一份
                    shutil.rmtree(item, ignore_errors=True)
                    removed.append(f"feedback/{item.name}")
        if removed:
            self._append(
                "HistoryAblationApplied",
                {
                    "iteration": iteration,
                    "history_mode": self.history_mode,
                    "removed": removed,
                },
                f"goal-led-history-ablation:{iteration}",
            )

    def _thread_context_tokens(self) -> int | None:
        """**当前 thread** 的上下文占用（不是全 run 累计）。

        harness 报的 ``input_tokens`` 就是那一次请求的上下文大小，单调增长；
        换 thread 后会归零，所以这里从最后一次"会话重置/轮转/开跑"事件之后开始数。
        """

        latest: int | None = None
        for event in self.events.read_all():
            if event.event_type in ("GoalLedStarted", "GoalSessionReset", "GoalSessionRotated"):
                latest = None
                continue
            if event.event_type != "AgentTokenUsage":
                continue
            payload = event.payload if isinstance(event.payload, Mapping) else {}
            value = payload.get("input_tokens")
            if isinstance(value, int) and (latest is None or value > latest):
                latest = value
        return latest

    def _rotate_if_needed(
        self, session: AgentSession, *, iteration: int, force: bool = False
    ) -> tuple[AgentSession, str]:
        """上下文接近 codex 压缩阈值时换新 thread，返回 (会话, 接力说明)。

        为什么不让 codex 自己压缩：它的压缩必定失败（见 config 里的详注），
        失败会把整个 turn 打成 failed 并终止 run。所以框架宁可主动断掉对话记忆，
        也不能让上下文走到那一步——**跑完一轮永远比"记得上一轮说过的话"重要**。

        ``force``（由 ``thread_rotate_each_iteration`` 驱动）是必需的：阈值只能在
        turn **之前**检查，而单个 turn 内部就能把上下文从 90k 推过压缩线
        （实测 antwar2 死在 97k，当时阈值是 110k，根本没轮到检查）。每轮从零开始，
        单轮增量才是上下文的上界。

        接力说明不是可选的礼貌用语：新 thread 里 agent 没有任何对话记忆，
        必须明确告诉它"你的历史都在文件里、先读你自己写的经验"，
        否则它会当成第一次见到这个游戏，从零重写策略。
        """

        threshold = self.thread_rotate_context_tokens
        used = self._thread_context_tokens()
        if not force:
            if threshold is None:
                return session, ""
            if used is None or used < threshold:
                return session, ""
        fresh = self.runtime.start(self._context(""))
        self._append(
            "GoalSessionRotated",
            {
                "iteration": iteration,
                "previous_thread_id": session.thread_id,
                "thread_id": fresh.thread_id,
                "context_tokens": used,
                "threshold_tokens": threshold,
                "trigger": "each_iteration" if force else "context_threshold",
                "reason": "avoid codex remote compaction (known-fatal with this model)",
                "history_mode": self.history_mode,
            },
            f"goal-led-session-rotated:{iteration}:{session.thread_id}",
        )
        self._write_state(fresh.thread_id, iteration - 1)
        return fresh, self._handoff_prompt(used)

    def _handoff_prompt(self, used_tokens: int | None) -> str:
        """换 thread 后的接力说明（自包含，且明确指出工作区里什么都没丢）。"""

        experience = (
            f"② research/{EXPERIENCE_FILE} 是你自己历轮写的经验——**先读它**，"
            "尤其是你上一轮写下的「下一步该验证什么」；"
            if self.experience_skills
            else ""
        )
        size = f"上一个对话线程的上下文到了 {used_tokens // 1000}k，" if used_tokens else ""
        return (
            f"【会话已接力】{size}框架给你换了一个干净的线程。"
            "你**没有**之前的对话记忆，但工作区一样东西都没丢：\n"
            "① 工作区根目录是当前这一版策略代码，历史候选与历轮 action 都还在；\n"
            f"{experience}"
            "③ feedback/ 下是历轮官方对局反馈（每轮一份 all-replays.md 自然语言复盘"
            "+ feedback.json 汇总数字）。\n"
            "请把这些当作你的记忆来读，不要从零重写策略、也不要重复已经被证伪过的假设。\n"
        )

    def _session_for_iteration(self, thread_id: str, *, iteration: int) -> AgentSession:
        """取得本轮要用的会话。

        ``last_only`` 每轮开一个**全新**会话（没有任何对话历史），并记事件；
        其它模式沿用常驻会话（``resume``），保留跨轮上下文。
        """

        if self.history_mode != "last_only":
            return self.runtime.resume(thread_id, self._context(""))
        session = self.runtime.start(self._context(""))
        self._append(
            "GoalSessionReset",
            {
                "iteration": iteration,
                "history_mode": self.history_mode,
                "previous_thread_id": thread_id,
                "thread_id": session.thread_id,
            },
            f"goal-led-session-reset:{iteration}",
        )
        return session

    def advance(self, *, prompt_next: bool = True) -> GoalLedOutcome:
        """推进一轮。

        ``prompt_next=False`` 表示"这是本 run 的最后一轮"：照常跑对局、出指标、
        落反馈文件，但**不再让 agent 去想下一轮**。

        为什么需要这个开关：``_deliver_feedback`` 里的 ``_turn`` 是**同步**的——它会一直等到
        agent 写出下一份 ``action.json`` 才返回。所以每轮 ``advance()`` 返回时，agent 其实
        已经把下一轮想完了。轮数用尽时 driver 直接退出循环，那次思考的产出被整份丢弃。
        实测这一次白干烧掉 **1369s 墙钟 + 66k input tokens**（agent 思考占全程 84%，
        这是里面最容易省的一块）。
        """

        thread_id, request_count = self._load_state()
        self._apply_history_mode(iteration=request_count + 1)
        session = self._session_for_iteration(thread_id, iteration=request_count + 1)
        # 轮转必须发生在**任何 turn 之前**：turn 一旦开始，上下文越过 codex 的压缩
        # 阈值就没救了（压缩失败 = 整轮 failed）。而阈值检查挡不住"单个 turn 内部"
        # 的增长，所以 thread_rotate_each_iteration 下每轮无条件换。
        session, handoff = self._rotate_if_needed(
            session,
            iteration=request_count + 1,
            force=self.thread_rotate_each_iteration and request_count >= 1,
        )
        pending = self._pending_feedback()
        if pending is not None:
            request, feedback, match_count = pending
            summary_payload = None
            try:
                summary_payload = json.loads(feedback.read_text(encoding="utf-8")).get("summary")
            except (OSError, json.JSONDecodeError):
                summary_payload = None
            return self._deliver_feedback(
                session,
                request,
                feedback,
                match_count,
                request_count,
                summary_payload,
                prompt_next=prompt_next,
                handoff=handoff,
            )
        if self._request_path() is None:
            session = self._turn(
                session,
                handoff + self._continue_prompt(request_count + 1),
                iteration=request_count + 1,
                cleared=self._cleared_count(),
            )
        return self._run_request(session, request_count, prompt_next=prompt_next)

    def _continue_prompt(self, iteration: int) -> str:
        """推进下一轮的提示词。

        ``last_only`` 下会话是全新的，提示词必须**自包含**：明确告诉它工作区里
        已有一份当前策略、最近一份官方反馈在哪，以及它看不到更早的历史。
        """

        if self.history_mode != "last_only":
            return "继续研究：基于已有 Experience 与公开反馈改进策略，然后写下一份 action.json。"
        return (
            "这是一次**只看上一版**的迭代：你没有之前的对话历史、没有经验笔记、"
            "也看不到更早的历史候选代码。"
            "工作区根目录里是**上一版**策略；feedback/ 下只保留最近一份官方对局反馈"
            "（含 replay 与 feedback.json）。请只依据这些材料改进策略，"
            "所有代码继续写在同一个文件里（不要建新版本文件、不要 import 历史版本），"
            "然后写一份新的 .agentbench/action.json。"
        )

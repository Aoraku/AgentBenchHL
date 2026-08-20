"""决策级行为信息增益（behavioral IG）—— 通用测量流程。

它回答的问题
------------

"这一轮迭代真的改变了策略的**决策**吗？改了多少？"——单位是 nats/决策，定义是
``docs/metrics-schema.md`` 的 ``local_policy_kl_trace`` 在**参考占据分布**上的均值：

    ``behavioral_ig = E_{z ~ d_parent} [ KL( π_parent(·|z) ‖ π_candidate(·|z) ) ]``

流程（每个配对 case 做一遍）
----------------------------

1. **录参考轨迹**：把父版本快照克隆成"会录音"的版本（:mod:`..adapters.transcript.shim`），
   在同一 ``(对手, 座次, seed)`` 上真跑一局，录下判题器→选手的完整入站字节流，
   以及父版本每一步写回的动作帧。
2. **录候选自己的轨迹**：同样录一局候选自己的对局，只用于算 occupancy 位移
   （状态访问分布的变化，**单独报告，永不与 KL 相加**）。
3. **确定性自校验**：把参考入站流重新喂给父版本快照，要求逐帧复现步骤 1 的动作。
   复现不了 ⇒ 该策略不是"观测流 → 动作"的确定性函数（用了时钟/随机/读写时序），
   此时 KL 的前提不成立，**记 null + 原因**，绝不给一个凑出来的数。
4. **冻结上下文上重放候选**：把同一条参考入站流喂给候选，取它的动作序列。因为喂的
   是同一条流，候选的内部记忆也沿参考轨迹演化，两边第 i 个决策处在同一个上下文上。
5. **算 KL**：|A| 取自 ``decision_space.yaml`` 的 ``information_gain.support``
   （非对称游戏按座次取），ε 取自 ``measurement.epsilon``。

诚实性纪律
----------

* 任何一步不成立就整体记 null 并写明原因，绝不用 outcome IG / 胜率差等别的量顶替；
* ``support_mode``（精确枚举 or 操作类型字母表约定）随每个数一起上报；
* 候选提前崩溃导致的缺失决策**不算作"与父版本一致"**，而是截断并把截断量报出来。
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from agentbench_hl.adapters.transcript import (
    build_recording_snapshot,
    read_transcript,
    replay_actions,
    snapshot_root,
    transcript_root,
)
from agentbench_hl.adapters.transcript.coupling import (
    COUPLING_COMMON_RANDOM,
    normalize_coupling,
)
from agentbench_hl.domain.wire_policy import (
    WireEpisode,
    compare_wire_policies,
    first_divergence,
)

#: 逐决策点真实 |A(s)| 的提供者：``(候选目录, 参考局回放, 座次) -> 每个决策点的支持集大小``。
#:
#: 为什么做成注入点而不是写死：只有实现了状态探针的游戏才拿得到精确支持集
#: （antwar 用 ``adapters/antwar/policy_trace_worker``，antwar2 用
#: ``adapters/antwar2/policy_probe``）。没有探针的游戏就该老老实实回落到
#: 字母表常量，并把 ``support_mode`` 如实上报，而不是让这一层假装自己什么都能算。
SupportProvider = Callable[[Path, Path, str], Sequence[int]]

#: 状态探针与线协议的决策点数允许的绝对缺口。
#:
#: 两者的计数口径天然差一点：探针按回放记录数走，线协议按选手实际回复的帧数走，
#: 尾帧（终局通知）常常只出现在后者。实测 antwar 是 246 vs 247。
#: 缺口在这个容差内就用探针给出的精确值、尾部按位置退回字母表；
#: 超过它才认为"决策点定义没对上"并整体回落——那种情况下把错位的 |A(s)| 套上去，
#: 会得到一个看似精确的错数，比诚实回落更糟。
SUPPORT_ALIGN_ABS_TOLERANCE = 2


@dataclass(frozen=True)
class BehavioralIgCase:
    """一个被测量的配对 case（父/子在完全相同的条件下各跑一局）。"""

    opponent_id: str
    role: str
    seed: int

    @property
    def tag(self) -> str:
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in self.opponent_id)
        return f"{safe}-{self.role}-seed-{self.seed}"

    def as_dict(self) -> dict[str, object]:
        return {"opponent_id": self.opponent_id, "role": self.role, "seed": self.seed}


@dataclass(frozen=True)
class CaseMeasurement:
    """单个 case 的测量结果（失败也是结果，reason 必填）。"""

    case: BehavioralIgCase
    mean_kl_nats: float | None = None
    disagreement_rate: float | None = None
    occupancy_shift: float | None = None
    compared_decisions: int = 0
    reference_decisions: int = 0
    candidate_decisions: int = 0
    first_divergence: int | None = None
    reason: str | None = None
    kl_trace: tuple[float, ...] = ()
    #: 这一 case 的 |A(s)| 是怎么来的：``exact_enumeration``（状态探针逐点枚举）
    #: 还是 ``opcode_alphabet``（操作类型字母表约定的常量）。
    #: 必须随数一起上报——KL 的尺度由 |A| 定，读者不知道口径就无法比较两个数。
    support_mode: str = "opcode_alphabet"
    #: 这一 case 里有多少个决策点用上了**精确** |A(s)|（其余回落到字母表常量）。
    #: 报它是因为"精确"可能只覆盖一部分决策点，而 KL 的尺度逐点由 |A| 决定。
    support_exact_decisions: int = 0
    #: 支撑集口径的来龙去脉（对齐差多少、为什么回落）。缺了它，"这个数是精确的吗"
    #: 就只能靠读代码猜——我们已经为此误判过一次。
    support_note: str | None = None

    @property
    def usable(self) -> bool:
        return self.mean_kl_nats is not None and self.compared_decisions > 0

    def as_dict(self) -> dict[str, object]:
        return {
            **self.case.as_dict(),
            "mean_kl_nats": self.mean_kl_nats,
            "disagreement_rate": self.disagreement_rate,
            "occupancy_shift": self.occupancy_shift,
            "compared_decisions": self.compared_decisions,
            "reference_decisions": self.reference_decisions,
            "candidate_decisions": self.candidate_decisions,
            "first_divergence": self.first_divergence,
            "reason": self.reason,
            "support_mode": self.support_mode,
            "support_exact_decisions": self.support_exact_decisions,
            "support_note": self.support_note,
        }


@dataclass(frozen=True)
class BehavioralIgMeasurement:
    """跨 case 汇总后的行为信息增益。"""

    value: float | None
    reason: str
    disagreement_rate: float | None = None
    occupancy_shift: float | None = None
    compared_decisions: int = 0
    support: Mapping[str, object] = field(default_factory=dict)
    cases: tuple[CaseMeasurement, ...] = ()
    #: 随机流耦合口径。``common_random_seed`` 表示两个版本在同一条随机流下比较
    #: （标准的公共随机数方差缩减），这是一个**测量约定**，必须随数一起出现。
    coupling: str = COUPLING_COMMON_RANDOM

    def payload(self) -> dict[str, object]:
        """写进事件的扁平字段。

        ⚠️ ``support`` 是 ``decision_space.yaml`` 里的**静态声明**（回落口径与 |A| 常量），
        而每个 case 实际用的口径在 ``CaseMeasurement.support_mode`` 里。早先这里直接
        ``**self.support`` 展开，静态声明就把真实口径覆盖掉了：探针明明算出了逐点
        精确的 |A(s)|，事件里却永远显示 ``opcode_alphabet``，我们据此误判了 14 轮。
        所以真实口径必须**后写**，盖在静态声明之上。
        """

        return {
            "behavioral_ig": None if self.value is None else round(self.value, 6),
            "behavioral_ig_reason": self.reason,
            "behavioral_ig_coupling": self.coupling,
            "behavioral_action_disagreement": (
                None if self.disagreement_rate is None else round(self.disagreement_rate, 6)
            ),
            "behavioral_occupancy_shift": (
                None if self.occupancy_shift is None else round(self.occupancy_shift, 6)
            ),
            "behavioral_ig_decisions": self.compared_decisions,
            "behavioral_ig_cases": len(self.cases),
            **{str(key): value for key, value in self.support.items()},
            **self.observed_support(),
        }

    def observed_support(self) -> dict[str, object]:
        """各 case 实际用到的支撑集口径（聚合后）。

        ``exact_enumeration``：所有被采纳的 case 都用上了逐点精确 |A(s)|；
        ``mixed``：部分 case 精确、部分回落；``opcode_alphabet``：全部回落。
        另外报出精确覆盖了多少个决策点，以及每个 case 的对齐说明——
        "这个数是不是精确的"必须能从事件里直接读出来，而不是靠读代码猜。
        """

        usable = [item for item in self.cases if item.usable]
        if not usable:
            return {}
        modes = {item.support_mode for item in usable}
        exact = sum(item.support_exact_decisions for item in usable)
        total = sum(item.compared_decisions for item in usable)
        if modes == {"exact_enumeration"}:
            observed = "exact_enumeration"
        elif "exact_enumeration" in modes:
            observed = "mixed"
        else:
            observed = "opcode_alphabet"
        notes = [item.support_note for item in usable if item.support_note]
        return {
            "support_mode": observed,
            "support_mode_declared": self.support.get("support_mode"),
            "support_exact_decisions": exact,
            "support_exact_fraction": round(exact / total, 4) if total else 0.0,
            "support_alignment_notes": notes or None,
        }

    def trace_document(self) -> dict[str, object]:
        """落盘用的完整 trace（曲线与复核都读它）。"""

        return {
            "behavioral_ig": self.value,
            "reason": self.reason,
            "coupling": self.coupling,
            "support": dict(self.support),
            "compared_decisions": self.compared_decisions,
            "cases": [
                {**item.as_dict(), "kl_trace": list(item.kl_trace)} for item in self.cases
            ],
        }


#: 跑一局对局的回调：``(candidate_id, candidate_root, case) -> 结果行``。
#: 结果行里必须有 ``status``；``complete`` 之外的都视作这个 case 不可用。
RunMatch = Callable[[str, Path, BehavioralIgCase], Mapping[str, object]]


def _unavailable(
    reason: str,
    support: Mapping[str, object],
    coupling: str = COUPLING_COMMON_RANDOM,
) -> BehavioralIgMeasurement:
    return BehavioralIgMeasurement(
        value=None, reason=reason, support=dict(support), coupling=coupling
    )


def _measure_case(
    *,
    case: BehavioralIgCase,
    work_root: Path,
    baseline_id: str,
    baseline_root: Path,
    candidate_id: str,
    candidate_root: Path,
    run_match: RunMatch,
    support_size: int,
    epsilon: float,
    replay_timeout_s: float,
    coupling: str,
    keep_recordings: bool = False,
    support_provider: SupportProvider | None = None,
) -> CaseMeasurement:
    recordings = transcript_root(work_root) / case.tag
    snapshots = snapshot_root(work_root) / case.tag
    try:
        return _measure_case_inner(
            case=case,
            recordings=recordings,
            snapshots=snapshots,
            baseline_id=baseline_id,
            baseline_root=baseline_root,
            candidate_id=candidate_id,
            candidate_root=candidate_root,
            run_match=run_match,
            support_size=support_size,
            epsilon=epsilon,
            replay_timeout_s=replay_timeout_s,
            coupling=coupling,
            support_provider=support_provider,
        )
    finally:
        if not keep_recordings:
            # 不限轮数的 run（实验 2）每轮都会产出两份完整快照 + 两条整局字节流。
            # 结论（逐决策 KL trace）已经落盘，原始素材必须回收，否则磁盘先爆。
            for path in (snapshots, recordings):
                shutil.rmtree(path, ignore_errors=True)


def _measure_case_inner(
    *,
    case: BehavioralIgCase,
    recordings: Path,
    snapshots: Path,
    baseline_id: str,
    baseline_root: Path,
    candidate_id: str,
    candidate_root: Path,
    run_match: RunMatch,
    support_size: int,
    epsilon: float,
    replay_timeout_s: float,
    coupling: str,
    support_provider: SupportProvider | None = None,
) -> CaseMeasurement:
    reference_transcript = recordings / "reference.jsonl"
    candidate_transcript = recordings / "candidate.jsonl"

    try:
        recordings.mkdir(parents=True, exist_ok=True)
        # 录制局与之后两次重放共享同一个种子：参考与候选因此走同一条随机流，
        # 动作差异才归因于策略变化而非两次抽样的运气。种子取 case 自己的 seed，
        # 于是不同 case 之间仍然是不同的随机流。
        reference_clone = build_recording_snapshot(
            baseline_root,
            snapshots / "reference",
            reference_transcript,
            coupling=coupling,
            random_seed=case.seed,
        )
        candidate_clone = build_recording_snapshot(
            candidate_root,
            snapshots / "candidate",
            candidate_transcript,
            coupling=coupling,
            random_seed=case.seed,
        )
    except (OSError, FileNotFoundError, ValueError) as error:
        return CaseMeasurement(case, reason=f"recording clone failed: {error}")

    reference_row = run_match(baseline_id, reference_clone, case)
    if str(reference_row.get("status")) != "complete":
        # 一定要把对战器给的 error 带出来：只报 "incomplete" 等于把唯一的线索扔掉。
        detail = str(reference_row.get("error") or "no error detail from arena")
        return CaseMeasurement(
            case,
            reason=(
                f"reference recording match not complete "
                f"({reference_row.get('status')}): {detail}"
            ),
        )
    candidate_row = run_match(candidate_id, candidate_clone, case)

    reference = read_transcript(reference_transcript)
    if reference.error is not None:
        hint = ""
        if "missing" in reference.error:
            # 线上最容易踩的坑：录制目录没被声明为沙箱可写，垫片开不了文件，
            # 于是指标静默变 null。把排查方向直接写进原因里。
            hint = (
                f" — is {recordings.parent} declared in "
                "ContractArena.extra_writable_roots for this run?"
            )
        return CaseMeasurement(
            case, reason=f"reference transcript unusable: {reference.error}{hint}"
        )
    if reference.decision_count == 0:
        return CaseMeasurement(case, reason="reference match produced no player decision")

    # 步骤 3：确定性自校验。父版本在自己的观测流上必须复现自己的动作。
    #
    # 区分两种"不完全一致"，它们的科研含义完全不同：
    # * **动作不同** ⇒ 该策略不是"观测流 → 动作"的确定性函数（用了时钟/随机/读写时序），
    #   ε 通道 KL 的前提不成立 ⇒ 记 null。
    # * **只是提前结束**（前缀完全一致但更短）⇒ 只是重放的尾部收束方式不同（一次性喂完
    #   + 关 stdin 与真实对局的收尾时序不同），共享前缀依然是合法的参考状态集
    #   ⇒ 截断到可复现前缀并把截断量报出来。
    episode = reference.episode(match_id=case.tag, role=case.role)
    expected = episode.action_tokens
    self_replay = replay_actions(
        baseline_root,
        reference,
        timeout_s=replay_timeout_s,
        expected_frames=reference.decision_count,
    )
    if self_replay.error is not None:
        return CaseMeasurement(
            case,
            reference_decisions=reference.decision_count,
            reason=f"reference self-replay failed: {self_replay.error}",
        )
    reproduced = self_replay.action_tokens
    shared = min(len(reproduced), len(expected))
    if reproduced[:shared] != expected[:shared]:
        diverged = next(
            index for index in range(shared) if reproduced[index] != expected[index]
        )
        advice = (
            ""
            if coupling == COUPLING_COMMON_RANDOM
            else " — measurement.behavioral_ig_coupling is 'none'; "
            "'common_random_seed' would make random-using policies measurable"
        )
        return CaseMeasurement(
            case,
            reference_decisions=reference.decision_count,
            reason=(
                "reference policy is not a deterministic function of the observation stream "
                f"(self-replay diverged at decision {diverged}); "
                f"behavioral IG is undefined for it{advice}"
            ),
        )
    self_replay_notes: list[str] = []
    if shared < len(expected):
        if shared == 0:
            return CaseMeasurement(
                case,
                reference_decisions=reference.decision_count,
                reason="reference self-replay reproduced no decision at all",
            )
        self_replay_notes.append(
            f"reference self-replay reproduced only {shared}/{len(expected)} decisions; "
            "comparison truncated to that prefix"
        )
        episode = WireEpisode(
            match_id=episode.match_id,
            role=episode.role,
            decisions=episode.decisions[:shared],
            truncated=True,
        )

    # 步骤 4：候选在同一条冻结流上的动作。
    candidate_replay = replay_actions(
        candidate_root,
        reference,
        timeout_s=replay_timeout_s,
        expected_frames=len(episode.decisions),
    )
    if not candidate_replay.action_bodies:
        detail = candidate_replay.error or (
            "timed out" if candidate_replay.timed_out else "produced no frame"
        )
        return CaseMeasurement(
            case,
            reference_decisions=reference.decision_count,
            reason=f"candidate replay produced no decision: {detail}",
        )

    candidate_tokens = candidate_replay.action_tokens

    # occupancy 位移只在候选自己那局也录成功时才有意义，否则诚实留 null。
    own_observation_ids: tuple[str, ...] = ()
    if str(candidate_row.get("status")) == "complete":
        own = read_transcript(candidate_transcript)
        if own.error is None and own.decision_count:
            own_observation_ids = own.observation_ids

    # |A(s)|：能拿到**逐决策点的真实合法集**就用它，拿不到才回落到常量字母表。
    # 这一步直接决定 KL 的尺度：闭式解 (m−u)·ln(m/u) 里 u = ε/|A|。
    # 实测 antwar 一整局真实 |A(s)| 中位数只有 4，而字母表常量是 10，
    # 98% 的决策点上常量偏大 —— 不修的话每个决策点的 KL 都被系统性压低。
    support_sizes: tuple[int, ...] = ()
    support_mode = "opcode_alphabet"
    support_exact = 0
    if support_provider is not None:
        # 参考局的回放由对战器写出（真后端产物），是状态探针唯一可信的输入。
        raw_replay = reference_row.get("replay_path")
        replay_path = Path(str(raw_replay)) if raw_replay else None
        probed: Sequence[int] = ()
        if replay_path is not None and replay_path.is_file():
            try:
                probed = support_provider(baseline_root, replay_path, case.role)
            except Exception as error:  # noqa: BLE001 - 探针失败只降级，不能毁掉整轮测量
                self_replay_notes.append(
                    f"state probe failed ({type(error).__name__}: {error}); "
                    "falling back to the opcode-alphabet support"
                )
        else:
            self_replay_notes.append(
                "reference match produced no replay; falling back to the opcode-alphabet support"
            )
        if probed:
            # 对齐是硬前提，但"差一两个决策点"和"整体错位"是两件事。
            #
            # 实测教训：antwar 的状态探针按回放记录数给出 246 个决策点，而线协议
            # 记到 247 个（尾部多一帧）。原先的条件是 `len(probed) >= len(decisions)`，
            # 于是 246 >= 247 为假，**整局精确数据被全部丢弃**、静默回落到 |A|=10，
            # 而回落原因还没进事件——连续 14 轮的 IG 都是近似值，我们却以为是精确的。
            #
            # 现在按"缺口"判断：缺口在容差内就用前 n 个精确值，其余决策点由
            # ``wire_decision_samples`` 自动退回字母表常量（它本来就按位置降级）；
            # 缺口过大才说明两者的决策点定义没对上，那时仍然整体回落。
            gap = len(episode.decisions) - len(probed)
            tolerance = max(SUPPORT_ALIGN_ABS_TOLERANCE, int(len(episode.decisions) * 0.01))
            if gap <= 0:
                support_sizes = tuple(int(value) for value in probed[: len(episode.decisions)])
                support_mode = "exact_enumeration"
                support_exact = len(support_sizes)
            elif gap <= tolerance:
                support_sizes = tuple(int(value) for value in probed)
                support_mode = "exact_enumeration"
                support_exact = len(support_sizes)
                self_replay_notes.append(
                    f"state probe covered {len(probed)}/{len(episode.decisions)} decisions "
                    f"(gap {gap} ≤ tolerance {tolerance}); the tail falls back to the "
                    "opcode-alphabet support"
                )
            else:
                self_replay_notes.append(
                    f"state probe produced {len(probed)} decisions but the wire episode has "
                    f"{len(episode.decisions)} (gap {gap} > tolerance {tolerance}); "
                    "falling back to the opcode-alphabet support"
                )

    comparison = compare_wire_policies(
        episode,
        candidate_tokens,
        support_size=support_size,
        epsilon=epsilon,
        candidate_observation_ids=own_observation_ids,
        support_sizes=support_sizes,
    )
    compared = len(comparison.trace)
    notes: list[str] = list(self_replay_notes)
    if len(candidate_tokens) < len(episode.decisions):
        notes.append(
            f"candidate replay stopped after {len(candidate_tokens)}/"
            f"{len(episode.decisions)} decisions"
        )
    if candidate_replay.timed_out:
        notes.append("candidate replay timed out")
    if not reference.complete:
        notes.append("reference recording incomplete")
    if reference.coalesced_decisions:
        # KL 不受影响（只看动作），但 occupancy 的状态粒度变粗了，必须报出来。
        notes.append(
            f"{reference.coalesced_decisions} decision(s) had no fresh observation chunk; "
            "occupancy granularity is coarser than one state per decision"
        )
    return CaseMeasurement(
        case,
        mean_kl_nats=comparison.mean_kl_nats,
        disagreement_rate=comparison.disagreement_rate,
        occupancy_shift=comparison.occupancy_shift,
        compared_decisions=compared,
        reference_decisions=reference.decision_count,
        candidate_decisions=len(candidate_tokens),
        first_divergence=first_divergence(episode, candidate_tokens),
        reason="; ".join(notes) or None,
        kl_trace=tuple(round(item.kl_nats, 6) for item in comparison.trace),
        support_mode=support_mode,
        support_exact_decisions=min(support_exact, compared),
        support_note="; ".join(self_replay_notes) or None,
    )


def measure_behavioral_ig(
    *,
    spec: object,
    epsilon: float,
    work_root: str | Path,
    baseline_id: str,
    baseline_root: str | Path,
    candidate_id: str,
    candidate_root: str | Path,
    cases: Sequence[BehavioralIgCase],
    run_match: RunMatch,
    replay_timeout_s: float = 900.0,
    max_cases: int = 1,
    coupling: str = COUPLING_COMMON_RANDOM,
    keep_recordings: bool = False,
    support_provider: SupportProvider | None = None,
) -> BehavioralIgMeasurement:
    """在若干配对 case 上测量决策级行为信息增益。

    ``spec`` 是 ``agentbench.core.decision_space.InformationGainSpec``（用鸭子类型接收，
    避免 B 在导入期硬依赖 A 的包）。``spec is None`` 表示该游戏没有声明测量契约，
    此时诚实返回 null。

    ``coupling`` 决定要不要用公共随机流耦合两个版本（见
    ``adapters/transcript/coupling.py``）。默认耦合：否则任何调 ``random`` 的选手
    都会在确定性自校验那步被判 null，而真实选手里这类占相当比例。口径会随数上报。

    ``keep_recordings=True`` 会保留录制克隆与字节流（排查用）；默认测完即删，
    否则不限轮数的 run 会把磁盘写满。

    ``support_provider`` 给出**逐决策点的真实 |A(s)|**。给了就用精确枚举，
    没给（或对不齐）就回落到 ``spec`` 声明的字母表常量，并在 ``support_mode`` 里
    如实标注 —— KL 的尺度由 |A| 定（闭式解 ``u = ε/|A|``），口径不写清楚数就没法比。
    """

    try:
        coupling = normalize_coupling(coupling)
    except ValueError as error:
        return _unavailable(str(error), {})
    if spec is None:
        return _unavailable(
            "game declares no information_gain contract in decision_space.yaml",
            {},
            coupling,
        )
    if max_cases <= 0:
        return _unavailable(
            "behavioral IG measurement disabled (max_cases <= 0)", {}, coupling
        )
    selected = tuple(cases)[:max_cases]
    if not selected:
        return _unavailable("no paired case available for behavioral IG", {}, coupling)

    role = selected[0].role
    try:
        support_size = int(spec.support.size_for(role))  # type: ignore[attr-defined]
        support = dict(spec.describe(role))  # type: ignore[attr-defined]
    except Exception as error:  # noqa: BLE001 - 口径缺失必须如实上报
        return _unavailable(
            f"declared support unusable for role {role!r}: {error}", {}, coupling
        )

    work = Path(work_root)
    transcript_root(work).mkdir(parents=True, exist_ok=True)
    measurements: list[CaseMeasurement] = []
    for case in selected:
        try:
            measurements.append(
                _measure_case(
                    case=case,
                    work_root=work,
                    baseline_id=baseline_id,
                    baseline_root=Path(baseline_root),
                    candidate_id=candidate_id,
                    candidate_root=Path(candidate_root),
                    run_match=run_match,
                    support_size=support_size,
                    epsilon=epsilon,
                    replay_timeout_s=replay_timeout_s,
                    coupling=coupling,
                    keep_recordings=keep_recordings,
                    support_provider=support_provider,
                )
            )
        except Exception as error:  # noqa: BLE001 - 单个 case 失败不能拖垮整轮迭代
            measurements.append(
                CaseMeasurement(case, reason=f"{type(error).__name__}: {error}")
            )

    usable = [item for item in measurements if item.usable]
    if not usable:
        reasons = "; ".join(
            f"{item.case.tag}: {item.reason or 'no decision compared'}" for item in measurements
        )
        return BehavioralIgMeasurement(
            value=None,
            reason=f"no case yielded a comparable decision — {reasons}",
            support=support,
            cases=tuple(measurements),
            coupling=coupling,
        )

    weight = sum(item.compared_decisions for item in usable)
    # reason 里的口径也必须是**实际用到的**，不能是静态声明——否则日志和事件
    # 会一起撒同一个谎（我们已经因此误判过一次）。
    observed_modes = {item.support_mode for item in usable}
    exact_decisions = sum(item.support_exact_decisions for item in usable)
    if observed_modes == {"exact_enumeration"}:
        observed_mode = "exact_enumeration"
    elif "exact_enumeration" in observed_modes:
        observed_mode = "mixed"
    else:
        observed_mode = "opcode_alphabet"

    def weighted(field_name: str) -> float | None:
        rows = [
            (getattr(item, field_name), item.compared_decisions)
            for item in usable
            if getattr(item, field_name) is not None
        ]
        if not rows:
            return None
        total = sum(count for _, count in rows)
        return sum(float(value) * count for value, count in rows) / total

    return BehavioralIgMeasurement(
        value=weighted("mean_kl_nats"),
        reason=(
            f"measured on {len(usable)} case(s) / {weight} decisions via transcript replay "
            f"(support_mode={observed_mode}, exact_decisions={exact_decisions}/{weight}, "
            f"declared_|A|={support.get('support_cardinality')}, coupling={coupling})"
        ),
        disagreement_rate=weighted("disagreement_rate"),
        occupancy_shift=weighted("occupancy_shift"),
        compared_decisions=weight,
        support=support,
        cases=tuple(measurements),
        coupling=coupling,
    )

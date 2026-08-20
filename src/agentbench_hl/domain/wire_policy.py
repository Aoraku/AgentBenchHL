"""线协议层面的确定性策略比较 —— 行为信息增益的域逻辑。

背景
----

Plan I 的候选是**一个进程**（``main.py``），不是一个可以在进程内反复调用的
``ai.py`` 对象。所以 :mod:`agentbench_hl.domain.policy` 里那套"原子动作 + 精确合法
支撑集"的探针在 Plan I 上根本没法用，`behavioral_ig` 长期是 null。

本模块换一个**对所有游戏都成立**的观察角度：判题器与选手之间只有一条线协议
（本仓 8 个游戏统一是 ``[len:4 BE][body]``）。于是

* **决策点** = 选手写出的第 i 个回复帧；
* **规范动作** = 该帧帧体的 sha256（逐字节相同 ⇔ 做了同一件事，无需游戏语义）；
* **决策上下文** = 该帧之前新读入的那段观测字节的 sha256（用于 occupancy）。

参考占据分布
------------

比较必须发生在**同一批冻结的决策上下文**上。做法是：录下基线（父版本）真实对局的
完整入站字节流，再把这条流原样喂给候选进程，收集候选的回复。于是候选的内部记忆
``m`` 沿着参考轨迹演化，得到的正是

    ``E_{z ~ d_parent} [ KL( π_parent(·|z) ‖ π_candidate(·|z) ) ]``

即 ``docs/metrics-schema.md`` 里的 ``local_policy_kl_trace``（在参考占据上取平均）。
occupancy 位移用两个版本**各自**的真实对局算，单独报告，永不与 KL 相加。

|A| 的来源
----------

ε 正则通道下确定性策略的 KL 只依赖 |A(s)| 与"是否换了动作"，因此 |A| 必须有出处：
它来自 ``games/<game>/decision_space.yaml`` 的 ``information_gain.support``，并且
``support_mode``（精确枚举 / 操作类型字母表约定）会随每个数一起上报。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from agentbench_hl.domain.policy import (
    BehaviorComparison,
    DecisionSample,
    compare_decisions,
    occupancy_total_variation,
)

#: 合成支撑集里的填充符号前缀。真实 token 是 64 位十六进制 sha256，不会与之冲突。
FILLER_PREFIX = "__A"


@dataclass(frozen=True)
class WireDecision:
    """选手在线协议上的一次决策。"""

    index: int
    observation_id: str
    action_token: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("wire decision index must be non-negative")
        if not self.observation_id or not self.action_token:
            raise ValueError("wire decision requires observation and action ids")


@dataclass(frozen=True)
class WireEpisode:
    """一局对局里某个座次的全部线协议决策。"""

    match_id: str
    role: str
    decisions: tuple[WireDecision, ...]
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.match_id:
            raise ValueError("wire episode requires a match id")
        if not self.role:
            raise ValueError("wire episode requires a role")

    @property
    def action_tokens(self) -> tuple[str, ...]:
        return tuple(item.action_token for item in self.decisions)

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(item.observation_id for item in self.decisions)


def synthetic_support(reference: str, candidate: str, size: int) -> tuple[str, ...]:
    """构造一个大小为 ``size`` 的支撑集，且必然含这两个动作 token。

    我们只知道 |A|（有出处的声明）与两个被实际选中的动作，不知道另外
    ``|A| - 2`` 个动作长什么样。ε 正则 KL 的取值只依赖这三样东西，所以用匿名填充符
    补齐是**精确**的，不是近似：换任何一组互不相同的填充符，KL 一模一样。
    """

    if size < 2:
        raise ValueError("declared support size must be at least 2")
    chosen = [reference] if reference == candidate else [reference, candidate]
    if len(chosen) > size:
        raise ValueError("declared support size is smaller than the observed action count")
    fillers = [f"{FILLER_PREFIX}{index}" for index in range(size - len(chosen))]
    return (*chosen, *fillers)


def wire_decision_samples(
    reference: WireEpisode,
    candidate_actions: tuple[str, ...],
    *,
    support_size: int,
    support_sizes: Sequence[int] = (),
) -> tuple[DecisionSample, ...]:
    """把"参考决策序列 + 候选在同一批上下文上的动作"对齐成可比样本。

    只比较两边都真的产出了动作的前缀：候选如果提前崩溃/超时，缺失的决策**不算作
    "与基线一致"**（那会把一次崩溃粉饰成"行为没变"），而是直接截断并由调用方把
    截断比例报出来。

    ``support_sizes`` 是**逐决策点**的真实 |A(s)|（来自状态探针的合法集枚举）；
    给了就用它，缺位（探针决策点比线协议少）时回落到常量 ``support_size``。

    为什么这件事重要而不是锦上添花：ε-smoothing 后两个确定性策略的 KL 有闭式解
    ``(m−u)·ln(m/u)``，其中 ``u = ε/|A|``、``m = (1−ε)+u`` —— **|A| 直接定标**。
    实测 antwar 一整局的真实 |A(s)| 中位数只有 4（均值 4.2、最小 1、最大 40），
    而按操作类型字母表约定取的常量是 10：**98% 的决策点上常量偏大**，
    于是每个决策点的 KL 被系统性压低。这不是精度问题，是尺度问题。
    """

    length = min(len(reference.decisions), len(candidate_actions))
    return tuple(
        DecisionSample(
            state_id=f"{reference.match_id}:d{decision.index:05d}",
            legal_actions=synthetic_support(
                decision.action_token,
                candidate_actions[position],
                support_sizes[position] if position < len(support_sizes) else support_size,
            ),
            parent_action=decision.action_token,
            candidate_action=candidate_actions[position],
            # 重放发生在参考占据上：两边的决策上下文按构造完全相同。
            # occupancy 位移另用两个版本各自的真实对局计算（见 compare_wire_policies）。
            parent_occupancy=decision.observation_id,
            candidate_occupancy=decision.observation_id,
        )
        for position, decision in enumerate(reference.decisions[:length])
    )


def compare_wire_policies(
    reference: WireEpisode,
    candidate_actions: tuple[str, ...],
    *,
    support_size: int,
    epsilon: float,
    candidate_observation_ids: tuple[str, ...] = (),
    support_sizes: Sequence[int] = (),
) -> BehaviorComparison:
    """参考占据上的 policy KL + 各自 rollout 上的 occupancy 位移。"""

    samples = wire_decision_samples(
        reference,
        candidate_actions,
        support_size=support_size,
        support_sizes=support_sizes,
    )
    comparison = compare_decisions(samples, epsilon=epsilon)
    if not candidate_observation_ids:
        # 没有候选自己的 rollout 就诚实记 null，绝不用"参考流上的 0"冒充"没有位移"。
        return replace(comparison, occupancy_shift=None)
    return replace(
        comparison,
        occupancy_shift=occupancy_total_variation(
            reference.observation_ids, candidate_observation_ids
        ),
    )


def first_divergence(
    reference: WireEpisode, candidate_actions: tuple[str, ...]
) -> int | None:
    """第一次动作分歧发生在第几个决策（都一致则 None）。"""

    for position, decision in enumerate(reference.decisions):
        if position >= len(candidate_actions):
            return None
        if decision.action_token != candidate_actions[position]:
            return decision.index
    return None

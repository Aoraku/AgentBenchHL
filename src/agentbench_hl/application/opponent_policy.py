"""对手选择策略 —— 一轮出 1 个策略、打 b 个对手。

图上的 Action = ``{new code, selected rivals}``：写什么代码由 agent 决定，
打谁由本模块决定。

为什么默认是 k=1 × b 个对手（而不是 k 个候选各打 1 个）
------------------------------------------------------
原设计是"一轮 rollout k=4 个有多样性的候选，各打同一个对手"。它有两个实测问题：

1. **胜率没有分辨力**。固定打同一个对手时，4 轮里胜率要么恒 0 要么恒 1：
   r4b 四轮全是 1.0（对手太弱），g4/m4/snakego4/lostspace4 四轮全是 0.0
   （对手太强）。曲线是一条直线，看不出任何学习。
2. **多样性是硬约束下的伪多样性**。框架要求 k 个候选"代码差异 ≥ N 行且落在
   不同判断路径"，agent 的实际应对是复制一份改几个阈值，于是一轮花掉 4 倍
   对局开销却只探到 1 个点。

改成 k=1 × b 个对手之后：一轮只写一个策略（agent 可以把全部推理预算投进这一个
版本），但拿它去打 b 个不同强度的对手。胜率立刻有 0 / 1/b / 2/b / … / 1 的分辨
率，Elo 反解也从"单点约束"变成"跨强度约束"（b 个不同 Elo 的锚点一起拟合，
比一个锚点稳得多）。回放也从 b 个不同对手那里拿到 b 份不同的证据。

四种选择方式（都可消融，b 也可消融）
------------------------------------
==============  ==============================================================
名称             语义
==============  ==============================================================
``random``       随机选 b 个（抗过拟合；同一轮内不重复，跨轮换人）
``self``         agent 自己读榜 + 读攻打历史决定 b 个（框架只校验可运行性）
``progress``     从第 i..i+b-1 名起，稳定打过（得分率 > 阈值）的就往前进一名，
                 且跳过已经打过的名次
``fix``          固定打榜单前 b 名
==============  ==============================================================

``progress`` 的"跳过已打过的名次"是必须的
-----------------------------------------
b=4 时窗口是 [20, 19, 18, 17]。如果 19 已经被稳定击败并晋级到 18，那么 20
晋级时**不能**也去 19（那是已经打过的），要直接跳到窗口外第一个未打过的名次
（这里是 16）。不跳的话 b 个槽位会互相踩，实际只在少数几个对手上打转。
b=1 时它退化成"一个一个往上升"，也就是原来的 ladder_up。

历史别名
--------
``fixed_top`` / ``fixed_rank`` / ``ladder_up`` / ``ladder_down`` / ``k_diverse``
仍然可用（老配置与已完成的 run 要能复现），但新实验只用上面四个名字。
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

SELF_DECIDE = "self"
# 老名字，仍然接受（已完成的 run 与 exp5 消融配置在用）。
LEGACY_SELF_DECIDE = "self_decide"
DEFAULT_BATCH = 4
DEFAULT_PROGRESS_START = 20
DEFAULT_LADDER_UP_START = 10
DEFAULT_LADDER_DOWN_START = 1
# progress 的晋级门槛：得分率严格高于它才往前进一名。
#: ``progress`` 的晋级判据：**最近 N 轮里至少 W 胜**才算稳定打过。
#:
#: 为什么是"最近 N 轮"而不是"累计得分率 > 0.75"
#: ------------------------------------------
#: 累计得分率有两个毛病，都会让课程走歪：
#:
#: 1. **早期噪声会永久拖住游标**。第 1 轮运气不好 0/2，之后连赢 6 局，
#:    累计得分率是 6/8 = 0.75，不严格大于 0.75 —— 明明已经稳定赢了 3 轮，
#:    游标还卡在原地。
#: 2. **它不衡量"当前实力"而是"历史平均"**。策略在迭代，第 1 轮的败绩
#:    不该被算进第 10 轮的实力判断里。
#:
#: 改成滑动窗口：只看**最近 2 轮**对该对手的 4 局（每轮双座次各 1 局），
#: 至少 3 胜才晋级。这既要求"连续两轮都能赢"（不是单轮运气），
#: 又不会被远古败绩污染。
PROGRESS_WINDOW_ITERATIONS = 2
PROGRESS_WINDOW_WINS = 3

#: 每轮只打**一个**对手的策略：``batch`` 写多少它们都只返回 1 个。
#:
#: 之所以要把这件事也做成一个模块级常量（而不是只有策略类上的
#: :meth:`_Base.effective_batch`）：写 ``run-manifest.json`` 的地方
#: （``adapters/contract/factory.py``）在装配策略实例之前就要落盘，
#: 手里只有策略的**名字**。而 b 是一等实验变量（"一轮 = k × b × 座次"），
#: 清单里写错就等于让所有读账本的人算错对局数。
#:
#: 两条路径必须给出同一个答案，有测试守着（test_effective_batch.py）。
SINGLE_TARGET_POLICIES = frozenset({"ladder_up", "ladder_down", "fixed_rank"})


def effective_batch_for(policy: str, batch: int) -> int:
    """只知道策略**名字**时，这个策略每轮实际打几个对手。

    与 :meth:`_Base.effective_batch` 同口径（有测试逐个策略对齐两者），
    区别只是这里不需要策略实例，因此不受"榜单有多少人"的进一步限制。
    """

    if canonical_policy_name(policy) in SINGLE_TARGET_POLICIES:
        return 1
    return max(1, int(batch))



@dataclass(frozen=True)
class LadderEntry:
    opponent_id: str
    rank: int
    score: float | None


@dataclass(frozen=True)
class OpponentHistory:
    """agent 迄今对每个对手的战绩（用于 progress 晋级与 self 的决策材料）。

    ``by_opponent`` 的每一项是 ``{"played": int, "points": float}``：
    points 是胜 1 / 平 0.5 / 负 0 的累加，与 :mod:`conquest` 同一套口径。

    ``rounds`` 是**逐轮**明细 ``{对手: {轮次: {"played", "points"}}}``。
    progress 的晋级判据需要它：判据是"最近 2 轮共 4 局至少 3 胜"，
    那是个滑动窗口，从累计值里还原不出来。
    """

    by_opponent: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    rounds: Mapping[str, Mapping[int, Mapping[str, float]]] = field(default_factory=dict)

    def played(self, opponent_id: str) -> int:
        entry = self.by_opponent.get(opponent_id) or {}
        return int(entry.get("played") or 0)

    def score_rate(self, opponent_id: str) -> float | None:
        entry = self.by_opponent.get(opponent_id) or {}
        played = float(entry.get("played") or 0)
        if played <= 0:
            return None
        return float(entry.get("points") or 0.0) / played

    def recent_window(
        self, opponent_id: str, *, iterations: int
    ) -> tuple[float, int]:
        """最近 ``iterations`` 轮对该对手的 (得分, 局数)。

        只取**该对手实际出场过**的轮次：progress 的窗口会滑动，
        某个对手可能连续几轮都不在名单里，那些轮次不该被算成"0 胜"。
        """

        per_round = self.rounds.get(opponent_id) or {}
        if not per_round:
            return 0.0, 0
        recent = sorted(per_round)[-max(1, iterations):]
        points = sum(float((per_round[key] or {}).get("points") or 0.0) for key in recent)
        played = sum(int((per_round[key] or {}).get("played") or 0) for key in recent)
        return points, played

    def beaten(
        self,
        opponent_id: str,
        *,
        window_iterations: int = PROGRESS_WINDOW_ITERATIONS,
        window_wins: int = PROGRESS_WINDOW_WINS,
    ) -> bool:
        """是否算"稳定打过"：**最近 window_iterations 轮**里至少 window_wins 胜。

        默认 2 轮 / 3 胜：每轮双座次各 1 局，2 轮就是 4 局，4 局 3 胜。

        窗口必须**填满**才判晋级（``played`` 要够 ``window_iterations`` 轮的量）：
        只打了 1 轮就 2 胜也不能晋级 —— 判据的核心是"连续两轮都赢"，
        单轮全胜可能只是座次或 seed 的运气。
        """

        points, played = self.recent_window(opponent_id, iterations=window_iterations)
        if played < window_wins:
            return False
        return points >= float(window_wins)


class OpponentPolicy(Protocol):
    name: str

    def select(
        self,
        *,
        iteration: int,
        batch: int,
        history: OpponentHistory | None = None,
    ) -> tuple[str, ...]:
        """返回本轮要打的对手（长度 ≤ batch）。空 = 交给 agent 自己决定。"""
        ...

    def instruction(
        self,
        *,
        iteration: int,
        batch: int,
        history: OpponentHistory | None = None,
    ) -> str:
        """注入给 Goal 的自然语言说明。"""
        ...

    def effective_batch(self, batch: int) -> int:
        """这个策略**实际**每轮会打几个对手。见 :meth:`_Base.effective_batch`。"""
        ...


@dataclass(frozen=True)
class _Base:
    ladder: tuple[LadderEntry, ...]

    def effective_batch(self, batch: int) -> int:
        """这个策略**实际**每轮会打几个对手（默认：要多少给多少，受榜单大小限制）。

        为什么需要这个方法而不是直接用配置里的 ``batch``
        ------------------------------------------------
        单目标策略（``ladder_up`` / ``ladder_down`` / ``fixed_rank``）无论
        ``batch`` 写多少都**只返回 1 个**对手。于是配置里的 ``batch: 4`` 是个
        谎：事件账本、``run-manifest.json``、提示词里都写着 4，而实际只打 1 个。

        后果不只是"数字不好看"：

        * 提示词会告诉 agent "这一版会被拿去打 4 个对手，你能从 4 份不同的回放里
          拿到证据"——而它只会拿到 1 份，等于在指导里撒谎；
        * ``watch_runs.py`` 的对手数告警拿 b 当阈值，b 虚高会让告警失灵；
        * 一轮对局数的公式 ``k × b × 座次`` 算出来是实际值的 4 倍，
          读账本的人会以为丢了 3/4 的对局。

        所以口径统一到**策略自己说**：配置写多少不重要，这里返回的才是真值。
        """

        return max(1, min(int(batch), len(self.ladder))) if self.ladder else max(1, int(batch))

    def _sorted(self) -> tuple[LadderEntry, ...]:
        return tuple(sorted(self.ladder, key=lambda item: item.rank))

    def _describe(self, ids: Sequence[str]) -> str:
        by_id = {item.opponent_id: item for item in self.ladder}
        return "、".join(
            f"{item}（rank {by_id[item].rank}）" if item in by_id else item for item in ids
        )


@dataclass(frozen=True)
class FixedTop(_Base):
    """``fix``：固定打榜单前 b 名。"""

    name: str = "fix"

    def select(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> tuple[str, ...]:
        return tuple(item.opponent_id for item in self._sorted()[: max(1, batch)])

    def instruction(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> str:
        targets = self.select(iteration=iteration, batch=batch, history=history)
        return (
            f"本轮固定打榜单前 {len(targets)} 名：{self._describe(targets)}。"
            "擒贼先擒王：你这一个策略要同时面对这几个最强的对手，"
            "所以别为某一个对手做过拟合的特判。"
        )


@dataclass(frozen=True)
class RandomOpponent(_Base):
    """``random``：随机选 b 个（同一轮内不重复）。"""

    seed: int = 0
    name: str = "random"

    def select(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> tuple[str, ...]:
        ordered = self._sorted()
        if not ordered:
            return ()
        rng = random.Random(f"{self.seed}:{iteration}")
        size = min(max(1, batch), len(ordered))
        return tuple(item.opponent_id for item in rng.sample(list(ordered), size))

    def instruction(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> str:
        targets = self.select(iteration=iteration, batch=batch, history=history)
        return (
            f"本轮随机抽到 {len(targets)} 个对手：{self._describe(targets)}。"
            "每轮换人（抗过拟合），所以要追求**普适**的强度，"
            "而不是针对某个对手的定点战术。"
        )


@dataclass(frozen=True)
class Progressive(_Base):
    """``progress``：从第 start..start+b-1 名起，稳定打过的往前进一名。

    晋级规则（确定性、只依赖战绩，因此可复现）：

    1. 初始窗口是 ``[start, start-1, …, start-b+1]``（名次数字变小 = 越来越强）；
    2. 某个槽位的对手若已"稳定打过"（**最近 2 轮共 4 局里至少 3 胜**），
       该槽位往榜首方向前进；
    3. 前进时**跳过所有已经打过的名次**（否则 b 个槽位会互相踩）；
    4. 一直打不赢的对手留在原槽位不动，不会阻塞其它槽位前进。

    b=1 时退化成"一个一个往上升"，即原来的 ``ladder_up``。

    判据为什么是"最近 2 轮 4 局 3 胜"而不是"累计得分率 > 0.75"
    -------------------------------------------------------
    累计得分率会被早期噪声永久拖住：第 1 轮运气不好 0/2、之后连赢 6 局，
    累计是 6/8 = 0.75，**不严格大于** 0.75，于是明明已经连赢三轮，
    游标还卡在原地。而且它衡量的是"历史平均"而不是"当前实力"——
    策略在迭代，第 1 轮的败绩不该算进第 10 轮的判断里。

    滑动窗口两者都解决：要求"连续两轮都能赢"（排除单轮运气），
    又不被远古败绩污染。
    """

    start_rank: int = DEFAULT_PROGRESS_START
    window_iterations: int = PROGRESS_WINDOW_ITERATIONS
    window_wins: int = PROGRESS_WINDOW_WINS
    name: str = "progress"

    def _window(self, batch: int, history: OpponentHistory | None) -> tuple[str, ...]:
        ordered = self._sorted()
        if not ordered:
            return ()
        size = min(max(1, batch), len(ordered))
        # 起点：从 start_rank 往下（数字更大的方向）不一定有人，所以取"名次 ≤
        # start_rank 的最弱那个"作为窗口底部，保证窗口一定落在真实榜单里。
        eligible = [item for item in ordered if item.rank <= self.start_rank] or list(ordered)
        bottom = len(eligible) - 1  # eligible 按 rank 升序，末尾是最弱的
        slots: list[int] = []
        for offset in range(size):
            index = bottom - offset
            if index >= 0:
                slots.append(index)
        if not slots:
            slots = [0]

        if history is None:
            return tuple(eligible[index].opponent_id for index in slots)

        # 已"稳定打过"的名次集合：晋级时要跳过它们。
        beaten = {
            item.opponent_id
            for item in ordered
            if history.beaten(
                item.opponent_id,
                window_iterations=self.window_iterations,
                window_wins=self.window_wins,
            )
        }
        chosen: list[str] = []
        taken: set[str] = set()
        for index in slots:
            cursor = index
            # 一路往榜首方向找第一个"还没稳定打过、且本轮没被别的槽位占掉"的对手。
            while cursor >= 0:
                candidate = eligible[cursor].opponent_id
                if candidate not in beaten and candidate not in taken:
                    break
                cursor -= 1
            if cursor < 0:
                # 整条路径都打过了：说明已经打到榜首，留在最强的那个上巩固。
                candidate = eligible[0].opponent_id
                if candidate in taken:
                    continue
            else:
                candidate = eligible[cursor].opponent_id
            chosen.append(candidate)
            taken.add(candidate)
        return tuple(chosen)

    def select(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> tuple[str, ...]:
        return self._window(batch, history)

    def instruction(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> str:
        targets = self.select(iteration=iteration, batch=batch, history=history)
        if not targets:
            return "榜单为空，本轮对手由框架回落选择。"
        return (
            f"本轮阶梯窗口 {len(targets)} 个对手：{self._describe(targets)}。"
            f"规则：对同一个对手在**最近 {self.window_iterations} 轮**共 "
            f"{self.window_iterations * 2} 局里赢下至少 {self.window_wins} 局，"
            "就算稳定击败，框架会把那个槽位换成更强的一名（已打过的名次会被跳过）；"
            "打不赢的会留在原位，所以请把它研究透。"
        )


@dataclass(frozen=True)
class SelfDecide(_Base):
    """``self``：agent 自己读榜 + 读攻打历史决定 b 个对手。"""

    name: str = SELF_DECIDE

    def select(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> tuple[str, ...]:
        return ()  # 空 = 不干预，由 Goal 在 action.json 里决定

    def instruction(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> str:
        lines = [
            f"对手由你自主决定，本轮请挑 {batch} 个：读 leaderboard.json（含 rank 与 Elo）"
            "与 feedback/ 下你自己的历史战绩，在 action.json 的 selected_rivals 里写"
            f"{batch} 个可运行对手 id（写单个 selected_rival 也兼容）。",
            "选择权真的在你手上，所以要有策略：可以先挑战强者拿信息量，"
            "也可以先扫荡弱者建立稳定的基线；**已经稳定打赢的对手可以不再打**"
            "（重复赢不提供新信息），**一直打不赢的可以先放着**、"
            "等把中间段清完再回来面对。把这一轮的取舍理由写进 rationale。",
        ]
        if history is not None and history.by_opponent:
            beaten = sorted(item for item in history.by_opponent if history.beaten(item))
            stuck = sorted(
                item
                for item in history.by_opponent
                if history.played(item) >= 2 and (history.score_rate(item) or 0.0) <= 0.25
            )
            if beaten:
                lines.append(
                    f"框架统计：已稳定打赢 {len(beaten)} 个对手"
                    f"（{self._describe(beaten[:8])}）。"
                )
            if stuck:
                lines.append(
                    f"框架统计：长期打不赢 {len(stuck)} 个对手"
                    f"（{self._describe(stuck[:8])}）。"
                )
        return "".join(lines)


@dataclass(frozen=True)
class SequentialConquest(_Base):
    """历史别名 ``ladder_up`` / ``ladder_down``：单目标顺序征服。

    新实验用 ``progress``（它在 b=1 时与 ``ladder_up`` 等价）。这里保留是为了
    让已完成的 run 与 exp5 消融配置仍能复现。
    """

    start_rank: int = DEFAULT_LADDER_UP_START
    direction: str = "up"
    name: str = "ladder_up"

    def effective_batch(self, batch: int) -> int:
        """恒为 1：顺序征服一次只打一个目标，``batch`` 写多少都不看。

        exp2 主线就是这个策略配着默认 ``batch: 4`` 在跑，于是账本里写着 4、
        实际只打 1 个。见 :meth:`_Base.effective_batch` 的详注。
        """

        return 1

    def _sequence_entries(self) -> tuple[LadderEntry, ...]:
        ordered = self._sorted()
        if not ordered:
            return ()
        if self.direction == "up":
            pool = [item for item in ordered if item.rank <= self.start_rank] or list(ordered)
            return tuple(sorted(pool, key=lambda item: -item.rank))
        pool = [item for item in ordered if item.rank >= self.start_rank] or list(ordered)
        return tuple(sorted(pool, key=lambda item: item.rank))

    def target_sequence(self) -> tuple[str, ...]:
        return tuple(item.opponent_id for item in self._sequence_entries())

    def _cleared(self, history: OpponentHistory | None) -> int:
        if history is None:
            return 0
        cleared = 0
        for opponent_id in self.target_sequence():
            if history.beaten(opponent_id):
                cleared += 1
            else:
                break
        return cleared

    def select(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> tuple[str, ...]:
        sequence = self._sequence_entries()
        if not sequence:
            return ()
        index = min(max(self._cleared(history), 0), len(sequence) - 1)
        return (sequence[index].opponent_id,)

    def instruction(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> str:
        targets = self.select(iteration=iteration, batch=batch, history=history)
        if not targets:
            return "榜单为空，本轮对手由框架回落选择。"
        arrow = "往榜首方向" if self.direction == "up" else "往榜尾方向"
        return (
            f"本轮固定对手：{self._describe(targets)}。"
            f"课程{arrow}逐个征服，只有稳定击败当前对手才会切到下一个。"
        )


@dataclass(frozen=True)
class FixedRank(_Base):
    """历史别名 ``fixed_rank``：定点打某一个名次。"""

    target_rank: int = 1
    name: str = "fixed_rank"

    def effective_batch(self, batch: int) -> int:
        """恒为 1：定点突破只打那一个名次。见 :meth:`_Base.effective_batch`。"""

        return 1

    def select(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> tuple[str, ...]:
        ordered = self._sorted()
        if not ordered:
            return ()
        exact = [item for item in ordered if item.rank == self.target_rank]
        chosen = (
            exact[0]
            if exact
            else min(ordered, key=lambda item: abs(item.rank - self.target_rank))
        )
        return (chosen.opponent_id,)

    def instruction(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> str:
        return (
            f"本轮固定对手：{self._describe(self.select(iteration=iteration, batch=batch))}"
            f"（定点突破 rank {self.target_rank}）。"
        )


@dataclass(frozen=True)
class KDiverse(_Base):
    """历史别名 ``k_diverse``：在榜单上分层取 b 个（覆盖强中弱）。"""

    name: str = "k_diverse"

    def select(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> tuple[str, ...]:
        ordered = self._sorted()
        if not ordered:
            return ()
        size = max(1, batch)
        if size == 1 or len(ordered) == 1:
            return (ordered[0].opponent_id,)
        step = (len(ordered) - 1) / (size - 1)
        picked: list[str] = []
        for index in range(size):
            entry = ordered[int(round(index * step))]
            if entry.opponent_id not in picked:
                picked.append(entry.opponent_id)
        return tuple(picked)

    def instruction(
        self, *, iteration: int, batch: int, history: OpponentHistory | None = None
    ) -> str:
        targets = self.select(iteration=iteration, batch=batch, history=history)
        return (
            f"本轮按强弱分层取 {len(targets)} 个对手：{self._describe(targets)}。"
            "覆盖强中弱三档，用于看这一个策略在不同强度上的表现。"
        )


# 新名字 -> 实现；老名字通过 _ALIASES 映射过来。
_ALIASES = {
    "fixed_top": "fix",
    LEGACY_SELF_DECIDE: SELF_DECIDE,
}


def canonical_policy_name(name: str) -> str:
    """把历史别名归一到新名字（``fixed_top`` → ``fix``）。"""

    return _ALIASES.get(name, name)


def build_policy(
    name: str,
    ladder: Sequence[LadderEntry],
    *,
    target_rank: int | None = None,
    start_rank: int | None = None,
    seed: int = 0,
    # progress 的晋级判据：最近 N 轮共 2N 局里至少 W 胜。
    # 旧参数名（advance_min_matches / advance_win_rate）已移除：那套"累计得分率"
    # 口径会被早期噪声永久拖住游标（0/2 之后连赢 6 局，累计恰好 0.75，
    # 不严格大于门槛，游标不动）。
    window_iterations: int = PROGRESS_WINDOW_ITERATIONS,
    window_wins: int = PROGRESS_WINDOW_WINS,
) -> OpponentPolicy:
    entries = tuple(ladder)
    if not entries:
        raise ValueError("opponent policy requires a non-empty ladder")
    resolved = canonical_policy_name(name)
    if resolved == "fix":
        return FixedTop(entries)
    if resolved == "random":
        return RandomOpponent(entries, seed=seed)
    if resolved == "progress":
        return Progressive(
            entries,
            start_rank=start_rank or DEFAULT_PROGRESS_START,
            window_iterations=max(1, int(window_iterations)),
            window_wins=max(1, int(window_wins)),
        )
    if resolved == SELF_DECIDE:
        return SelfDecide(entries)
    if resolved == "ladder_up":
        return SequentialConquest(
            entries,
            start_rank=start_rank or DEFAULT_LADDER_UP_START,
            direction="up",
            name="ladder_up",
        )
    if resolved == "ladder_down":
        return SequentialConquest(
            entries,
            start_rank=start_rank or DEFAULT_LADDER_DOWN_START,
            direction="down",
            name="ladder_down",
        )
    if resolved == "fixed_rank":
        return FixedRank(entries, target_rank=target_rank or 1)
    if resolved == "k_diverse":
        return KDiverse(entries)
    raise ValueError(f"unknown opponent policy: {name}")

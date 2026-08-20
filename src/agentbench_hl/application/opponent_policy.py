"""对手选择策略 —— 实验 5 的消融维度。

图上的 Action = ``{new code, selected rival}@k``：谁是 rival 由本模块决定。
``self_decide`` 时框架不干预（Goal 自己从公开排行榜里挑，只做可运行性校验）；
其余策略由框架**规定**本轮对手，并把规定写进给 Goal 的指令，同时在执行时强制生效
（Goal 若写了别的对手，框架按策略覆盖并在事件里留痕，保证消融变量干净）。

实验 5 的四个主设置
-------------------
==================  ====================================================
名称                 语义
==================  ====================================================
``self_decide``      模型自己读榜选对手（框架只校验可运行性）
``fixed_top``        固定打榜首（等价 ``fixed_rank`` + rank=1）
``ladder_up``        从第 ``start_rank`` 名（默认 10）开始**往上**逐个征服
``ladder_down``      从第 ``start_rank`` 名（默认 1）开始**往下**逐个征服
==================  ====================================================

``ladder_*`` 的"打赢了才换人"由 :mod:`agentbench_hl.application.conquest` 判定
（可配置：本轮局数下限 / 得分率下限 / 连续达标轮数）。目标序列只包含**可运行**
且有名次的对手；名次缺口（审计淘汰的选手）会被跳过并在指令里说明，避免课程停在
一个根本跑不起来的对手上。

另外两个辅助策略保留：``random``（抗过拟合）与 ``k_diverse``（k 个候选打不同强度
的对手，做探索多样性）。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

SELF_DECIDE = "self_decide"
DEFAULT_LADDER_UP_START = 10
DEFAULT_LADDER_DOWN_START = 1


@dataclass(frozen=True)
class LadderEntry:
    opponent_id: str
    rank: int
    score: float | None


class OpponentPolicy(Protocol):
    name: str

    def select(self, *, iteration: int, k: int, cleared: int) -> tuple[str, ...]:
        """返回本轮对手；长度为 1（全部候选打同一对手）或 k（每候选一个）。"""
        ...

    def instruction(self, *, iteration: int, k: int, cleared: int) -> str:
        """注入给 Goal 的自然语言说明。"""
        ...

    def target_sequence(self) -> tuple[str, ...]:
        """有序课程的目标序列；非顺序策略返回空元组。

        返回非空时，框架用 :mod:`conquest` 沿这个序列推进游标，``cleared`` 只统计
        **按顺序**征服的对手数。
        """
        ...


@dataclass(frozen=True)
class _Base:
    ladder: tuple[LadderEntry, ...]

    def _sorted(self) -> tuple[LadderEntry, ...]:
        return tuple(sorted(self.ladder, key=lambda item: item.rank))

    def target_sequence(self) -> tuple[str, ...]:
        return ()


@dataclass(frozen=True)
class FixedTop(_Base):
    name: str = "fixed_top"

    def select(self, *, iteration: int, k: int, cleared: int) -> tuple[str, ...]:
        return (self._sorted()[0].opponent_id,)

    def instruction(self, *, iteration: int, k: int, cleared: int) -> str:
        target = self._sorted()[0]
        return (
            f"本轮固定对手：{target.opponent_id}（榜首，rank {target.rank}）。"
            "擒贼先擒王：所有候选都打它。"
        )


@dataclass(frozen=True)
class SequentialConquest(_Base):
    """有序征服：沿一条固定的名次序列逐个推进。"""

    start_rank: int = DEFAULT_LADDER_UP_START
    direction: str = "up"  # up = 名次数字变小（越来越强）；down = 名次数字变大
    name: str = "ladder_up"

    def _sequence_entries(self) -> tuple[LadderEntry, ...]:
        ordered = self._sorted()
        if not ordered:
            return ()
        if self.direction == "up":
            # 从 start_rank 起往榜首方向；start_rank 若超出榜单长度，从最弱的可用对手起。
            pool = [item for item in ordered if item.rank <= self.start_rank]
            if not pool:
                pool = list(ordered)
            return tuple(sorted(pool, key=lambda item: -item.rank))
        pool = [item for item in ordered if item.rank >= self.start_rank]
        if not pool:
            pool = list(ordered)
        return tuple(sorted(pool, key=lambda item: item.rank))

    def target_sequence(self) -> tuple[str, ...]:
        return tuple(item.opponent_id for item in self._sequence_entries())

    def select(self, *, iteration: int, k: int, cleared: int) -> tuple[str, ...]:
        sequence = self._sequence_entries()
        if not sequence:
            return ()
        index = min(max(cleared, 0), len(sequence) - 1)
        return (sequence[index].opponent_id,)

    def instruction(self, *, iteration: int, k: int, cleared: int) -> str:
        sequence = self._sequence_entries()
        if not sequence:
            return "榜单为空，本轮对手由框架回落选择。"
        index = min(max(cleared, 0), len(sequence) - 1)
        entry = sequence[index]
        remaining = len(sequence) - index - 1
        arrow = "往榜首方向" if self.direction == "up" else "往榜尾方向"
        finished = cleared >= len(sequence)
        if finished:
            return (
                f"目标序列已全部征服（{len(sequence)} 个），本轮继续巩固最后一个对手："
                f"{entry.opponent_id}（rank {entry.rank}）。"
            )
        return (
            f"本轮固定对手：{entry.opponent_id}（rank {entry.rank}）。"
            f"课程：从 rank {sequence[0].rank} 起{arrow}逐个征服，"
            f"已征服 {cleared} 个，之后还有 {remaining} 个。"
            "只有**稳定击败**当前对手，框架才会把目标切到下一个；"
            "所以请把这一个对手研究透，不要跳级。"
        )


@dataclass(frozen=True)
class FixedRank(_Base):
    target_rank: int = 1
    name: str = "fixed_rank"

    def select(self, *, iteration: int, k: int, cleared: int) -> tuple[str, ...]:
        ordered = self._sorted()
        exact = [item for item in ordered if item.rank == self.target_rank]
        chosen = (
            exact[0]
            if exact
            else min(ordered, key=lambda item: abs(item.rank - self.target_rank))
        )
        return (chosen.opponent_id,)

    def instruction(self, *, iteration: int, k: int, cleared: int) -> str:
        return (
            f"本轮固定对手：{self.select(iteration=iteration, k=k, cleared=cleared)[0]}"
            f"（定点突破 rank {self.target_rank}）。"
        )


@dataclass(frozen=True)
class RandomOpponent(_Base):
    seed: int = 0
    name: str = "random"

    def select(self, *, iteration: int, k: int, cleared: int) -> tuple[str, ...]:
        rng = random.Random(f"{self.seed}:{iteration}")
        return (rng.choice(self._sorted()).opponent_id,)

    def instruction(self, *, iteration: int, k: int, cleared: int) -> str:
        return (
            f"本轮随机对手：{self.select(iteration=iteration, k=k, cleared=cleared)[0]}"
            "（抗过拟合，每轮换人）。"
        )


@dataclass(frozen=True)
class KDiverse(_Base):
    name: str = "k_diverse"

    def select(self, *, iteration: int, k: int, cleared: int) -> tuple[str, ...]:
        ordered = self._sorted()
        if not ordered:
            return ()
        if k <= 1 or len(ordered) == 1:
            return (ordered[0].opponent_id,)
        # 在榜单上分层取 k 个（覆盖强中弱），保证候选之间对手不同。
        step = (len(ordered) - 1) / (k - 1)
        picked: list[str] = []
        for index in range(k):
            entry = ordered[int(round(index * step))]
            if entry.opponent_id not in picked:
                picked.append(entry.opponent_id)
        while len(picked) < k:  # 榜单过短时允许重复，保持长度 = k
            picked.append(ordered[len(picked) % len(ordered)].opponent_id)
        return tuple(picked)

    def instruction(self, *, iteration: int, k: int, cleared: int) -> str:
        targets = self.select(iteration=iteration, k=k, cleared=cleared)
        listing = "、".join(targets)
        return (
            f"本轮 {len(targets)} 个候选**分别**打不同对手（按强弱分层）：{listing}。"
            "第 i 个候选对应第 i 个对手，用于多样化探索。"
        )


@dataclass(frozen=True)
class SelfDecide(_Base):
    name: str = SELF_DECIDE

    def select(self, *, iteration: int, k: int, cleared: int) -> tuple[str, ...]:
        return ()  # 空 = 不干预，由 Goal 在 action.json 里决定

    def instruction(self, *, iteration: int, k: int, cleared: int) -> str:
        return (
            "对手由你自主决定：读 leaderboard.json（含 rank 与 Elo），"
            "在 action.json 的 selected_rival 里写一个可运行对手 id，并说明理由。"
            "选择理由请写进 rationale（我们会分析你的选敌偏好）。"
        )


def build_policy(
    name: str,
    ladder: Sequence[LadderEntry],
    *,
    target_rank: int | None = None,
    start_rank: int | None = None,
    seed: int = 0,
) -> OpponentPolicy:
    entries = tuple(ladder)
    if not entries:
        raise ValueError("opponent policy requires a non-empty ladder")
    if name == "fixed_top":
        return FixedTop(entries)
    if name == "ladder_up":
        return SequentialConquest(
            entries,
            start_rank=start_rank or DEFAULT_LADDER_UP_START,
            direction="up",
            name="ladder_up",
        )
    if name == "ladder_down":
        return SequentialConquest(
            entries,
            start_rank=start_rank or DEFAULT_LADDER_DOWN_START,
            direction="down",
            name="ladder_down",
        )
    if name == "fixed_rank":
        return FixedRank(entries, target_rank=target_rank or 1)
    if name == "random":
        return RandomOpponent(entries, seed=seed)
    if name == "k_diverse":
        return KDiverse(entries)
    if name == SELF_DECIDE:
        return SelfDecide(entries)
    raise ValueError(f"unknown opponent policy: {name}")

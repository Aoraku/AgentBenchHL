"""Goal-led 多轮驱动器 —— 一条命令跑 N 轮，可中断可恢复。

原来 `goal-led start/continue` 每次只推进**一个** Action/反馈点，必须人在外面反复
敲命令；服务化场景需要"设定轮数 → 后台自己跑完"。本模块提供这个循环，并在每轮
之间检查：

- 轮数上限（表单的"最大迭代轮数"，或 config 的 ``runtime.max_iterations``）；
- token / wall 预算（``budget.*``）；
- 失败处理：单轮异常记 ``GoalLedIterationFailed`` 并停止（保留可 resume 的状态）。

驱动器**不做**任何科学决策，只做节流与记账。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentbench_hl.application.goal_led_service import GoalLedOutcome, GoalLedService
from agentbench_hl.domain.events import FinalizedEvent


@dataclass
class DriveResult:
    iterations_completed: int
    stop_reason: str
    outcomes: list[GoalLedOutcome] = field(default_factory=list)
    error: str | None = None

    @property
    def last(self) -> GoalLedOutcome | None:
        return self.outcomes[-1] if self.outcomes else None

    def as_dict(self) -> dict[str, Any]:
        last = self.last
        return {
            "iterations_completed": self.iterations_completed,
            "stop_reason": self.stop_reason,
            "error": self.error,
            "last_request_id": None if last is None else last.request_id,
            "last_win_rate": None if last is None else last.win_rate,
            "thread_id": None if last is None else last.thread_id,
        }


def drive(
    service: GoalLedService,
    *,
    iterations: int | None,
    on_iteration: Callable[[GoalLedOutcome], None] | None = None,
) -> DriveResult:
    """推进 ``iterations`` 轮（第一次自动 start，其后 advance）。

    ``iterations=None`` 表示**不限轮数**：一直跑到预算耗尽
    （``budget.tokens`` / ``budget.wall_seconds``）或出错为止。
    实验 2 需要这个——"不限迭代轮数持续迭代，尽可能刷到 SOTA"。
    以前只能填一个很大的数字，于是"跑到预算耗尽"这件事得靠人盯着；
    更糟的是那个大数字还会被 ``max_corrections = max(3, iterations)`` 放大成
    "允许几万次协议纠正"，一个坏掉的候选能把预算烧光都不停。

    无限模式下**必须**有预算，否则拒绝启动：没有任何停止条件的后台长跑
    会一直烧 token，这不是实验而是事故。
    """

    if iterations is not None and iterations < 1:
        raise ValueError("iterations must be positive (or None for unlimited)")

    unlimited = iterations is None
    if unlimited:
        budget = service.budget_status()
        if budget.get("token_budget") is None and budget.get("wall_budget_s") is None:
            raise ValueError(
                "unlimited iterations require a budget: set budget.tokens or "
                "budget.wall_seconds (否则这个 run 没有任何停止条件)"
            )

    completed = 0
    outcomes: list[GoalLedOutcome] = []
    started = service._state_path.exists()  # noqa: SLF001 - 同包协作，语义即"是否已 start"
    stop_reason = "iterations_exhausted"
    error: str | None = None
    corrections = 0
    # 协议纠正的额度与轮数**彻底解耦**：它衡量的是"agent 连续写不对 action.json"，
    # 跟要跑多少轮无关。原来是 max(3, iterations)——想跑 5000 轮就等于容忍 5000 次
    # 连续协议错误，一个坏掉的候选能空转到预算烧光都不停。
    # 计数用"连续"而不是"累计"：偶尔写错一次并自我纠正是正常的学习过程。
    MAX_CONSECUTIVE_CORRECTIONS = 3
    consecutive_corrections = 0

    while True:
        if not unlimited and completed >= iterations:  # type: ignore[operator]
            break
        budget = service.budget_status()
        if budget.get("exhausted"):
            stop_reason = str(budget["exhausted"])
            break
        # 这一轮是不是最后一轮？是的话就不要再驱动 agent 去想下一轮——
        # ``advance()`` 里的反馈投递是**同步**的（会一直等到 agent 写出下一份
        # action.json），而循环随后就退出，那份产出必然被丢弃。
        # 实测这一次白干 = 1369s 墙钟 + 66k input tokens。
        final_iteration = not unlimited and completed + 1 >= iterations  # type: ignore[operator]
        try:
            outcome = (
                service.advance(prompt_next=not final_iteration)
                if started
                else service.start(prompt_next=not final_iteration)
            )
        except ValueError as exc:
            # 协议层问题（没写 action.json / 字段非法 / request_id 重复）：不终止 run，
            # 把错误当反馈交回 Goal 让它自我纠正。
            started = True
            corrections += 1
            consecutive_corrections += 1
            service.events.append(
                FinalizedEvent.create(
                    "GoalLedCorrectionRequested",
                    {"attempt": corrections, "error": str(exc)},
                    f"goal-led-correction:{corrections}:{int(time.time())}",
                )
            )
            if consecutive_corrections > MAX_CONSECUTIVE_CORRECTIONS:
                error = f"ValueError: {exc}"
                stop_reason = "too_many_protocol_corrections"
                break
            try:
                service.request_correction(str(exc))
                continue
            except BaseException as inner:  # noqa: BLE001
                error = f"{type(inner).__name__}: {inner}"
                stop_reason = "iteration_failed"
                break
        except BaseException as exc:  # noqa: BLE001 - 单轮失败不应丢掉已有进度
            error = f"{type(exc).__name__}: {exc}"
            stop_reason = "iteration_failed"
            service.events.append(
                FinalizedEvent.create(
                    "GoalLedIterationFailed",
                    {"iteration": completed + 1, "error": error},
                    f"goal-led-failed:{completed + 1}:{int(time.time())}",
                )
            )
            break
        started = True
        completed += 1
        consecutive_corrections = 0
        outcomes.append(outcome)
        if on_iteration is not None:
            on_iteration(outcome)

    service.events.append(
        FinalizedEvent.create(
            "GoalLedDriveFinished",
            {
                "iterations_completed": completed,
                "iterations_requested": iterations,
                "unlimited": unlimited,
                "stop_reason": stop_reason,
                "error": error,
                "corrections": corrections,
                "budget": service.budget_status(),
            },
            f"goal-led-drive:{int(time.time())}:{completed}",
        )
    )
    return DriveResult(
        iterations_completed=completed, stop_reason=stop_reason, outcomes=outcomes, error=error
    )

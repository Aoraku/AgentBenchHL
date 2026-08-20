"""``goal_led_driver.drive`` 的轮数与停止条件。

实验 2 要"不限迭代轮数持续迭代，尽可能刷到 SOTA"。以前只能填一个很大的
``--iterations``，于是：

* "跑到预算耗尽"这件事得靠人盯着；
* 那个大数字还会被 ``max_corrections = max(3, iterations)`` 放大成
  "允许几万次协议纠正"——一个坏掉的候选能把预算烧光都不停。

所以这里钉住三件事：``None`` = 无限、无限模式必须有预算、纠正额度与轮数解耦。
"""

from __future__ import annotations

import pytest

from agentbench_hl.application.goal_led_driver import drive


class FakeEvents:
    def __init__(self) -> None:
        self.appended: list[object] = []

    def append(self, event: object) -> None:
        self.appended.append(event)

    def types(self) -> list[str]:
        return [getattr(event, "event_type", "?") for event in self.appended]


class FakeService:
    """最小 service 替身：只实现 drive 用到的那几个协作点。"""

    def __init__(
        self,
        *,
        token_budget: int | None = None,
        wall_budget: float | None = None,
        stop_after: int | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.events = FakeEvents()
        self.starts = 0
        self.advances = 0
        self.corrections: list[str] = []
        #: 每次 start/advance 收到的 prompt_next，用来钉住"末轮不再驱动 agent 思考"。
        self.prompt_next_calls: list[bool] = []
        self._token_budget = token_budget
        self._wall_budget = wall_budget
        self._stop_after = stop_after
        self._fail_with = fail_with
        self._state = False

    # drive 用 _state_path.exists() 判断"是否已 start"
    @property
    def _state_path(self):  # noqa: ANN202 - 仅供 drive 探测
        class _Probe:
            def __init__(self, exists: bool) -> None:
                self._exists = exists

            def exists(self) -> bool:
                return self._exists

        return _Probe(self._state)

    def budget_status(self) -> dict[str, object]:
        done = self.starts + self.advances
        exhausted = None
        if self._stop_after is not None and done >= self._stop_after:
            exhausted = "token_budget"
        return {
            "tokens": done * 100,
            "token_budget": self._token_budget,
            "elapsed_s": float(done),
            "wall_budget_s": self._wall_budget,
            "exhausted": exhausted,
        }

    def start(self, *, prompt_next: bool = True):  # noqa: ANN201 - 返回值 drive 只做透传
        self.starts += 1
        self._state = True
        self.prompt_next_calls.append(prompt_next)
        if self._fail_with is not None:
            raise self._fail_with
        return f"outcome-{self.starts}"

    def advance(self, *, prompt_next: bool = True):  # noqa: ANN201
        self.advances += 1
        self.prompt_next_calls.append(prompt_next)
        if self._fail_with is not None:
            raise self._fail_with
        return f"outcome-{self.starts + self.advances}"

    def request_correction(self, message: str) -> None:
        self.corrections.append(message)


def test_fixed_iterations_run_exactly_that_many_times() -> None:
    service = FakeService()
    result = drive(service, iterations=3)
    assert result.iterations_completed == 3
    assert result.stop_reason == "iterations_exhausted"
    assert (service.starts, service.advances) == (1, 2)


def test_last_iteration_does_not_prompt_the_agent_again() -> None:
    """末轮不能再驱动 agent 想下一轮——那份产出必然被丢弃。

    反馈投递是同步的（会等到 agent 写出下一份 action.json 才返回），而循环随后就退出。
    实测这一次白干 = 1369s 墙钟 + 66k input tokens，是全流程里最容易省掉的一块。
    """

    service = FakeService()
    drive(service, iterations=3)

    assert service.prompt_next_calls == [True, True, False]


def test_single_iteration_run_never_prompts_for_a_second_round() -> None:
    service = FakeService()
    drive(service, iterations=1)

    assert service.prompt_next_calls == [False]


def test_unlimited_runs_always_prompt_because_the_end_is_not_known_in_advance() -> None:
    """无限模式下无法预知哪轮是最后一轮，只能照常驱动。

    钉住它是为了说明：这里**不做**投机式的"猜预算快没了就不问"——猜错会白白截断
    一轮真实迭代，比省下那次思考的代价大得多。
    """

    service = FakeService(token_budget=10_000, stop_after=3)
    drive(service, iterations=None)

    assert service.prompt_next_calls == [True, True, True]


def test_zero_or_negative_iterations_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        drive(FakeService(), iterations=0)


def test_unlimited_runs_until_the_budget_is_exhausted() -> None:
    """``iterations=None`` = 不限轮数，靠预算收尾。"""

    service = FakeService(token_budget=10_000, stop_after=5)
    result = drive(service, iterations=None)
    assert result.iterations_completed == 5
    assert result.stop_reason == "token_budget"


def test_unlimited_without_any_budget_is_refused() -> None:
    """没有预算的无限跑没有任何停止条件——那不是实验，是烧钱事故。"""

    with pytest.raises(ValueError, match="unlimited iterations require a budget"):
        drive(FakeService(), iterations=None)


def test_wall_budget_alone_is_enough_to_allow_unlimited() -> None:
    service = FakeService(wall_budget=3600.0, stop_after=2)
    result = drive(service, iterations=None)
    assert result.iterations_completed == 2


def test_protocol_corrections_are_capped_independently_of_iterations() -> None:
    """纠正额度衡量的是"agent 连续写不对 action.json"，与总轮数无关。

    以前 ``max_corrections = max(3, iterations)``：要跑 5000 轮就等于容忍 5000 次
    连续协议错误，一个坏候选可以一直空转到预算耗尽。
    """

    service = FakeService(token_budget=10**9, fail_with=ValueError("no action.json"))
    result = drive(service, iterations=5000)
    assert result.stop_reason == "too_many_protocol_corrections"
    # 3 次纠正机会 + 第 4 次触顶
    assert len(service.corrections) == 3
    assert "GoalLedCorrectionRequested" in service.events.types()


def test_drive_finished_event_records_the_unlimited_flag() -> None:
    service = FakeService(token_budget=10_000, stop_after=1)
    drive(service, iterations=None)
    finished = service.events.appended[-1]
    payload = finished.payload  # type: ignore[attr-defined]
    assert payload["unlimited"] is True
    assert payload["iterations_requested"] is None
    assert payload["stop_reason"] == "token_budget"

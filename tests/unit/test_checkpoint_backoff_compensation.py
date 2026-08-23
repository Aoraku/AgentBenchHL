"""checkpoint 预算不该被上游限流吃掉。

实测事故
--------
random 组死在第 26 轮，报::

    TimeoutError: Codex Goal did not reach a checkpoint in time

而当时日志里是连续十几次 ``HTTP 503``。agent 根本没有卡住 —— 是限流的退避
等待把这个**固定墙钟**耗光了。最坏的地方是它**归错了因**：看事件账本会以为
"模型这一轮没产出候选"，实际是基建抖动，而基建抖动是等得起的。

修法：proxy 累计"在退避上等了多久"，app_server 每次检查剩余时间时把这段
时间加回 deadline。于是"等上游"不占用 agent 的思考预算，而 agent 真的卡住时
（退避计数器不增长）仍然会照常超时。
"""

from __future__ import annotations

import time

from agentbench_hl.adapters.codex_goal.responses_proxy import ResponsesCompatProxy


def test_proxy_starts_with_no_backoff_recorded() -> None:
    proxy = ResponsesCompatProxy("https://relay.invalid/v1")

    assert proxy.backoff_seconds == 0.0
    assert proxy.absorbed_failures == 0


def test_backoff_accumulates_across_requests() -> None:
    """多次请求的退避时间要累加：一轮 turn 里可能有几十次模型请求。"""

    proxy = ResponsesCompatProxy("https://relay.invalid/v1")

    proxy._record_backoff(12.5, 3)  # noqa: SLF001 - 测的就是这个记账
    proxy._record_backoff(30.0, 5)  # noqa: SLF001

    assert proxy.backoff_seconds == 42.5
    assert proxy.absorbed_failures == 8


def test_negative_durations_are_ignored() -> None:
    """时钟回拨不该让补偿变成惩罚（deadline 反而缩短）。"""

    proxy = ResponsesCompatProxy("https://relay.invalid/v1")

    proxy._record_backoff(-5.0, -1)  # noqa: SLF001

    assert proxy.backoff_seconds == 0.0
    assert proxy.absorbed_failures == 0


def test_deadline_compensation_extends_the_budget() -> None:
    """复现 app_server 的补偿算式，确认限流不会误杀一轮迭代。

    场景：checkpoint 预算 300s，agent 已经思考 280s，期间被限流拖了 200s。
    * 不补偿：剩余 20s，再等一会就超时 —— 这正是 random 组的死法。
    * 补偿后：deadline 往后挪 200s，agent 实际只用了 80s 思考，还有余量。
    """

    checkpoint_timeout_s = 300.0
    proxy = ResponsesCompatProxy("https://relay.invalid/v1")

    backoff_at_start = proxy.backoff_seconds
    started = time.monotonic()
    deadline = started + checkpoint_timeout_s

    # 限流累计等了 200s（真实的墙钟推进这里用算式模拟，不 sleep）。
    proxy._record_backoff(200.0, 14)  # noqa: SLF001
    now = started + 280.0

    without = deadline - now
    compensation = proxy.backoff_seconds - backoff_at_start
    with_compensation = deadline + compensation - now

    assert without == 20.0
    assert with_compensation == 220.0
    assert with_compensation > without


def test_agent_that_is_genuinely_stuck_still_times_out() -> None:
    """agent 真的卡住时（没有退避）必须照常超时 —— 补偿不能变成永不超时。"""

    checkpoint_timeout_s = 300.0
    proxy = ResponsesCompatProxy("https://relay.invalid/v1")

    backoff_at_start = proxy.backoff_seconds
    started = time.monotonic()
    deadline = started + checkpoint_timeout_s
    now = started + 301.0  # 思考 301s，一次退避都没有

    compensation = proxy.backoff_seconds - backoff_at_start
    assert compensation == 0.0
    assert deadline + compensation - now < 0, "没有限流时必须照常超时"


def test_compensation_is_measured_per_turn_not_cumulatively() -> None:
    """补偿要用"本轮新增的退避"，不是进程启动至今的总量。

    否则第 10 轮会白拿前 9 轮累积的几千秒补偿，checkpoint 保护形同失效。
    """

    proxy = ResponsesCompatProxy("https://relay.invalid/v1")
    proxy._record_backoff(1000.0, 50)  # noqa: SLF001 - 前几轮攒下的

    backoff_at_start = proxy.backoff_seconds  # 本轮开始时的快照
    proxy._record_backoff(30.0, 2)  # noqa: SLF001 - 本轮新增

    assert proxy.backoff_seconds - backoff_at_start == 30.0

"""上游退避的补偿必须**实时**计入，不能等请求返回才补。

背景：这个补偿机制原来恰好漏掉了它要保护的那种情况
------------------------------------------------
checkpoint 超时是一个**固定墙钟**（``app_server`` 的 ``checkpoint_timeout_s``），
而上游退避的等待会直接吃掉它。所以 ``responses_proxy`` 会累计退避耗时，让
``app_server`` 把它从 deadline 里扣掉。

但记账原来只在 ``request_bytes_with_backoff`` **返回之后**做一次。于是一个在内部
反复重试几十分钟的请求，在它返回之前一秒都不会被计入 —— 而这正是补偿机制存在的
唯一理由。**要保护的场景恰好是唯一没被覆盖的场景。**

实测 fix3-aquawar 第 2 轮：前面几个带重试的请求 credited 721s，最后那个请求还在
长重试循环里（``IncompleteRead`` 每次都要先生成几分钟才断，一断整段重来）就撞上
deadline::

    TimeoutError: Codex Goal did not reach a checkpoint in time
    （已扣除上游退避等待 721s，仍超过 3600s 预算）

那一轮的 8 局对局本身是干净的，整个 run 却以 ``iteration_failed`` 报废；
而从账本上看像"agent 想了一小时"，其实是几十分钟被丢弃的生成。
"""

from __future__ import annotations

import time

from agentbench_hl.adapters.codex_goal.responses_proxy import ResponsesCompatProxy


def _proxy() -> ResponsesCompatProxy:
    return ResponsesCompatProxy("http://127.0.0.1:1")


def test_starts_at_zero() -> None:
    assert _proxy().backoff_seconds == 0.0


def test_inflight_retry_is_credited_before_it_finishes() -> None:
    """核心断言：请求还没返回，它的重试耗时就必须已经被计入。"""

    proxy = _proxy()
    proxy._mark_retrying(1, time.monotonic() - 500.0)

    # 修复前这里是 0.0 —— deadline 检查因此看不到这 500 秒。
    assert proxy.backoff_seconds >= 500.0


def test_finishing_does_not_double_count() -> None:
    """收尾时先摘在飞标记再记账，同一段时间不能算两次。"""

    proxy = _proxy()
    proxy._mark_retrying(1, time.monotonic() - 500.0)
    proxy._clear_retrying(1)
    proxy._record_backoff(500.0, 3)

    assert 500.0 <= proxy.backoff_seconds < 1000.0
    assert proxy.absorbed_failures == 3


def test_multiple_inflight_requests_are_summed() -> None:
    """并发请求各自计时（ThreadingHTTPServer 会同时有多个）。"""

    proxy = _proxy()
    now = time.monotonic()
    proxy._mark_retrying(1, now - 100.0)
    proxy._mark_retrying(2, now - 200.0)

    assert proxy.backoff_seconds >= 300.0


def test_marking_twice_keeps_the_earliest_start() -> None:
    """同一个请求重复登记不能把起点往后挪（否则少扣时间）。"""

    proxy = _proxy()
    now = time.monotonic()
    proxy._mark_retrying(1, now - 400.0)
    proxy._mark_retrying(1, now - 10.0)  # 第二次应被忽略

    assert proxy.backoff_seconds >= 400.0


def test_clearing_an_unknown_token_is_harmless() -> None:
    """没登记过的请求收尾时不能抛异常（正常请求走的就是这条路）。"""

    proxy = _proxy()
    proxy._clear_retrying(12345)
    assert proxy.backoff_seconds == 0.0

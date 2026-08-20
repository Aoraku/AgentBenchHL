"""上游 LLM 请求的重试退避。

为什么需要
----------
两个 harness 桥（``codex_goal/responses_proxy.py``、``cc_goal/anthropic_bridge.py``）
都是把 CLI 的请求原样转发给中转站。之前**任何**上游抖动都会直接落到那一轮
Goal turn 上：

* ``429``（限流）：批量实验会同时跑多个 run，每个 run 内还有多轮，撞限流是常态；
* ``502/503/504``：中转站重启或上游拥塞时的常见返回；
* ``URLError`` / 读超时：网络瞬断。

这些都会表现为"这一轮 agent 没产出候选"，而事件账本里只留下一次失败的迭代——
看起来像模型能力问题，实际是基建抖动。**这类混淆必须在最底层消掉。**

设计要点
--------
* 只重试**幂等且明确可重试**的状态码；``4xx`` 里除限流/超时类一律不重试
  （比如 ``400`` 参数错、``401`` 凭据错，重试只会浪费时间并掩盖真实原因）；
* 尊重上游的 ``Retry-After``；
* 指数退避 + **随机抖动**：多个 worker 同时被限流时，固定间隔会让它们继续
  同步撞墙（惊群）；
* 重试耗尽后**保持原有语义**把最后一次响应/异常交回调用方，不自己编造响应；
* 日志走调用方给的 ``log``（通常是 stderr），不要污染转发出去的响应体。

两次实测事故与对应的加固
------------------------
**其一：退避预算太短。** 原来 ``attempts=5, max_delay=30`` 的总等待上限约
25 s，而中转站的限流窗口是分钟级。实测 antwar2 连续 15 次 ``429/503``
把 5 次额度用光后终止，run 死在第 56 轮。现在改为按**总退避预算**
（``max_total_delay``，默认 10 分钟）驱动重试，次数只是上限而不是瓶颈——
限流是**等得起**的故障，等 10 分钟远好过丢掉一轮几十万 token 的迭代。

**其二：读响应体的失败根本没进重试路径。** 退避只包住了
``urlopen``，而 glm 的长响应经常在 ``response.read()`` 阶段断流
（``http.client.IncompleteRead``）。实测 antwar 就是这样死的：
502 重试成功后，接着在读 205 KB 响应体时断掉，异常直接冒到 HTTP 处理线程。
所以本模块额外提供 ``request_bytes_with_backoff``——**把读体也放进重试循环**。
只有这个函数能保证"一次完整的请求-响应"具备原子的重试语义。
"""

from __future__ import annotations

import http.client
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "RETRYABLE_STATUS",
    "UpstreamResponse",
    "request_bytes_with_backoff",
    "urlopen_with_backoff",
]

# 408 请求超时 / 409 冲突 / 425 too early / 429 限流 / 5xx 上游故障。
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

# 读响应体阶段的可重试异常：流被上游截断、连接被对端关掉、超时。
# 这些都意味着"这次请求没拿到完整答案"，重发是安全的（请求本身幂等）。
RETRYABLE_READ_ERRORS = (
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    ConnectionError,
    TimeoutError,
)

DEFAULT_ATTEMPTS = 12
DEFAULT_MAX_DELAY = 90.0
#: 单次请求允许累计等待多久（秒）。限流是等得起的故障，见模块头注释。
DEFAULT_MAX_TOTAL_DELAY = 600.0


@dataclass(frozen=True)
class UpstreamResponse:
    """一次**已经读完**的上游响应。"""

    status: int
    headers: dict[str, str]
    body: bytes

    @property
    def content_type(self) -> str | None:
        for name, value in self.headers.items():
            if name.lower() == "content-type":
                return value
        return None


def _retry_after_seconds(error: urllib.error.HTTPError) -> float | None:
    """解析 ``Retry-After``（只认秒数形式；HTTP 日期形式少见且解析易错）。"""

    raw = error.headers.get("Retry-After") if error.headers else None
    if not raw:
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _delay(attempt: int, *, base: float, cap: float) -> float:
    """指数退避 + 抖动。attempt 从 1 起。"""

    window = min(cap, base * (2 ** (attempt - 1)))
    return window * (0.75 + 0.5 * random.random())  # noqa: S311 - 退避抖动，非密码学用途


def urlopen_with_backoff(
    request: urllib.request.Request,
    *,
    timeout_s: float,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = 1.0,
    max_delay: float = DEFAULT_MAX_DELAY,
    max_total_delay: float = DEFAULT_MAX_TOTAL_DELAY,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """发请求，可重试的失败按指数退避重试（**不含**读响应体）。

    返回值与 ``urllib.request.urlopen`` 一致；若最终仍是 HTTP 错误，则**返回**
    那个 ``HTTPError``（它同样是个可读的响应对象），由调用方决定怎么转发——
    这与两个桥原有的 ``except HTTPError as error: response = error`` 写法一致。
    连接层异常在重试耗尽后照常抛出。

    需要"读体也可重试"时用 ``request_bytes_with_backoff``。
    """

    last_http_error: urllib.error.HTTPError | None = None
    spent = 0.0
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout_s)  # noqa: S310 - 固定上游
        except urllib.error.HTTPError as error:
            last_http_error = error
            exhausted = attempt >= attempts or spent >= max_total_delay
            if error.code not in RETRYABLE_STATUS or exhausted:
                return error
            pause = _retry_after_seconds(error)
            if pause is None:
                pause = _delay(attempt, base=base_delay, cap=max_delay)
            error.close()
            if log:
                log(
                    f"[llm-retry] HTTP {error.code}，{pause:.1f}s 后重试"
                    f"（第 {attempt}/{attempts} 次，已等待 {spent:.0f}s）"
                )
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt >= attempts or spent >= max_total_delay:
                raise
            pause = _delay(attempt, base=base_delay, cap=max_delay)
            if log:
                log(
                    f"[llm-retry] {type(error).__name__}: {error}，"
                    f"{pause:.1f}s 后重试（第 {attempt}/{attempts} 次，已等待 {spent:.0f}s）"
                )
        pause = min(pause, max(0.0, max_total_delay - spent))
        spent += pause
        sleep(pause)
    # 只有 attempts <= 0 这种非法配置才会走到这里。
    if last_http_error is not None:
        return last_http_error
    raise RuntimeError("urlopen_with_backoff: no attempt was made")


def request_bytes_with_backoff(
    request: urllib.request.Request,
    *,
    timeout_s: float,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = 1.0,
    max_delay: float = DEFAULT_MAX_DELAY,
    max_total_delay: float = DEFAULT_MAX_TOTAL_DELAY,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> UpstreamResponse:
    """发请求**并读完响应体**，全过程可重试。

    与 ``urlopen_with_backoff`` 的差别就是这一点，而这一点是必需的：
    glm 的长响应经常在读体阶段被截断（``IncompleteRead``），而那时
    ``urlopen`` 早已成功返回、退避层已经退场。实测这会让一轮迭代直接失败。

    非可重试的 HTTP 错误（4xx 里的参数/凭据问题）会**照原样返回**给调用方
    （``status`` 是那个错误码、``body`` 是错误响应体），由调用方翻译给 CLI；
    绝不把它伪装成成功响应。
    """

    spent = 0.0
    last_error: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        pause: float | None = None
        try:
            response = urllib.request.urlopen(request, timeout=timeout_s)  # noqa: S310
            try:
                body = response.read()
                headers = {name: value for name, value in response.headers.items()}
                return UpstreamResponse(int(response.status), headers, body)
            finally:
                response.close()
        except urllib.error.HTTPError as error:
            last_error = error
            try:
                body = error.read()
                headers = {name: value for name, value in (error.headers or {}).items()}
            except Exception:  # noqa: BLE001 - 错误体读不到就用空体，状态码才是关键
                body, headers = b"", {}
            finally:
                error.close()
            if error.code not in RETRYABLE_STATUS:
                return UpstreamResponse(int(error.code), headers, body)
            if attempt >= attempts or spent >= max_total_delay:
                return UpstreamResponse(int(error.code), headers, body)
            pause = _retry_after_seconds(error)
            if pause is None:
                pause = _delay(attempt, base=base_delay, cap=max_delay)
            if log:
                log(
                    f"[llm-retry] HTTP {error.code}，{pause:.1f}s 后重试"
                    f"（第 {attempt}/{attempts} 次，已等待 {spent:.0f}s）"
                )
        except (*RETRYABLE_READ_ERRORS, urllib.error.URLError, OSError) as error:
            last_error = error
            if attempt >= attempts or spent >= max_total_delay:
                raise
            pause = _delay(attempt, base=base_delay, cap=max_delay)
            if log:
                log(
                    f"[llm-retry] {type(error).__name__}: {error}，"
                    f"{pause:.1f}s 后重试（第 {attempt}/{attempts} 次，已等待 {spent:.0f}s）"
                )
        pause = min(pause or 0.0, max(0.0, max_total_delay - spent))
        spent += pause
        sleep(pause)
    if last_error is not None:
        raise last_error
    raise RuntimeError("request_bytes_with_backoff: no attempt was made")

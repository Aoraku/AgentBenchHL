"""上游 LLM 请求退避重试的行为测试。

这层的价值全在"哪些该重试、哪些不该"：把 ``400``/``401`` 也重试会把配置错误
拖成 5 次慢失败并掩盖原因；而不重试 ``429`` 会让批量实验里的限流变成
"这一轮模型没产出候选"的假结论。
"""

from __future__ import annotations

import http.client
import urllib.error
import urllib.request

import pytest

from agentbench_hl.adapters.http_retry import (
    RETRYABLE_STATUS,
    request_bytes_with_backoff,
    urlopen_with_backoff,
)


class _FakeResponse:
    status = 200

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _http_error(code: int, *, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = {"Retry-After": retry_after} if retry_after else {}
    return urllib.error.HTTPError("http://x", code, "boom", headers, None)  # type: ignore[arg-type]


def _request() -> urllib.request.Request:
    return urllib.request.Request("http://127.0.0.1/v1/x", data=b"{}", method="POST")


def test_success_returns_immediately(monkeypatch) -> None:
    calls: list[int] = []
    response = _FakeResponse()

    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        calls.append(1)
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    assert (
        urlopen_with_backoff(_request(), timeout_s=1.0, sleep=slept.append) is response
    )
    assert len(calls) == 1
    assert slept == []


def test_rate_limit_is_retried_then_succeeds(monkeypatch) -> None:
    response = _FakeResponse()
    outcomes: list[object] = [_http_error(429), _http_error(503), response]

    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        item = outcomes.pop(0)
        if isinstance(item, urllib.error.HTTPError):
            raise item
        return item

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    assert (
        urlopen_with_backoff(_request(), timeout_s=1.0, sleep=slept.append) is response
    )
    assert len(slept) == 2
    # 指数退避：第二次等待必须比第一次长（抖动 0.75–1.25 倍不会翻转大小关系）。
    assert slept[1] > slept[0]


def test_retry_after_header_is_honoured(monkeypatch) -> None:
    response = _FakeResponse()
    outcomes: list[object] = [_http_error(429, retry_after="7"), response]

    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        item = outcomes.pop(0)
        if isinstance(item, urllib.error.HTTPError):
            raise item
        return item

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    urlopen_with_backoff(_request(), timeout_s=1.0, sleep=slept.append)
    assert slept == [7.0]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_client_errors_are_not_retried(monkeypatch, code: int) -> None:
    """配置类错误必须**立刻**返回：重试只会让 5 次慢失败掩盖真实原因。"""

    calls: list[int] = []

    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        calls.append(1)
        raise _http_error(code)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    result = urlopen_with_backoff(_request(), timeout_s=1.0, sleep=slept.append)
    assert isinstance(result, urllib.error.HTTPError) and result.code == code
    assert len(calls) == 1
    assert slept == []
    assert code not in RETRYABLE_STATUS


def test_returns_last_http_error_after_exhausting_attempts(monkeypatch) -> None:
    """重试耗尽后**保持原语义**把响应交回调用方，不自己编造成功响应。"""

    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        raise _http_error(503)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    result = urlopen_with_backoff(
        _request(), timeout_s=1.0, attempts=3, sleep=slept.append
    )
    assert isinstance(result, urllib.error.HTTPError) and result.code == 503
    assert len(slept) == 2


def test_connection_errors_are_retried_then_raised(monkeypatch) -> None:
    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    with pytest.raises(urllib.error.URLError):
        urlopen_with_backoff(_request(), timeout_s=1.0, attempts=3, sleep=slept.append)
    assert len(slept) == 2


def test_delay_is_capped_and_jittered(monkeypatch) -> None:
    """退避有上限且带抖动：多个 worker 同时被限流时不能继续同步撞墙。"""

    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        raise _http_error(429)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    urlopen_with_backoff(
        _request(), timeout_s=1.0, attempts=8, base_delay=1.0, max_delay=5.0, sleep=slept.append
    )
    assert slept, "应当发生过等待"
    assert max(slept) <= 5.0 * 1.25 + 1e-9
    assert len(set(slept)) > 1, "全部等待时长相同说明没有抖动"


def test_total_delay_budget_bounds_the_wait(monkeypatch) -> None:
    """总退避预算是硬上限：限流可以等，但不能无限等。

    次数上限之所以不够用：中转站的限流窗口是分钟级，5 次 × 30s 上限只有 25s
    总预算，实测 antwar2 就是这样在连续 429/503 里被判死的。改成按总预算驱动
    之后，"等得起的故障"就真的等得起，而失控的上游仍然会在预算处停下。
    """

    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        raise _http_error(429)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    urlopen_with_backoff(
        _request(),
        timeout_s=1.0,
        attempts=1000,
        base_delay=1.0,
        max_delay=10.0,
        max_total_delay=200.0,
        sleep=slept.append,
    )
    assert sum(slept) <= 200.0 + 1e-9
    # 而且必须真的比"5 次固定额度"等得更久，否则这次加固没有意义。
    assert len(slept) > 5
    # 预算必须**用尽**才停（允许最后一次被裁到刚好填满）。
    assert sum(slept) == pytest.approx(200.0, abs=1e-6)


# --------------------------------------------------------- request_bytes_*


class _BodyResponse:
    """能被读一次的假响应（可选在 read() 时抛错）。"""

    def __init__(
        self, status: int, body: bytes, *, error: BaseException | None = None
    ) -> None:
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self._body = body
        self._error = error
        self.closed = False

    def read(self) -> bytes:
        if self._error is not None:
            raise self._error
        return self._body

    def close(self) -> None:
        self.closed = True


def test_read_body_failure_is_retried(monkeypatch) -> None:
    """读体断流必须重试。

    这是 antwar 的真实死因：``urlopen`` 已经成功、退避层退场，然后
    ``response.read()`` 在 205 KB 处抛 ``IncompleteRead``，异常一路冒到
    HTTP 处理线程，整个 run 终止。只包住 urlopen 的重试对此完全无效。
    """

    truncated = _BodyResponse(200, b"", error=http.client.IncompleteRead(b"partial"))
    good = _BodyResponse(200, b'{"ok":true}')
    outcomes: list[_BodyResponse] = [truncated, good]

    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        return outcomes.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    result = request_bytes_with_backoff(_request(), timeout_s=1.0, sleep=slept.append)
    assert result.status == 200
    assert result.body == b'{"ok":true}'
    assert len(slept) == 1
    assert truncated.closed and good.closed, "两个响应都必须被关闭"


def test_read_body_returns_complete_response(monkeypatch) -> None:
    response = _BodyResponse(200, b'{"ok":1}')
    monkeypatch.setattr(urllib.request, "urlopen", lambda _r, timeout: response)  # noqa: ARG005
    slept: list[float] = []
    result = request_bytes_with_backoff(_request(), timeout_s=1.0, sleep=slept.append)
    assert (result.status, result.body) == (200, b'{"ok":1}')
    assert result.content_type == "application/json"
    assert slept == []


def test_read_body_client_error_is_returned_not_retried(monkeypatch) -> None:
    """4xx 配置错误照原样交回调用方，且不重试。"""

    calls: list[int] = []

    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        calls.append(1)
        raise urllib.error.HTTPError(
            "http://x", 401, "unauthorized", {"Content-Type": "application/json"}, None  # type: ignore[arg-type]
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    result = request_bytes_with_backoff(_request(), timeout_s=1.0, sleep=slept.append)
    assert result.status == 401
    assert len(calls) == 1
    assert slept == []


def test_read_body_raises_after_exhausting_attempts(monkeypatch) -> None:
    def fake_urlopen(_request, timeout):  # noqa: ANN001, ARG001
        return _BodyResponse(200, b"", error=http.client.IncompleteRead(b""))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    slept: list[float] = []
    with pytest.raises(http.client.IncompleteRead):
        request_bytes_with_backoff(
            _request(), timeout_s=1.0, attempts=3, sleep=slept.append
        )
    assert len(slept) == 2

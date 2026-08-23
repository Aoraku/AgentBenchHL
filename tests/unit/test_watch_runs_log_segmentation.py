"""监控脚本读日志时必须按"最后一次启动"分段。

为什么这件事值得一个测试
----------------------
``run_hl.sh`` 用 ``>>`` 追加写日志（续跑时上一段是排查依据，覆盖掉就没法
回溯"第 32 轮到底怎么收尾的"）。代价是同一个文件里混着多次启动的记录。

不分段的后果是**验收结论直接错掉**，而且错得很隐蔽：实测 verify-glm-5.3
用修好的配置跑通了第 1 轮，监控却仍报它有 remote-compact 失败 ——
那条记录来自修复前那次 run。看到报错的人会以为修复没生效，
然后继续排查一个已经不存在的问题。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from watch_runs import LAUNCH_MARKER, _last_launch_segment  # noqa: E402


def _launch(stamp: str) -> str:
    return f"\n{LAUNCH_MARKER} {stamp} pid-placeholder =====\n"


def test_only_the_last_launch_is_read() -> None:
    """上一次启动的失败不能算到这一次头上。"""

    log = (
        _launch("2026-08-23T18:00:00+08:00")
        + '{"error": "RuntimeError: remote compaction v2 ...", '
        '"iterations_completed": 0, "stop_reason": "iteration_failed"}\n'
        + _launch("2026-08-23T18:55:06+08:00")
        + '{"status": "slow_eval_started", "pid": 2148190}\n'
    )

    segment = _last_launch_segment(log)

    assert "remote compaction" not in segment, "上一次 run 的错误必须被排除"
    assert "slow_eval_started" in segment


def test_retry_counts_come_from_the_current_launch_only() -> None:
    """退避次数同理：混进上一次的会让"这一轮被限流多严重"完全失真。"""

    log = (
        _launch("t1")
        + "[llm-retry] 503\n" * 50
        + _launch("t2")
        + "[llm-retry] 503\n" * 3
    )

    assert _last_launch_segment(log).count("[llm-retry]") == 3


def test_a_single_launch_is_returned_whole() -> None:
    log = _launch("t1") + "line A\nline B\n"

    segment = _last_launch_segment(log)

    assert "line A" in segment and "line B" in segment


def test_logs_without_markers_fall_back_to_the_whole_file() -> None:
    """老日志（或不是 run_hl.sh 起的 run）没有标记时返回全文。

    宁可多报也不要漏报 —— 少报会让真实故障消失，而多报至少还会促使人去看。
    """

    log = '{"error": "something broke"}\n'

    assert _last_launch_segment(log) == log


def test_marker_survives_interleaved_output() -> None:
    """标记要能在大量输出里被认出来（它写在启动那一刻，后面全是 driver 输出）。"""

    log = (
        _launch("t1")
        + "noise\n" * 1000
        + _launch("t2")
        + "current run output\n"
    )

    segment = _last_launch_segment(log)

    assert segment.count(LAUNCH_MARKER) == 1
    assert "current run output" in segment
    assert segment.count("noise") == 0

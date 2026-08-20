"""IG 的支撑集口径必须**如实**上报——这是一次真实误判的回归测试。

发生了什么
----------
状态探针按回放记录数给出 246 个决策点，线协议记到 247 个（尾帧只出现在后者）。
原先的对齐条件是 ``len(probed) >= len(decisions)``，246 >= 247 为假，于是：

1. 整局逐点精确的 |A(s)| 被**全部丢弃**，退回 ``|A|=10`` 的字母表常量；
2. 回落原因写进了局部变量却没有进事件；
3. 汇总层 ``payload()`` 直接展开 ``decision_space.yaml`` 的**静态声明**，
   把每个 case 实际用的口径覆盖掉。

三件事叠起来的后果：连续 14 轮的 behavioral_ig 都是近似值，而事件、日志、
进度表异口同声地显示"一切正常"。KL 的尺度逐点由 |A| 决定（闭式解 ``u = ε/|A|``），
所以近似与精确不是同一个量纲的数——这类"看起来对"的错最贵。

因此锁三条：小缺口要救回来、大缺口仍须诚实回落、以及**实际口径必须盖在静态声明之上**。
"""

from __future__ import annotations

from agentbench_hl.application.behavioral_ig import (
    SUPPORT_ALIGN_ABS_TOLERANCE,
    BehavioralIgCase,
    BehavioralIgMeasurement,
    CaseMeasurement,
)

DECLARED = {"support_mode": "opcode_alphabet", "support_cardinality": 10}


def _case(mode: str, *, exact: int, compared: int, note: str | None = None) -> CaseMeasurement:
    return CaseMeasurement(
        BehavioralIgCase(opponent_id="rank10", role="P0", seed=1),
        mean_kl_nats=0.5,
        compared_decisions=compared,
        support_mode=mode,
        support_exact_decisions=exact,
        support_note=note,
    )


def test_observed_mode_overrides_the_static_declaration() -> None:
    """探针成功时，事件里的 support_mode 必须是 exact，而不是 yaml 里写的常量。"""

    measurement = BehavioralIgMeasurement(
        value=0.5,
        reason="x",
        support=DECLARED,
        cases=(_case("exact_enumeration", exact=246, compared=247),),
        compared_decisions=247,
    )
    payload = measurement.payload()
    assert payload["support_mode"] == "exact_enumeration"
    # 静态声明不能丢，但只能作为对照出现。
    assert payload["support_mode_declared"] == "opcode_alphabet"
    assert payload["support_exact_decisions"] == 246
    assert payload["support_exact_fraction"] == round(246 / 247, 4)


def test_mixed_cases_are_labelled_mixed_with_coverage() -> None:
    """部分精确就得说部分——"mixed"标签之外还要给出覆盖率。"""

    measurement = BehavioralIgMeasurement(
        value=0.5,
        reason="x",
        support=DECLARED,
        cases=(
            _case("exact_enumeration", exact=100, compared=100),
            _case("opcode_alphabet", exact=0, compared=100),
        ),
        compared_decisions=200,
    )
    payload = measurement.payload()
    assert payload["support_mode"] == "mixed"
    assert payload["support_exact_fraction"] == 0.5


def test_full_fallback_keeps_reporting_the_alphabet() -> None:
    measurement = BehavioralIgMeasurement(
        value=0.5,
        reason="x",
        support=DECLARED,
        cases=(_case("opcode_alphabet", exact=0, compared=50, note="probe failed"),),
        compared_decisions=50,
    )
    payload = measurement.payload()
    assert payload["support_mode"] == "opcode_alphabet"
    assert payload["support_exact_decisions"] == 0
    # 回落原因必须出现在事件里，否则又变成"静默降级"。
    assert payload["support_alignment_notes"] == ["probe failed"]


def test_unusable_cases_do_not_fabricate_a_mode() -> None:
    """没有可用 case 时不能编口径——那时 IG 本身就是 null。"""

    measurement = BehavioralIgMeasurement(
        value=None,
        reason="no case",
        support=DECLARED,
        cases=(CaseMeasurement(BehavioralIgCase("rank10", "P0", 1), reason="crashed"),),
    )
    payload = measurement.payload()
    assert payload["support_mode"] == "opcode_alphabet"  # 仅来自静态声明
    assert "support_exact_decisions" not in payload


def test_alignment_tolerance_is_small_and_explicit() -> None:
    """容差必须小：它的用途是吃掉"尾帧差一两个"，不是掩盖决策点定义不一致。"""

    assert 1 <= SUPPORT_ALIGN_ABS_TOLERANCE <= 3

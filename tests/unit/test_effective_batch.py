"""b（一轮打几个对手）的口径：账本必须记**实际值**，不是配置里写的值。

这组测试来自一个真实的账本失真
------------------------------
``exp2`` 主线用 ``opponent_policy: ladder_up`` 配着 ``batch`` 的默认值 4 在跑，
而 ``SequentialConquest.select()`` **无论 batch 写多少都只返回 1 个对手**。
于是配置说 4、实际打 1，而 b 是一等实验变量：

* ``run-manifest.json`` 里以前**根本没有 batch** —— 读账本的人无从校验对局数；
* ``RunReproducibilityManifest`` 也没有它 —— "统计可复现"少了一个自变量；
* 提示词会告诉 agent "这一版会被拿去打 4 个对手，你能从 4 份不同的回放里拿到
  证据"，而它只会拿到 1 份 —— 这是在指导里撒谎；
* ``watch_runs.py`` 的对手数告警用 b 当阈值，虚高的 b 让告警失灵；
* "一轮对局数 = k × b × 座次"算出来是实际的 4 倍。
"""

from __future__ import annotations

import pytest

from agentbench_hl.application.opponent_policy import (
    LadderEntry,
    build_policy,
    effective_batch_for,
)

#: 每轮只打一个对手的策略（batch 写多少都一样）。
SINGLE = ("ladder_up", "ladder_down", "fixed_rank")
#: 真正会铺开 b 个对手的策略。
MULTI = ("progress", "random", "fix", "k_diverse")


def _ladder(size: int = 30) -> tuple[LadderEntry, ...]:
    return tuple(
        LadderEntry(opponent_id=f"p{index}", rank=index, score=2000 - index * 10)
        for index in range(1, size + 1)
    )


@pytest.mark.parametrize("name", SINGLE)
def test_single_target_policies_report_one(name: str) -> None:
    policy = build_policy(name, _ladder(), target_rank=5, start_rank=10)
    assert policy.effective_batch(4) == 1
    assert effective_batch_for(name, 4) == 1


@pytest.mark.parametrize("name", MULTI)
def test_multi_target_policies_report_the_request(name: str) -> None:
    policy = build_policy(name, _ladder(), target_rank=5, start_rank=10)
    assert policy.effective_batch(4) == 4
    assert effective_batch_for(name, 4) == 4


@pytest.mark.parametrize("name", SINGLE + MULTI)
def test_reported_batch_matches_what_select_actually_returns(name: str) -> None:
    """核心断言：报出来的 b 必须等于 ``select()`` 真的给出的对手数。

    这是唯一能防止账本再次失真的不变式 —— 两者一旦分叉，
    "一轮打了几局"这个问题就只能靠读源码回答。
    """

    policy = build_policy(name, _ladder(), target_rank=5, start_rank=10)
    reported = policy.effective_batch(4)
    chosen = policy.select(iteration=1, batch=reported)
    assert len(chosen) == reported


def test_self_policy_defers_to_the_agent() -> None:
    """``self`` 由 agent 自己挑，框架只给上限，所以 select 返回空是**正确**的。

    单独列出来是因为它会破坏上面那条不变式，而那是设计如此，不是 bug。
    """

    policy = build_policy("self", _ladder())
    assert policy.effective_batch(4) == 4
    assert policy.select(iteration=1, batch=4) == ()


@pytest.mark.parametrize("name", SINGLE + MULTI)
def test_two_code_paths_agree(name: str) -> None:
    """策略实例上的方法与只知道名字的模块函数必须同口径。

    两条路径都存在是因为 ``run-manifest.json`` 在装配策略实例**之前**就要落盘
    （见 adapters/contract/factory.py），那时手里只有策略名字。
    """

    policy = build_policy(name, _ladder(), target_rank=5, start_rank=10)
    for requested in (1, 2, 4, 8):
        assert policy.effective_batch(requested) == effective_batch_for(name, requested)


def test_batch_is_clamped_to_the_ladder_size() -> None:
    """榜单比 b 小的时候不能报一个打不出来的数。"""

    policy = build_policy("progress", _ladder(size=2), start_rank=10)
    assert policy.effective_batch(4) == 2

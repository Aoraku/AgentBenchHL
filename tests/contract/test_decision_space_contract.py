"""B 侧真的能从 A 仓读到行为信息增益的口径。

这条测试守的是"三仓不各说各话"：schema 只在 A 定义一份，B 通过
:mod:`agentbench_hl.application.decision_space` 读它。如果哪天有人给 A 的
``information_gain:`` 段改了字段名或删了 |A|，这里会红。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentbench_hl.application.decision_space import (
    decision_space_path,
    load_information_gain_spec,
)

MEASURED_GAMES = (
    "antwar",
    "antwar2",
    "aquawar",
    "generals",
    "lostspace",
    "miracle",
    "rollman",
    "snakego",
)


def _agentbench_root() -> Path | None:
    value = os.environ.get("AGENTBENCH_ROOT")
    if value and (Path(value) / "games").is_dir():
        return Path(value)
    # 三仓并列布局下的默认位置（本地开发与服务器都成立）。
    sibling = Path(__file__).resolve().parents[2].parent / "AgentBench"
    return sibling if (sibling / "games").is_dir() else None


ROOT = _agentbench_root()
pytestmark = pytest.mark.skipif(
    ROOT is None, reason="AgentBench 仓不可见（AGENTBENCH_ROOT 未设置且无并列目录）"
)


@pytest.mark.parametrize("game", MEASURED_GAMES)
def test_every_game_exposes_a_declared_support(game: str) -> None:
    spec, note = load_information_gain_spec(game, agentbench_root=ROOT)

    assert spec is not None, note
    assert spec.probe == "transcript_replay"
    roles = sorted(spec.support.cardinality_by_role) or [None]
    for role in roles:
        assert spec.support.size_for(role) >= 2
    described = spec.describe(roles[0])
    # 口径三件套必须齐：没有出处的 |A| 不允许出现在曲线上。
    assert described["support_mode"] in ("enumerated", "opcode_alphabet")
    assert described["support_cardinality"] >= 2
    assert described["support_provenance"]


def test_rollman_support_is_per_role() -> None:
    spec, _ = load_information_gain_spec("rollman", agentbench_root=ROOT)

    assert spec is not None
    assert spec.support.size_for("rollman") != spec.support.size_for("ghost")


def test_free_text_game_is_reported_as_absent() -> None:
    spec, note = load_information_gain_spec("deepclue", agentbench_root=ROOT)

    assert spec is None
    assert "declares no information_gain contract" in note


def test_unknown_game_is_reported_not_crashing() -> None:
    spec, note = load_information_gain_spec("no_such_game", agentbench_root=ROOT)

    assert spec is None
    assert "no decision_space.yaml" in note
    assert decision_space_path("no_such_game", agentbench_root=ROOT) is None

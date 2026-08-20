from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest

from agentbench_hl.adapters.antwar2.policy_probe import probe_policy_episode
from agentbench_hl.adapters.antwar2.runtime import AntWar2Layout

REPLAY = Path(__file__).parents[1] / "golden/antwar2_replays/fixture.json"

# `gamepacks/antwar2/candidate_support` 里**已经**vendor 了官方 `SDK/`
# （见 scripts/gen_candidate_support.py）。所以搭候选目录只需要拷这一份：
# 再单独拷一次 `layout.public_sdk_root` → `candidate/SDK` 会直接
# FileExistsError。这也正是容器里的真实布局，测试应当与它一致。
CANDIDATE_SUPPORT = Path(__file__).parents[2] / "gamepacks/antwar2/candidate_support"

# ⚠️ `BaseAgent` 的唯一抽象方法是 `choose_bundle`（官方
# `games/antwar2/public_sdk/common.py`）。只覆盖 `choose_operations` 的类无法实例化。
MINIMAL_AI = """from common import BaseAgent

class AI(BaseAgent):
    def choose_bundle(self, state, player, bundles=None):
        bundles = bundles or self.list_bundles(state, player)
        return bundles[0]
"""


def test_policy_probe_reconstructs_public_decision_states_in_a_subprocess(
    tmp_path: Path,
) -> None:
    agentbench = os.environ.get("AGENTBENCH_ROOT")
    if not agentbench:
        pytest.skip("AGENTBENCH_ROOT is required for the frozen SDK probe")
    layout = AntWar2Layout.from_root(agentbench, tmp_path / "build")
    layout.validate()
    candidate = tmp_path / "candidate"
    shutil.copytree(CANDIDATE_SUPPORT, candidate)
    (candidate / "ai.py").write_text(MINIMAL_AI, encoding="utf-8")

    trace = probe_policy_episode(
        candidate,
        REPLAY,
        match_id="fixture-episode",
        role="P0",
    )

    assert trace.match_id == "fixture-episode"
    assert trace.role == "P0"
    assert len(trace.decisions) == 3
    assert tuple(item.state_id for item in trace.decisions) == (
        "fixture-episode:r0001:p0",
        "fixture-episode:r0028:p0",
        "fixture-episode:r0029:p0",
    )
    assert "HOLD" in trace.decisions[0].legal_supports[0]
    assert len(trace.decisions[0].occupancy_id) == 64


def test_policy_probe_runs_through_the_candidate_command_prefix(tmp_path: Path) -> None:
    agentbench = os.environ.get("AGENTBENCH_ROOT")
    if not agentbench:
        pytest.skip("AGENTBENCH_ROOT is required for the frozen SDK probe")
    layout = AntWar2Layout.from_root(agentbench, tmp_path / "build")
    layout.validate()
    candidate = tmp_path / "candidate"
    shutil.copytree(CANDIDATE_SUPPORT, candidate)
    (candidate / "ai.py").write_text(MINIMAL_AI, encoding="utf-8")
    marker = tmp_path / "prefix-used"
    wrapper = tmp_path / "candidate-wrapper"
    wrapper.write_text(
        f'#!/bin/sh\n/usr/bin/touch {marker}\nexec "$@"\n',
        encoding="utf-8",
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)

    probe_policy_episode(
        candidate,
        REPLAY,
        match_id="prefixed-episode",
        role="P0",
        command_prefix=(str(wrapper),),
    )

    assert marker.is_file()

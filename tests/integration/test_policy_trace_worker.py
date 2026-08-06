from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

import pytest

from agentbench_hl.adapters.antwar2.policy_probe import probe_policy_episode
from agentbench_hl.adapters.antwar2.runtime import AntWarLayout

REPLAY = Path(__file__).parents[1] / "golden/antwar2_replays/fixture.json"


def test_policy_probe_reconstructs_public_decision_states_in_a_subprocess(
    tmp_path: Path,
) -> None:
    agentbench = os.environ.get("AGENTBENCH_ROOT")
    if not agentbench:
        pytest.skip("AGENTBENCH_ROOT is required for the frozen SDK probe")
    layout = AntWarLayout.from_root(agentbench, tmp_path / "build")
    layout.validate()
    repository = Path(__file__).parents[2]
    candidate = tmp_path / "candidate"
    shutil.copytree(repository / "gamepacks/antwar2/candidate_support", candidate)
    shutil.copytree(layout.public_sdk_root, candidate / "SDK")
    (candidate / "ai.py").write_text(
        """from common import BaseAgent

class AI(BaseAgent):
    def choose_operations(self, state, player, bundles=None):
        return []
""",
        encoding="utf-8",
    )

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
    assert trace.decisions[0].actions == ()
    assert "HOLD" in trace.decisions[0].legal_supports[0]
    assert len(trace.decisions[0].occupancy_id) == 64


def test_policy_probe_runs_through_the_candidate_command_prefix(tmp_path: Path) -> None:
    agentbench = os.environ.get("AGENTBENCH_ROOT")
    if not agentbench:
        pytest.skip("AGENTBENCH_ROOT is required for the frozen SDK probe")
    layout = AntWarLayout.from_root(agentbench, tmp_path / "build")
    layout.validate()
    repository = Path(__file__).parents[2]
    candidate = tmp_path / "candidate"
    shutil.copytree(repository / "gamepacks/antwar2/candidate_support", candidate)
    shutil.copytree(layout.public_sdk_root, candidate / "SDK")
    (candidate / "ai.py").write_text(
        "from common import BaseAgent\n\n"
        "class AI(BaseAgent):\n"
        "    def choose_operations(self, state, player, bundles=None):\n"
        "        return []\n",
        encoding="utf-8",
    )
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

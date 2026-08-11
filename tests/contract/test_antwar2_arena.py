from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from agentbench_hl.adapters.antwar2.arena import (
    AntWar2Arena,
    match_result_from_replay,
)
from agentbench_hl.adapters.antwar2.runtime import (
    AntWar2Layout,
    audit_human_pool,
    build_backend,
    materialize_bootstrap,
)
from agentbench_hl.adapters.antwar2.smoke import verify_smoke
from agentbench_hl.adapters.codex_goal.read_isolation import (
    write_candidate_isolation_profile,
)
from agentbench_hl.ports.arena import MatchCase, ProcessSpec

AGENTBENCH_ROOT = Path("/Users/qingle/Code/SAST/AgentBench")
EXISTING_BUILD_ROOT = Path(
    "/Users/qingle/Documents/SAST/AgentBenchFramework-hl/.agentbench/30_antwar2/build"
)
EXPECTED_SDK_HASH = "f54280bb6b4407609ee7cc0b4df405b37c8009d5049dbaa642f12b5648a9efe0"


def write_terminal_replay(path: Path, *, winner: int, bases: list[int]) -> Path:
    replay = path / "replay.json"
    replay.write_text(
        json.dumps(
            [
                {
                    "op0": [],
                    "op1": [],
                    "round_state": {
                        "camps": [50, 50],
                        "winner": -1,
                    },
                },
                {
                    "op0": [],
                    "op1": [],
                    "round_state": {
                        "camps": bases,
                        "winner": winner,
                    },
                },
            ]
        ),
        encoding="utf-8",
    )
    return replay


def test_terminal_replay_becomes_role_aware_match_result(tmp_path: Path) -> None:
    replay = write_terminal_replay(tmp_path, winner=1, bases=[0, 7])
    result = match_result_from_replay(
        replay,
        candidate_id="v000",
        opponent_id="rank20",
        role="P1",
        seed=1,
    )

    assert result.status == "complete"
    assert result.result == "win"
    assert result.score_margin == 7.0
    assert result.payload["terminal_base_hp"] == (0.0, 7.0)


def test_human_pool_audit_is_ranked_and_records_unrunnable_entries(tmp_path: Path) -> None:
    layout = AntWar2Layout.from_root(AGENTBENCH_ROOT, tmp_path / "build")
    pool = audit_human_pool(layout)

    assert [item.rank for item in pool] == list(range(1, 21))
    assert pool[-1].opponent_id == "rank20"
    assert len([item for item in pool if item.runnable]) == 18
    assert {item.opponent_id for item in pool if not item.runnable} == {
        "rank03",
        "rank09",
    }
    assert all(len(item.archive_sha256) == 64 for item in pool)


def test_public_sdk_and_existing_official_backend_match_frozen_hashes() -> None:
    if not AGENTBENCH_ROOT.is_dir() or not EXISTING_BUILD_ROOT.is_dir():
        pytest.skip("local frozen AntWar2 resources are unavailable")
    layout = AntWar2Layout.from_root(AGENTBENCH_ROOT, EXISTING_BUILD_ROOT)

    layout.validate(expected_sdk_sha256=EXPECTED_SDK_HASH)
    backend = build_backend(layout)

    assert backend.archive_sha256 == (
        "01add42ce4bb545678fa953e4da4c049cc09f512131035739592d685a2dda22b"
    )
    assert len(backend.executable_sha256) == 64


def test_smoke_rejects_illegal_operation(tmp_path: Path) -> None:
    if not AGENTBENCH_ROOT.is_dir():
        pytest.skip("local frozen AntWar2 resources are unavailable")
    layout = AntWar2Layout.from_root(AGENTBENCH_ROOT, tmp_path / "build")
    support = Path(__file__).parents[2] / "gamepacks/antwar2/candidate_support"
    candidate = tmp_path / "candidate"
    materialize_bootstrap(layout, support, candidate)
    (candidate / "ai.py").write_text(
        """from common import BaseAgent
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType

class AI(BaseAgent):
    def choose_operations(self, state, player, bundles=None):
        return [Operation(OperationType.BUILD_TOWER, -99, -99)]
""",
        encoding="utf-8",
    )

    result = verify_smoke(candidate)

    assert result.status == "failed"
    assert "illegal operation" in (result.error or "")


def test_smoke_runs_through_the_candidate_command_prefix(tmp_path: Path) -> None:
    if not AGENTBENCH_ROOT.is_dir():
        pytest.skip("local frozen AntWar2 resources are unavailable")
    layout = AntWar2Layout.from_root(AGENTBENCH_ROOT, tmp_path / "build")
    support = Path(__file__).parents[2] / "gamepacks/antwar2/candidate_support"
    candidate = tmp_path / "candidate"
    materialize_bootstrap(layout, support, candidate)
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

    result = verify_smoke(candidate, command_prefix=(str(wrapper),))

    assert result.status == "complete"
    assert marker.is_file()


def test_match_failure_is_incomplete_not_loss(tmp_path: Path) -> None:
    seen = {}

    def timeout_runner(**_kwargs):
        seen.update(_kwargs)
        raise TimeoutError("fixture timeout")

    opponent_root = tmp_path / "human"
    opponent_root.mkdir()
    (opponent_root / "main.py").write_text("", encoding="utf-8")
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    (candidate_root / "main.py").write_text("", encoding="utf-8")
    arena = AntWar2Arena(
        game=ProcessSpec(("game",), tmp_path),
        opponents={
            "rank20": ProcessSpec(("python", "main.py"), opponent_root),
        },
        artifact_root=tmp_path / "matches",
        candidate_command_prefix=("candidate-sandbox", "--profile", "fixture.sb"),
        runner=timeout_runner,
    )

    result = arena.run_case(
        MatchCase("v000", "rank20", "P0", 1),
        candidate_root=candidate_root,
    )

    assert result.status == "incomplete"
    assert result.result is None
    assert "timeout" in result.error
    assert seen["candidate_process"].argv[:3] == (
        "candidate-sandbox",
        "--profile",
        "fixture.sb",
    )


@pytest.mark.live
def test_official_match_transport_completes_against_rank20(tmp_path: Path) -> None:
    if os.environ.get("ABHL_RUN_OFFICIAL_MATCH_TEST") != "1":
        pytest.skip("set ABHL_RUN_OFFICIAL_MATCH_TEST=1 to run native match")
    layout = AntWar2Layout.from_root(AGENTBENCH_ROOT, EXISTING_BUILD_ROOT)
    backend = build_backend(layout)
    pool = audit_human_pool(layout)
    rank20 = next(item for item in pool if item.opponent_id == "rank20")
    assert rank20.entry_command is not None
    candidate = tmp_path / "candidate"
    support = Path(__file__).parents[2] / "gamepacks/antwar2/candidate_support"
    materialize_bootstrap(layout, support, candidate)
    (candidate / "ai.py").write_text(
        """from common import BaseAgent

class AI(BaseAgent):
    def choose_operations(self, state, player, bundles=None):
        return []
""",
        encoding="utf-8",
    )
    profile = write_candidate_isolation_profile(
        tmp_path / "candidate.sb",
        denied_read_roots=(layout.human_manifest.parent,),
    )
    arena = AntWar2Arena(
        game=ProcessSpec((str(backend.executable),), backend.executable.parent.parent),
        opponents={
            "rank20": ProcessSpec(rank20.entry_command, rank20.package_root),
        },
        artifact_root=tmp_path / "matches",
        timeout_s=120,
        candidate_command_prefix=(
            "/usr/bin/sandbox-exec",
            "-f",
            str(profile),
        ),
    )

    result = arena.run_case(
        MatchCase("v000", "rank20", "P0", 1),
        candidate_root=candidate,
    )

    assert result.status == "complete", result.error
    assert result.result in {"win", "loss"}
    assert result.replay_path is not None and result.replay_path.is_file()

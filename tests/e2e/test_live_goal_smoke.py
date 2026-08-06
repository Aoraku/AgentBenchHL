from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from agentbench_hl.application.live_run import build_live_run


@pytest.mark.live
def test_live_goal_reaches_first_official_match() -> None:
    if os.environ.get("ABHL_LIVE") != "1":
        pytest.skip("set ABHL_LIVE=1 for the paid Goal smoke")
    if not os.environ.get("ABHL_API_KEY"):
        pytest.skip("ABHL_API_KEY is required")
    if not os.environ.get("AGENTBENCH_ROOT"):
        pytest.skip("AGENTBENCH_ROOT is required")
    repository = Path(__file__).parents[2]
    run = build_live_run(
        repository / "configs/experiments/antwar2-goal-k1.yaml",
        run_id=f"antwar2-goal-smoke-{uuid.uuid4().hex[:10]}",
    )
    try:
        result = run.execute_until("first_match_finalized")
    finally:
        close = getattr(run.runtime, "close", None)
        if callable(close):
            close()

    assert result.lineage.versions["v000"].parent_id is None
    assert result.event_count("MatchFinalized") == 1
    assert result.event_count("ReplayDecoded") == 1
    assert result.event_count("ExperienceRecorded") == 1
    assert result.metrics.research_iteration == 0
    assert (result.root / "checkpoint.json").is_file()

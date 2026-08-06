from __future__ import annotations

from pathlib import Path

import pytest

from agentbench_hl.application.run_lease import RunLease, RunLeaseBusy


def test_run_lease_rejects_a_second_concurrent_owner(tmp_path: Path) -> None:
    first = RunLease(tmp_path)
    second = RunLease(tmp_path)

    with first:
        with pytest.raises(RunLeaseBusy, match="already active"):
            second.acquire()

    with second:
        assert (tmp_path / "run.lock").is_file()

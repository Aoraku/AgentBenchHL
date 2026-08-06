from __future__ import annotations

from pathlib import Path

import pytest

from agentbench_hl.domain.lineage import CandidateVersion, LineageState


def version(
    version_id: str,
    parent: str | None,
    *,
    content_hash: str | None = None,
    duplicate_of: str | None = None,
) -> CandidateVersion:
    digest = content_hash or (version_id.removeprefix("v") or "0").zfill(64)
    return CandidateVersion(
        version_id=version_id,
        parent_id=parent,
        workspace_id=f"workspace-{version_id}",
        content_hash=digest,
        object_path=Path("objects") / digest,
        source_hashes={"ai.py": digest},
        reason="unit test",
        duplicate_of=duplicate_of,
    )


def test_weak_frontier_does_not_replace_champion() -> None:
    state = LineageState.empty().add_version(version("v000", parent=None))
    state = state.promote("v000")
    state = state.add_version(version("v001", parent="v000"))
    state = state.choose_frontier("v001", rationale="diagnostic improved")

    assert state.champion_id == "v000"
    assert state.frontier_id == "v001"
    assert state.archive_ids == frozenset({"v000", "v001"})


def test_soft_exploration_debt_never_auto_rolls_back() -> None:
    state = LineageState.empty().add_version(version("v000", parent=None))
    state = state.promote("v000")
    parent = "v000"
    for index in range(1, 5):
        child = f"v{index:03d}"
        state = state.add_version(version(child, parent=parent))
        state = state.choose_frontier(child, rationale=f"explore branch {index}")
        parent = child

    assert state.champion_id == "v000"
    assert state.frontier_id == "v004"
    assert state.exploration_debt == 4
    assert state.requires_continuation_rationale is True


def test_duplicate_cannot_be_selected_as_frontier() -> None:
    state = LineageState.empty().add_version(version("v000", parent=None))
    state = state.promote("v000")
    state = state.add_version(
        version(
            "v001",
            parent="v000",
            content_hash=state.versions["v000"].content_hash,
            duplicate_of="v000",
        )
    )

    with pytest.raises(ValueError, match="duplicate"):
        state.choose_frontier("v001", rationale="must be rejected")


def test_lineage_requires_existing_parent_and_nonempty_rationale() -> None:
    state = LineageState.empty()
    with pytest.raises(ValueError, match="parent"):
        state.add_version(version("v001", parent="v000"))

    state = state.add_version(version("v000", parent=None))
    with pytest.raises(ValueError, match="rationale"):
        state.choose_frontier("v000", rationale="  ")

from __future__ import annotations

from pathlib import Path

import pytest

from agentbench_hl.adapters.filesystem.artifact_store import FilesystemArtifactStore
from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.application.candidate_service import CandidateService


def make_bootstrap(root: Path) -> Path:
    bootstrap = root / "bootstrap"
    sdk = bootstrap / "SDK"
    sdk.mkdir(parents=True, exist_ok=True)
    (sdk / "__init__.py").write_text("# frozen SDK\n", encoding="utf-8")
    for name in ("main.py", "common.py", "protocol.py"):
        (bootstrap / name).write_text(f"# {name}\n", encoding="utf-8")
    return bootstrap


def build_service(root: Path) -> CandidateService:
    return CandidateService(
        run_root=root / "run",
        bootstrap_root=make_bootstrap(root),
        artifact_store=FilesystemArtifactStore(root / "run" / "candidates"),
        event_store=JsonlEventStore(root / "run" / "events.jsonl"),
    )


def write_policy(path: Path, body: str = "class AI:\n    pass\n") -> None:
    (path / "ai.py").write_text(body, encoding="utf-8")


def test_seal_is_immutable_and_resume_restores_lineage(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    workspace = service.create(parent_id=None, reason="generate v000 from rules")
    assert not (workspace.path / "ai.py").exists()

    write_policy(workspace.path)
    v000 = service.seal(workspace.workspace_id)
    service.promote(v000.version_id)
    sealed_policy = v000.object_path / "ai.py"
    original = sealed_policy.read_text(encoding="utf-8")

    write_policy(workspace.path, "class AI:\n    changed = True\n")
    assert sealed_policy.read_text(encoding="utf-8") == original

    resumed = build_service(tmp_path)
    assert resumed.state == service.state
    assert resumed.state.champion_id == "v000"
    assert resumed.state.frontier_id == "v000"


def test_same_parent_and_content_creates_archived_duplicate(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    first_workspace = service.create(parent_id=None, reason="first")
    write_policy(first_workspace.path)
    first = service.seal(first_workspace.workspace_id)

    second_workspace = service.create(parent_id=None, reason="repeat")
    write_policy(second_workspace.path)
    second = service.seal(second_workspace.workspace_id)

    assert second.version_id == "v001"
    assert second.content_hash == first.content_hash
    assert second.duplicate_of == first.version_id
    assert service.state.archive_ids == frozenset({"v000", "v001"})
    with pytest.raises(ValueError, match="duplicate"):
        service.choose_frontier(second.version_id, rationale="not eligible")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda workspace: (workspace / "ai.py").unlink(), "missing"),
        (
            lambda workspace: (workspace / "notes.txt").write_text(
                "token=sk-examplecredential123", encoding="utf-8"
            ),
            "credential",
        ),
        (
            lambda workspace: (workspace / "escape").symlink_to(workspace / "ai.py"),
            "symlink",
        ),
    ],
)
def test_seal_rejects_unsafe_or_incomplete_workspace(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    service = build_service(tmp_path)
    workspace = service.create(parent_id=None, reason="invalid candidate")
    write_policy(workspace.path)
    mutation(workspace.path)

    with pytest.raises(ValueError, match=message):
        service.seal(workspace.workspace_id)

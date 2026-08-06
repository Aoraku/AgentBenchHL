"""Candidate workspace, sealing, and lineage orchestration."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.lineage import CandidateVersion, CandidateWorkspace, LineageState
from agentbench_hl.ports.artifact_store import ArtifactStore
from agentbench_hl.ports.event_store import EventStore


class CandidateService:
    def __init__(
        self,
        *,
        run_root: str | Path,
        bootstrap_root: str | Path,
        artifact_store: ArtifactStore,
        event_store: EventStore,
        soft_non_improving_depth: int = 3,
    ) -> None:
        self.run_root = Path(run_root)
        self.bootstrap_root = Path(bootstrap_root)
        self.artifact_store = artifact_store
        self.event_store = event_store
        self.soft_non_improving_depth = soft_non_improving_depth
        self.state = LineageState.replay(
            event_store.read_all(),
            soft_non_improving_depth=soft_non_improving_depth,
        )

    def create(self, parent_id: str | None, reason: str) -> CandidateWorkspace:
        if not reason.strip():
            raise ValueError("candidate creation reason must be non-empty")
        if parent_id is None:
            source = self.bootstrap_root
        else:
            try:
                source = self.state.versions[parent_id].object_path
            except KeyError as exc:
                raise ValueError(f"unknown candidate parent: {parent_id}") from exc
        if not source.is_dir():
            raise ValueError(f"candidate source is unavailable: {source}")

        workspace_id = f"w-{uuid.uuid4().hex}"
        path = self.run_root / "workspaces" / workspace_id
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, path)
        metadata_root = path / ".agentbench"
        metadata_root.mkdir()
        (metadata_root / "workspace.json").write_text(
            json.dumps(
                {"workspace_id": workspace_id, "parent_id": parent_id, "reason": reason},
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return CandidateWorkspace(workspace_id, path, parent_id, reason)

    def seal(self, workspace_id: str) -> CandidateVersion:
        workspace = self._load_workspace(workspace_id)
        content_hash, object_path, source_hashes = self.artifact_store.materialize(workspace.path)
        version_id = f"v{len(self.state.versions):03d}"
        duplicate_of = next(
            (
                item.version_id
                for item in self.state.versions.values()
                if item.parent_id == workspace.parent_id and item.content_hash == content_hash
            ),
            None,
        )
        version = CandidateVersion(
            version_id=version_id,
            parent_id=workspace.parent_id,
            workspace_id=workspace.workspace_id,
            content_hash=content_hash,
            object_path=object_path,
            source_hashes=source_hashes,
            reason=workspace.reason,
            duplicate_of=duplicate_of,
        )
        event = FinalizedEvent.create(
            "CandidateSealed",
            version.to_payload(),
            idempotency_key=f"candidate-sealed:{version_id}",
        )
        self.event_store.append(event)
        self.state = self.state.add_version(version)
        return version

    def choose_frontier(self, version_id: str, rationale: str) -> None:
        new_state = self.state.choose_frontier(version_id, rationale)
        event = FinalizedEvent.create(
            "FrontierSelected",
            {"version_id": version_id, "rationale": rationale},
            idempotency_key=f"frontier-selected:{version_id}:{self.state.exploration_debt + 1}",
        )
        self.event_store.append(event)
        self.state = new_state

    def promote(self, version_id: str) -> None:
        new_state = self.state.promote(version_id)
        if (
            self.state.champion_id == version_id
            and self.state.frontier_id == version_id
            and self.state.exploration_debt == 0
        ):
            return
        event = FinalizedEvent.create(
            "CandidatePromoted",
            {"version_id": version_id},
            idempotency_key=f"candidate-promoted:{version_id}",
        )
        self.event_store.append(event)
        self.state = new_state

    def _load_workspace(self, workspace_id: str) -> CandidateWorkspace:
        path = self.run_root / "workspaces" / workspace_id
        metadata_path = path / ".agentbench" / "workspace.json"
        if not metadata_path.is_file():
            raise ValueError(f"unknown candidate workspace: {workspace_id}")
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
        if value.get("workspace_id") != workspace_id:
            raise ValueError("candidate workspace identity mismatch")
        parent = value.get("parent_id")
        return CandidateWorkspace(
            workspace_id=workspace_id,
            path=path,
            parent_id=None if parent is None else str(parent),
            reason=str(value["reason"]),
        )

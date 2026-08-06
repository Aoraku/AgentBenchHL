"""Candidate lineage and Champion/Frontier/Archive state."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from agentbench_hl.domain.events import FinalizedEvent


@dataclass(frozen=True)
class CandidateWorkspace:
    workspace_id: str
    path: Path
    parent_id: str | None
    reason: str


@dataclass(frozen=True)
class CandidateVersion:
    version_id: str
    parent_id: str | None
    workspace_id: str
    content_hash: str
    object_path: Path
    source_hashes: Mapping[str, str]
    reason: str
    origin: str = "from_scratch"
    duplicate_of: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(self.source_hashes)))

    def to_payload(self) -> dict[str, object]:
        return {
            "version_id": self.version_id,
            "parent_id": self.parent_id,
            "workspace_id": self.workspace_id,
            "content_hash": self.content_hash,
            "object_path": str(self.object_path),
            "source_hashes": dict(self.source_hashes),
            "reason": self.reason,
            "origin": self.origin,
            "duplicate_of": self.duplicate_of,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CandidateVersion:
        source_hashes = payload.get("source_hashes")
        if not isinstance(source_hashes, Mapping):
            raise ValueError("candidate source_hashes must be a mapping")
        parent = payload.get("parent_id")
        duplicate = payload.get("duplicate_of")
        return cls(
            version_id=str(payload["version_id"]),
            parent_id=None if parent is None else str(parent),
            workspace_id=str(payload["workspace_id"]),
            content_hash=str(payload["content_hash"]),
            object_path=Path(str(payload["object_path"])),
            source_hashes={str(key): str(value) for key, value in source_hashes.items()},
            reason=str(payload["reason"]),
            origin=str(payload.get("origin", "from_scratch")),
            duplicate_of=None if duplicate is None else str(duplicate),
        )


@dataclass(frozen=True)
class LineageState:
    versions: Mapping[str, CandidateVersion] = field(default_factory=dict)
    champion_id: str | None = None
    frontier_id: str | None = None
    exploration_debt: int = 0
    soft_non_improving_depth: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "versions", MappingProxyType(dict(self.versions)))

    @classmethod
    def empty(cls, soft_non_improving_depth: int = 3) -> LineageState:
        return cls(soft_non_improving_depth=soft_non_improving_depth)

    @property
    def archive_ids(self) -> frozenset[str]:
        return frozenset(self.versions)

    @property
    def requires_continuation_rationale(self) -> bool:
        return self.exploration_debt > self.soft_non_improving_depth

    def add_version(self, version: CandidateVersion) -> LineageState:
        if version.version_id in self.versions:
            if self.versions[version.version_id] == version:
                return self
            raise ValueError(f"candidate version already exists: {version.version_id}")
        if version.parent_id is not None and version.parent_id not in self.versions:
            raise ValueError(f"candidate parent does not exist: {version.parent_id}")
        if version.duplicate_of is not None and version.duplicate_of not in self.versions:
            raise ValueError(f"duplicate target does not exist: {version.duplicate_of}")
        updated = dict(self.versions)
        updated[version.version_id] = version
        return replace(self, versions=updated)

    def choose_frontier(self, version_id: str, rationale: str) -> LineageState:
        candidate = self._require_version(version_id)
        if not rationale.strip():
            raise ValueError("frontier rationale must be non-empty")
        if candidate.duplicate_of is not None:
            raise ValueError("duplicate candidate cannot become Frontier")
        debt = 0 if version_id == self.champion_id else self.exploration_debt + 1
        return replace(self, frontier_id=version_id, exploration_debt=debt)

    def promote(self, version_id: str) -> LineageState:
        candidate = self._require_version(version_id)
        if candidate.duplicate_of is not None:
            raise ValueError("duplicate candidate cannot become Champion")
        return replace(
            self,
            champion_id=version_id,
            frontier_id=version_id,
            exploration_debt=0,
        )

    def _require_version(self, version_id: str) -> CandidateVersion:
        try:
            return self.versions[version_id]
        except KeyError as exc:
            raise ValueError(f"unknown candidate version: {version_id}") from exc

    @classmethod
    def replay(
        cls,
        events: Iterable[FinalizedEvent],
        soft_non_improving_depth: int = 3,
    ) -> LineageState:
        state = cls.empty(soft_non_improving_depth=soft_non_improving_depth)
        for event in events:
            if event.event_type == "CandidateSealed":
                state = state.add_version(CandidateVersion.from_payload(event.payload))
            elif event.event_type == "CandidatePromoted":
                state = state.promote(str(event.payload["version_id"]))
            elif event.event_type == "FrontierSelected":
                state = state.choose_frontier(
                    str(event.payload["version_id"]),
                    rationale=str(event.payload["rationale"]),
                )
        return state

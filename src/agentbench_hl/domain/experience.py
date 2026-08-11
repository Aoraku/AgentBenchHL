"""Grounded, append-only scientific experience records."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

_VERDICTS = frozenset(
    {
        "supported",
        "refuted",
        "mixed",
        "inconclusive",
        "integration_failure",
        "not_activated",
    }
)
_SELECTIONS = frozenset({"promoted", "frontier", "archived", "rejected", "incomplete"})
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
_CREDENTIAL = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


@dataclass(frozen=True)
class EvidenceWindow:
    match_id: str
    start_state_id: str
    end_state_id: str

    def __post_init__(self) -> None:
        for field in ("match_id", "start_state_id", "end_state_id"):
            value = _required_text(getattr(self, field), field)
            if not _SAFE_ID.fullmatch(value):
                raise ValueError(f"{field} is not a safe evidence identifier")

    def to_dict(self) -> dict[str, str]:
        return {
            "match_id": self.match_id,
            "start_state_id": self.start_state_id,
            "end_state_id": self.end_state_id,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> EvidenceWindow:
        return cls(
            match_id=str(value["match_id"]),
            start_state_id=str(value["start_state_id"]),
            end_state_id=str(value["end_state_id"]),
        )


@dataclass(frozen=True)
class ExperienceRecord:
    experience_id: str
    scientific_iteration: int
    target_opponent: str
    role: str
    verdict: str
    condition: str
    mechanism: str
    proposed_change: str
    expected_observation: str
    parent_id: str
    candidate_id: str
    selection: str
    match_ids: tuple[str, ...]
    evidence_windows: tuple[EvidenceWindow, ...]
    measured_outcome: Mapping[str, object]
    supersedes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.experience_id):
            raise ValueError("experience_id must be a safe identifier")
        if (
            isinstance(self.scientific_iteration, bool)
            or not isinstance(self.scientific_iteration, int)
            or self.scientific_iteration < 0
        ):
            raise ValueError("scientific_iteration must be a non-negative integer")
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("role must be a non-empty string")
        if self.verdict not in _VERDICTS:
            raise ValueError(f"unknown experience verdict: {self.verdict}")
        if self.selection not in _SELECTIONS:
            raise ValueError(f"unknown candidate selection: {self.selection}")
        for field in (
            "target_opponent",
            "condition",
            "mechanism",
            "proposed_change",
            "expected_observation",
            "parent_id",
            "candidate_id",
        ):
            _required_text(getattr(self, field), field)
        if not self.match_ids or len(set(self.match_ids)) != len(self.match_ids):
            raise ValueError("experience requires unique match IDs")
        if not self.evidence_windows:
            raise ValueError("experience requires replay evidence windows")
        unknown_matches = {window.match_id for window in self.evidence_windows} - set(
            self.match_ids
        )
        if unknown_matches:
            raise ValueError(f"evidence references unknown matches: {sorted(unknown_matches)}")
        if not self.measured_outcome:
            raise ValueError("experience requires a measured outcome")
        if self.experience_id in self.supersedes:
            raise ValueError("experience cannot supersede itself")
        if len(set(self.supersedes)) != len(self.supersedes):
            raise ValueError("supersedes must be unique")
        serialized = json.dumps(self.to_payload(), ensure_ascii=False, sort_keys=True)
        if _CREDENTIAL.search(serialized):
            raise ValueError("experience contains credential material")
        object.__setattr__(
            self,
            "measured_outcome",
            MappingProxyType(json.loads(json.dumps(dict(self.measured_outcome)))),
        )

    @property
    def is_strategy_failure(self) -> bool:
        return self.verdict == "refuted"

    def to_payload(self) -> dict[str, object]:
        return {
            "experience_id": self.experience_id,
            "scientific_iteration": self.scientific_iteration,
            "target_opponent": self.target_opponent,
            "role": self.role,
            "verdict": self.verdict,
            "condition": self.condition,
            "mechanism": self.mechanism,
            "proposed_change": self.proposed_change,
            "expected_observation": self.expected_observation,
            "parent_id": self.parent_id,
            "candidate_id": self.candidate_id,
            "selection": self.selection,
            "match_ids": list(self.match_ids),
            "evidence_windows": [item.to_dict() for item in self.evidence_windows],
            "measured_outcome": dict(self.measured_outcome),
            "supersedes": list(self.supersedes),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ExperienceRecord:
        windows = payload.get("evidence_windows")
        outcome = payload.get("measured_outcome")
        if not isinstance(windows, list) or not all(isinstance(item, Mapping) for item in windows):
            raise ValueError("experience evidence_windows must be a list")
        if not isinstance(outcome, Mapping):
            raise ValueError("experience measured_outcome must be a mapping")
        return cls(
            experience_id=str(payload["experience_id"]),
            scientific_iteration=int(payload["scientific_iteration"]),
            target_opponent=str(payload["target_opponent"]),
            role=str(payload["role"]),
            verdict=str(payload["verdict"]),
            condition=str(payload["condition"]),
            mechanism=str(payload["mechanism"]),
            proposed_change=str(payload["proposed_change"]),
            expected_observation=str(payload["expected_observation"]),
            parent_id=str(payload["parent_id"]),
            candidate_id=str(payload["candidate_id"]),
            selection=str(payload["selection"]),
            match_ids=tuple(str(item) for item in payload["match_ids"]),
            evidence_windows=tuple(EvidenceWindow.from_mapping(item) for item in windows),
            measured_outcome=dict(outcome),
            supersedes=tuple(str(item) for item in payload.get("supersedes", [])),
        )

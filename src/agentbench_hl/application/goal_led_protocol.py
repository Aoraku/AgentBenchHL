"""Public request contract between one long-lived Goal and the match bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["P0", "P1"]


def _non_empty_strings(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    result = tuple(item.strip() for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _positive_integers(value: object, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty non-negative integer list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True)
class MatchRequest:
    request_id: str
    candidate_ids: tuple[str, ...]
    opponent_id: str
    roles: tuple[Role, ...]
    seeds: tuple[int, ...]
    rationale: str

    @classmethod
    def from_path(cls, path: str | Path) -> MatchRequest:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read match request: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("match request must be a JSON object")
        request_id = value.get("action_id", value.get("request_id"))
        opponent_id = value.get("selected_rival", value.get("opponent_id"))
        rationale = value.get("rationale")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(opponent_id, str) or not opponent_id.strip():
            raise ValueError("opponent_id must be a non-empty string")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        if "rollouts" in value:
            rollouts = value["rollouts"]
            if not isinstance(rollouts, list):
                raise ValueError("rollouts must be a non-empty list")
            candidates = _non_empty_strings(
                [item.get("candidate_id") if isinstance(item, dict) else item for item in rollouts],
                "rollouts",
            )
        else:
            candidates = _non_empty_strings(value.get("candidate_ids"), "candidate_ids")
        raw_roles = _non_empty_strings(value.get("roles"), "roles")
        if any(item not in {"P0", "P1"} for item in raw_roles):
            raise ValueError("roles may contain only P0 or P1")
        return cls(
            request_id=request_id.strip(),
            candidate_ids=candidates,
            opponent_id=opponent_id.strip(),
            roles=tuple(raw_roles),  # type: ignore[arg-type]
            seeds=_positive_integers(value.get("seeds"), "seeds"),
            rationale=rationale.strip(),
        )

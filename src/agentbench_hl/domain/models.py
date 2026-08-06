"""Shared immutable scientific records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


def _optional_count(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or null")
    return value


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    wall_time_s: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Usage:
        wall_time = value.get("wall_time_s")
        if wall_time is not None and (
            isinstance(wall_time, bool) or not isinstance(wall_time, (int, float)) or wall_time < 0
        ):
            raise ValueError("wall_time_s must be non-negative or null")
        return cls(
            input_tokens=_optional_count(value.get("input_tokens"), "input_tokens"),
            cached_input_tokens=_optional_count(
                value.get("cached_input_tokens"), "cached_input_tokens"
            ),
            output_tokens=_optional_count(value.get("output_tokens"), "output_tokens"),
            reasoning_tokens=_optional_count(value.get("reasoning_tokens"), "reasoning_tokens"),
            total_tokens=_optional_count(value.get("total_tokens"), "total_tokens"),
            wall_time_s=None if wall_time is None else float(wall_time),
        )

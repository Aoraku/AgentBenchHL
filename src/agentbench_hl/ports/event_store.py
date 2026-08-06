"""Finalized event storage port."""

from __future__ import annotations

from typing import Protocol

from agentbench_hl.domain.events import FinalizedEvent


class EventStore(Protocol):
    def append(self, event: FinalizedEvent) -> bool: ...

    def append_many(self, events: tuple[FinalizedEvent, ...]) -> tuple[bool, ...]: ...

    def read_all(self) -> tuple[FinalizedEvent, ...]: ...

"""Durable append-only JSONL event storage."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agentbench_hl.domain.events import FinalizedEvent


class JsonlEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # Keep the idempotency index in memory for the lifetime of a runner.
        # Re-reading a growing JSONL file for every append makes metric
        # collection quadratic in the number of decision samples.
        self._loaded = False
        self._events: list[FinalizedEvent] = []
        self._by_key: dict[str, FinalizedEvent] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("event line is not an object")
                event = FinalizedEvent.from_dict(value)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid finalized event at line {line_number}: {exc}") from exc
            previous = self._by_key.get(event.idempotency_key)
            if previous is not None:
                if previous.event_type == event.event_type and dict(previous.payload) == dict(
                    event.payload
                ):
                    continue
                raise ValueError(
                    f"duplicate idempotency key at line {line_number}: {event.idempotency_key}"
                )
            self._by_key[event.idempotency_key] = event
            self._events.append(event)

    def read_all(self) -> tuple[FinalizedEvent, ...]:
        self._load()
        return tuple(self._events)

    def append(self, event: FinalizedEvent) -> bool:
        return self.append_many((event,))[0]

    def append_many(self, events: tuple[FinalizedEvent, ...]) -> tuple[bool, ...]:
        if not events:
            return ()
        self._load()
        existing = self._by_key
        pending: list[FinalizedEvent] = []
        outcomes: list[bool] = []
        staged: dict[str, FinalizedEvent] = {}
        for event in events:
            previous = existing.get(event.idempotency_key) or staged.get(event.idempotency_key)
            if previous is not None:
                if previous.event_type != event.event_type or dict(previous.payload) != dict(
                    event.payload
                ):
                    raise ValueError(f"conflicting idempotency key: {event.idempotency_key}")
                outcomes.append(False)
                continue
            staged[event.idempotency_key] = event
            pending.append(event)
            outcomes.append(True)
        if not pending:
            return tuple(outcomes)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = "".join(
            (
                json.dumps(
                    event.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            for event in pending
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Commit the in-memory index only after the complete batch is durable;
        # a validation error above therefore leaves the store unchanged.
        existing.update(staged)
        self._events.extend(pending)
        return tuple(outcomes)

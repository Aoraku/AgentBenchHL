from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentbench_hl.adapters.filesystem.event_store import JsonlEventStore
from agentbench_hl.domain.events import FinalizedEvent


def sealed_event() -> FinalizedEvent:
    return FinalizedEvent.create(
        "CandidateSealed",
        {"version_id": "v000", "content_hash": "a" * 64},
        idempotency_key="seal:v000",
        occurred_at=datetime(2026, 8, 4, tzinfo=UTC),
    )


def test_event_store_is_idempotent_and_append_only(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    event = sealed_event()

    assert store.append(event) is True
    assert store.append(event) is False
    assert store.read_all() == (event,)


def test_event_store_rejects_conflicting_idempotency_key(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    store.append(sealed_event())
    conflict = FinalizedEvent.create(
        "CandidateSealed",
        {"version_id": "v001", "content_hash": "b" * 64},
        idempotency_key="seal:v000",
        occurred_at=datetime(2026, 8, 4, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="conflicting idempotency"):
        store.append(conflict)


def test_event_store_appends_a_batch_with_one_ledger_scan(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    first = sealed_event()
    second = FinalizedEvent.create(
        "CandidateSealed",
        {"version_id": "v001", "content_hash": "b" * 64},
        idempotency_key="seal:v001",
        occurred_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    third = FinalizedEvent.create(
        "CandidateSealed",
        {"version_id": "v002", "content_hash": "c" * 64},
        idempotency_key="seal:v002",
        occurred_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    store.append(first)

    assert store.append_many((first, second, third)) == (False, True, True)
    assert store.read_all() == (first, second, third)


def test_event_store_rejects_a_conflicting_batch_atomically(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "events.jsonl")
    first = sealed_event()
    store.append(first)
    conflict = FinalizedEvent.create(
        "CandidateSealed",
        {"version_id": "v999", "content_hash": "d" * 64},
        idempotency_key="seal:v000",
        occurred_at=datetime(2026, 8, 4, tzinfo=UTC),
    )
    later = FinalizedEvent.create(
        "CandidateSealed",
        {"version_id": "v001", "content_hash": "b" * 64},
        idempotency_key="seal:v001",
        occurred_at=datetime(2026, 8, 4, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="conflicting idempotency"):
        store.append_many((later, conflict))

    assert store.read_all() == (first,)


def test_event_store_reports_truncated_json_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"schema_version":', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        JsonlEventStore(path).read_all()

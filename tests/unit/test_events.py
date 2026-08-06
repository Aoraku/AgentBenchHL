from datetime import UTC, datetime

import pytest

from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.models import Usage


def test_missing_usage_fields_stay_unknown() -> None:
    usage = Usage.from_mapping({"input_tokens": 10, "wall_time_s": 1.25})

    assert usage.input_tokens == 10
    assert usage.output_tokens is None
    assert usage.reasoning_tokens is None
    assert usage.total_tokens is None
    assert usage.wall_time_s == 1.25


def test_event_identity_is_canonical_for_payload_key_order() -> None:
    timestamp = datetime(2026, 8, 4, tzinfo=UTC)
    first = FinalizedEvent.create(
        "CandidateSealed",
        {"version_id": "v000", "content_hash": "a" * 64},
        "seal:v000",
        occurred_at=timestamp,
    )
    second = FinalizedEvent.create(
        "CandidateSealed",
        {"content_hash": "a" * 64, "version_id": "v000"},
        "seal:v000",
        occurred_at=timestamp,
    )

    assert first == second


def test_event_rejects_credential_material() -> None:
    with pytest.raises(ValueError, match="credential"):
        FinalizedEvent.create(
            "GoalStarted",
            {"value": "sk-" + "abcdefghijk"},
            "goal:1",
        )

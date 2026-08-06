"""Canonical finalized scientific events."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

_CREDENTIAL = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}")


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _event_id(
    event_type: str,
    idempotency_key: str,
    occurred_at: str,
    payload: Mapping[str, object],
) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "event_type": event_type,
                "idempotency_key": idempotency_key,
                "occurred_at": occurred_at,
                "payload": payload,
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class FinalizedEvent:
    schema_version: str
    event_id: str
    event_type: str
    idempotency_key: str
    occurred_at: str
    payload: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        event_type: str,
        payload: Mapping[str, object],
        idempotency_key: str,
        occurred_at: datetime | None = None,
    ) -> FinalizedEvent:
        if not event_type or not idempotency_key:
            raise ValueError("event type and idempotency key must be non-empty")
        timestamp = occurred_at or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        timestamp_text = timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")
        payload_copy = json.loads(_canonical(payload))
        serialized = _canonical(payload_copy).decode("utf-8")
        if _CREDENTIAL.search(serialized):
            raise ValueError("event payload contains credential material")
        identifier = _event_id(
            event_type,
            idempotency_key,
            timestamp_text,
            payload_copy,
        )
        return cls(
            schema_version="1.0",
            event_id=identifier,
            event_type=event_type,
            idempotency_key=idempotency_key,
            occurred_at=timestamp_text,
            payload=MappingProxyType(payload_copy),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FinalizedEvent:
        required = {
            "schema_version",
            "event_id",
            "event_type",
            "idempotency_key",
            "occurred_at",
            "payload",
        }
        if set(value) != required or value.get("schema_version") != "1.0":
            raise ValueError("invalid finalized event fields")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("event payload must be a mapping")
        event_type = str(value["event_type"])
        idempotency_key = str(value["idempotency_key"])
        occurred_at = str(value["occurred_at"])
        expected = _event_id(event_type, idempotency_key, occurred_at, payload)
        if value.get("event_id") != expected:
            raise ValueError("event identity mismatch")
        serialized = _canonical(payload).decode("utf-8")
        if _CREDENTIAL.search(serialized):
            raise ValueError("event payload contains credential material")
        return cls(
            schema_version="1.0",
            event_id=expected,
            event_type=event_type,
            idempotency_key=idempotency_key,
            occurred_at=occurred_at,
            payload=MappingProxyType(json.loads(_canonical(payload))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "idempotency_key": self.idempotency_key,
            "occurred_at": self.occurred_at,
            "payload": dict(self.payload),
        }

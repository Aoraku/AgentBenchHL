"""Abstract contract for decoding an official replay into audit evidence.

A game adapter knows how to translate its raw official replay JSON into a
structured, game-agnostic *report*: a winner, canonical frames, an atomic event
timeline, scalar metrics, strategic claims, critical windows and a natural
language narrative.  The framework's ``ReplayService`` only serializes that
report, so it depends on this Protocol rather than importing any game adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class ReplayClaim(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


class ReplayEvent(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


class ReplayWindow(Protocol):
    def to_dict(self) -> Mapping[str, object]: ...


class ReplayReport(Protocol):
    @property
    def winner(self) -> object: ...

    @property
    def frames(self) -> Sequence[object]: ...

    @property
    def timeline(self) -> Sequence[ReplayEvent]: ...

    @property
    def metrics(self) -> Mapping[str, object]: ...

    @property
    def strategic_claims(self) -> Sequence[ReplayClaim]: ...

    @property
    def critical_windows(self) -> Sequence[ReplayWindow]: ...

    @property
    def narrative(self) -> str: ...


class ReplayDecoder(Protocol):
    def __call__(
        self,
        rounds: Sequence[Mapping[str, object]],
        *,
        match_id: str,
    ) -> ReplayReport: ...

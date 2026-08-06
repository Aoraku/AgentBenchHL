"""Candidate artifact storage port."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    def materialize(self, source: Path) -> tuple[str, Path, Mapping[str, str]]: ...

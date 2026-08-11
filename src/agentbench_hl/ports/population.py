"""Abstract contract for a frozen human/opponent population entry.

Game adapters build a concrete opponent record (for example
``adapters.antwar2.runtime.Opponent``) that carries adapter-specific launch and
provenance fields.  The framework core only needs a small, game-agnostic view of
each entry: a stable identifier, a curriculum rank, a frozen ladder score, and
whether the entry can actually be run.  Depending on this Protocol instead of a
concrete adapter class keeps the core free of any game import.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PopulationEntry(Protocol):
    @property
    def opponent_id(self) -> str: ...

    @property
    def rank(self) -> int: ...

    @property
    def score(self) -> int | None: ...

    @property
    def runnable(self) -> bool: ...

"""Game registry and adapter factory.

The framework core is game-agnostic: it never imports a concrete game adapter.
Instead, each game registers a :class:`GameAdapterFactory` under its GamePack
name.  At run time the composition root looks up the factory for
``config.game`` and asks it to assemble that game's live run.

Adding a new game therefore requires only:

* a ``gamepacks/<game>/`` directory (validated by :mod:`agentbench_hl.config`);
* an ``adapters/<game>/`` package that builds the game's Arena/Replay/Runtime/
  PolicyProbe/pool audit;
* one registry entry that exposes a :class:`GameAdapterFactory`.

No edits to ``core`` / ``domain`` / ``ports`` / ``config`` are needed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from agentbench_hl.application.run_service import RunService


class GameAdapterFactory(Protocol):
    """Assemble one game's live run from a frozen experiment config.

    A factory owns everything game-specific: locating and building the frozen
    backend, auditing the human pool, wiring the arena/replay/policy-probe
    adapters, and constructing the isolated agent runtime.  It returns a fully
    assembled, game-agnostic :class:`RunService`.
    """

    def build_run(self, config_path: str | Path, *, run_id: str) -> RunService: ...

    def resume_run(self, config_path: str | Path, *, run_id: str) -> RunService: ...


_REGISTRY: dict[str, Callable[[], GameAdapterFactory]] = {}


def register_game(name: str, factory_provider: Callable[[], GameAdapterFactory]) -> None:
    """Register a lazily-constructed adapter factory under a GamePack name.

    ``factory_provider`` is a zero-argument callable that imports and returns the
    game's :class:`GameAdapterFactory`.  It is called lazily so that importing a
    concrete game adapter (and its heavy dependencies) only happens when that
    game is actually selected.
    """

    if not name:
        raise ValueError("game name must be non-empty")
    _REGISTRY[name] = factory_provider


def registered_games() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_game_adapters(name: str) -> GameAdapterFactory:
    """Return the adapter factory registered for ``name``.

    Raises ``ValueError`` with the set of known games if ``name`` is unknown.
    """

    provider = _REGISTRY.get(name)
    if provider is None:
        known = ", ".join(registered_games()) or "<none>"
        raise ValueError(f"no game adapter registered for {name!r}; registered games: {known}")
    return provider()


def _antwar2_factory() -> GameAdapterFactory:
    from agentbench_hl.adapters.antwar2.factory import AntWar2AdapterFactory

    return AntWar2AdapterFactory()


register_game("antwar2", _antwar2_factory)

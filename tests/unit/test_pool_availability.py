from __future__ import annotations

import json
from pathlib import Path

from agentbench_hl.adapters.contract.factory import (
    _supported_player_build_systems,
    _supports_compiled_players,
)
from agentbench_hl.adapters.contract.pool import (
    classify_availability,
    evaluator_content_sha256,
    load_pool,
)


def test_recursive_python_entry_is_runnable(tmp_path: Path) -> None:
    package = tmp_path / "player"
    (package / "archive" / "sdk").mkdir(parents=True)
    (package / "archive" / "sdk" / "main.py").write_text("pass\n", encoding="utf-8")

    status, diagnostic = classify_availability(package, supports_compiled=False)

    assert status == "runnable"
    assert diagnostic is None


def test_nested_build_is_runnable_when_evaluator_supports_compiled(tmp_path: Path) -> None:
    package = tmp_path / "player"
    (package / "source").mkdir(parents=True)
    (package / "source" / "Makefile").write_text(
        "main: main.cpp\n\tg++ main.cpp -o main\n", encoding="utf-8"
    )

    status, diagnostic = classify_availability(package, supports_compiled=True)

    assert status == "runnable"
    assert diagnostic is None


def test_native_source_without_build_metadata_is_not_called_a_bad_strategy(tmp_path: Path) -> None:
    package = tmp_path / "player"
    package.mkdir()
    (package / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")

    status, diagnostic = classify_availability(package, supports_compiled=True)

    assert status == "build_metadata_required"
    assert diagnostic == "native source found but no supported build entry"


def test_missing_package_is_an_ingestion_error(tmp_path: Path) -> None:
    status, diagnostic = classify_availability(
        tmp_path / "missing", supports_compiled=True
    )

    assert status == "missing_package"
    assert diagnostic == "package directory is missing"


def test_explicit_compiled_capability_beats_legacy_function_name_probe(tmp_path: Path) -> None:
    runtime = tmp_path / "games" / "python-only" / "evaluator" / "runtime.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text(
        "SUPPORTS_COMPILED_PLAYERS = False\n\ndef prepare_player():\n    pass\n",
        encoding="utf-8",
    )

    assert _supports_compiled_players(tmp_path, "python-only") is False


def test_build_capabilities_distinguish_make_from_cmake(tmp_path: Path) -> None:
    runtime = tmp_path / "games" / "make-only" / "evaluator" / "runtime.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text(
        'SUPPORTED_PLAYER_BUILD_SYSTEMS = ("make",)\n', encoding="utf-8"
    )
    package = tmp_path / "cmake-player"
    package.mkdir()
    (package / "CMakeLists.txt").write_text(
        "add_executable(main main.cpp)\n", encoding="utf-8"
    )

    systems = _supported_player_build_systems(tmp_path, "make-only")
    status, _ = classify_availability(
        package, supports_compiled=False, supported_build_systems=systems
    )

    assert systems == frozenset({"make"})
    assert status == "evaluator_unsupported"


def test_verified_audit_cannot_resurrect_missing_package(tmp_path: Path) -> None:
    game_dir = tmp_path / "games" / "game"
    players = game_dir / "players"
    evaluator = game_dir / "evaluator"
    core = tmp_path / "src" / "agentbench" / "core"
    players.mkdir(parents=True)
    evaluator.mkdir()
    core.mkdir(parents=True)
    (evaluator / "runtime.py").write_text("# evaluator\n", encoding="utf-8")
    (core / "contract.py").write_text("# contract\n", encoding="utf-8")
    (players / "manifest.tsv").write_text(
        "player_id\tdir\nmissing\tpool/missing\n", encoding="utf-8"
    )
    content_hash = evaluator_content_sha256(tmp_path, "game")
    (players / "runnable.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "evaluator_revision": "fixture",
                "evaluator_content_sha256": content_hash,
                "audit_fingerprint": "fixture",
                "verified_ids": ["missing"],
                "rows": [{"player_id": "missing", "verified": True}],
            }
        ),
        encoding="utf-8",
    )

    player = load_pool(tmp_path, "game")[0]

    assert player.runnable is False
    assert player.availability_status == "missing_package"

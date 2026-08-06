from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_hl.adapters.antwar2.runtime import (
    AntWarLayout,
    AntWarRuntimeError,
    safe_extract,
)


def test_layout_resolves_only_frozen_public_roots(tmp_path: Path) -> None:
    agentbench = tmp_path / "AgentBench"
    layout = AntWarLayout.from_root(agentbench, tmp_path / "build")

    assert layout.backend_archive == (
        agentbench / "backend_sources/corpus/30_antwar2/archives/gamecode_logic__141.zip"
    )
    assert layout.human_manifest == (
        agentbench / "top_algorithms/corpus/30_antwar2_ladder/MANIFEST.tsv"
    )
    assert "30_antwar2_ladder/extracted" in layout.human_extracted_root.as_posix()
    assert layout.public_sdk_root.name == "SDK"


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("../escape.txt", "forbidden")

    with pytest.raises(AntWarRuntimeError, match="unsafe"):
        safe_extract(archive, tmp_path / "target")
    assert not (tmp_path / "escape.txt").exists()


def test_build_manifest_cache_rejects_binary_hash_mismatch(tmp_path: Path) -> None:
    from agentbench_hl.adapters.antwar2.runtime import validate_cached_backend

    executable = tmp_path / "game" / "output" / "main"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"binary")
    manifest = tmp_path / "build-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archive_sha256": "a" * 64,
                "executable_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )

    assert validate_cached_backend(executable, manifest, "a" * 64) is None

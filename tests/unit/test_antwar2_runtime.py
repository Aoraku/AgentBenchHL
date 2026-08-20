from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_hl.adapters.antwar2.runtime import (
    AntWar2Layout,
    AntWar2RuntimeError,
    safe_extract,
)


def test_layout_resolves_only_frozen_public_roots(tmp_path: Path) -> None:
    """布局必须落在 A 现行目录结构内（实现已 re-export 自 A 的 evaluator）。

    历史版本断言的是 A 重构前的 ``top_algorithms/corpus/...`` 路径；A 迁到
    ``games/<game>/...`` 之后，这里跟随事实源更新，同时保持原意：
    只解析**冻结的公开资源**（后端归档、人类榜清单、公开 SDK）。
    """

    agentbench = tmp_path / "AgentBench"
    layout = AntWar2Layout.from_root(agentbench, tmp_path / "build")

    assert layout.backend_archive.is_relative_to(agentbench)
    assert layout.backend_archive.suffix == ".zip"
    assert layout.human_manifest == (agentbench / "games/antwar2/players/manifest.tsv")
    assert layout.human_extracted_root.is_relative_to(agentbench / "games/antwar2/players")
    assert layout.public_sdk_root.name == "SDK"


def test_safe_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    import zipfile

    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("../escape.txt", "forbidden")

    with pytest.raises(AntWar2RuntimeError, match="unsafe"):
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

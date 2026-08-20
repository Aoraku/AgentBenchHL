"""codex 模型元数据：必须由厂商官方 catalog 说话，不能靠兜底。

代价已经付过了：不给 catalog 时 codex 报 "Unknown model … will use fallback
model metadata"，压缩于是在一个我们**没设过也看不见**的点触发——实测 antwar2
在 97k 上下文触发压缩并死掉，而我们当时在 config.toml 里写的是 200000。

所以这里锁住：catalog 会被真的写进 codex home、config.toml 指向它、
而且用了 catalog 就不再重复写 model_context_window（两处声明打架时，
"压缩线到底是多少"就变成一个要读 codex 源码才能回答的问题）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_hl.adapters.codex_goal.app_server import (
    MODEL_CATALOGS,
    ZHIPU_MODEL_CATALOG,
    write_codex_config,
)


def _config(root: Path, **kwargs: object) -> str:
    write_codex_config(
        root,
        base_url="https://example.invalid/api",
        model="glm-5.3",
        reasoning_effort="high",
        **kwargs,  # type: ignore[arg-type]
    )
    return (root / "config.toml").read_text(encoding="utf-8")


def test_catalog_is_written_and_referenced(tmp_path: Path) -> None:
    document = _config(tmp_path, model_catalog="zhipu", auto_compact_token_limit=900_000)

    catalog_path = tmp_path / "models.json"
    assert catalog_path.is_file()
    assert f'model_catalog_json = "{catalog_path}"' in document
    # 压缩线不在 catalog 的字段集里，必须仍由 config.toml 提供。
    assert "model_auto_compact_token_limit = 900000" in document

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    slugs = {entry["slug"] for entry in payload["models"]}
    assert "glm-5.3" in slugs
    flagship = next(entry for entry in payload["models"] if entry["slug"] == "glm-5.3")
    assert flagship["context_window"] == 1_048_576
    assert flagship["effective_context_window_percent"] == 95


def test_catalog_replaces_hand_written_window(tmp_path: Path) -> None:
    """给了 catalog 就不该再写 model_context_window——避免两处声明打架。"""

    document = _config(tmp_path, model_catalog="zhipu", context_window=200_000)
    assert "model_context_window" not in document
    assert "model_catalog_json" in document


def test_without_catalog_falls_back_to_explicit_window(tmp_path: Path) -> None:
    """没有 catalog 时仍要能手工声明，否则老配置直接失效。"""

    document = _config(tmp_path, context_window=200_000)
    assert "model_context_window = 200000" in document
    assert "model_catalog_json" not in document
    assert not (tmp_path / "models.json").exists()


def test_unknown_catalog_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown model_catalog"):
        _config(tmp_path, model_catalog="does-not-exist")


def test_catalog_entries_carry_required_codex_fields() -> None:
    """codex 以 --strict-config 启动，字段缺失/写错会被直接拒。"""

    required = {
        "slug",
        "display_name",
        "context_window",
        "max_context_window",
        "effective_context_window_percent",
        "supported_reasoning_levels",
        "default_reasoning_level",
        "shell_type",
        "apply_patch_tool_type",
        "truncation_policy",
        "input_modalities",
    }
    assert MODEL_CATALOGS["zhipu"] is ZHIPU_MODEL_CATALOG
    for entry in ZHIPU_MODEL_CATALOG:
        missing = required - set(entry)
        assert not missing, f"{entry.get('slug')} 缺少字段：{sorted(missing)}"

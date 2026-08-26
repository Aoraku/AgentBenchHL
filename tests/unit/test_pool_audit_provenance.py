from __future__ import annotations

import json
from pathlib import Path

from agentbench_hl.adapters.contract.pool import _audit_verdicts
from agentbench_hl.application.pool_audit import _merge_audit_rows, load_verified_ids


def test_changed_evaluator_fingerprint_drops_stale_verdicts() -> None:
    existing = {
        "audit_fingerprint": "old-evaluator",
        "rows": [{"player_id": "stale", "verified": False}],
    }
    current = [{"player_id": "fresh", "verified": True}]

    merged = _merge_audit_rows(existing, current, audit_fingerprint="new-evaluator")

    assert merged == {"fresh": current[0]}


def test_same_evaluator_fingerprint_supports_incremental_resume() -> None:
    existing = {
        "audit_fingerprint": "same-evaluator",
        "rows": [{"player_id": "kept", "verified": True}],
    }
    current = [{"player_id": "fresh", "verified": False}]

    merged = _merge_audit_rows(existing, current, audit_fingerprint="same-evaluator")

    assert merged == {"kept": existing["rows"][0], "fresh": current[0]}


def test_legacy_audit_without_evaluator_provenance_is_ignored(tmp_path: Path) -> None:
    game_dir = tmp_path / "games" / "game"
    players = game_dir / "players"
    players.mkdir(parents=True)
    (players / "runnable.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "rows": [
                    {"player_id": "stale", "verified": False, "diagnostic": "old failure"}
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _audit_verdicts(game_dir) == {}
    assert load_verified_ids(tmp_path, "game") is None

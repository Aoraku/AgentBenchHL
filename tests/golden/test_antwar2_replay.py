from __future__ import annotations

import json
from pathlib import Path

from agentbench_hl.adapters.antwar2.replay import decode_replay
from agentbench_hl.application.replay_service import ReplayService

FIXTURES = Path(__file__).with_name("antwar2_replays")


def load_fixture() -> list[dict[str, object]]:
    return json.loads((FIXTURES / "fixture.json").read_text(encoding="utf-8"))


def test_replay_decodes_numbers_into_grounded_chinese() -> None:
    report = decode_replay(load_fixture(), match_id="fixture")

    assert report.frames[0].base_hp == (50.0, 50.0)
    assert report.timeline[0].state_id == "fixture:r0001:p0"
    assert "在 (8,11) 建造基础塔" in report.timeline[0].text
    assert "state_id=fixture:r0001:p0" in report.narrative
    assert report.metrics["first_weapon_round"]["P0"] == 28
    assert report.metrics["weapon_target_coverage"]["P0"] == 1
    assert report.metrics["base_breaches"] == [
        {"round": 28, "player": "P1", "damage": 8.0},
        {"round": 29, "player": "P1", "damage": 42.0},
    ]
    assert report.metrics["downgrade_to_weapon_delay"]["P0"] is None
    assert report.frames[2].towers[0]["id"] == 5
    assert report.winner == "P0"


def test_every_strategic_claim_has_valid_evidence_reference() -> None:
    report = decode_replay(load_fixture(), match_id="fixture")

    for claim in report.strategic_claims:
        assert claim.evidence_state_ids
        assert all(state_id in report.frame_by_id for state_id in claim.evidence_state_ids)


def test_replay_service_materializes_deterministic_artifacts(tmp_path: Path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text(
        (FIXTURES / "fixture.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    artifacts = ReplayService(tmp_path / "artifacts", decode=decode_replay).materialize(
        match_id="fixture",
        replay_path=replay,
    )

    assert artifacts.summary_json.is_file()
    assert artifacts.timeline_jsonl.read_text(encoding="utf-8") == (
        FIXTURES / "expected_timeline.jsonl"
    ).read_text(encoding="utf-8")
    assert artifacts.narrative_md.read_text(encoding="utf-8") == (
        FIXTURES / "expected_narrative.md"
    ).read_text(encoding="utf-8")

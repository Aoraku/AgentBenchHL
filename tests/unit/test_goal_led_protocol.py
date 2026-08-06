from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentbench_hl.application.goal_led_protocol import MatchRequest


def write_request(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_request_loads_multiple_candidate_snapshots_and_match_matrix(tmp_path: Path) -> None:
    request = MatchRequest.from_path(
        write_request(
            tmp_path / "match_request.json",
            {
                "request_id": "rank01-opening-study",
                "candidate_ids": ["v000", "v001a"],
                "opponent_id": "rank01",
                "roles": ["P0", "P1"],
                "seeds": [1, 2],
                "rationale": "Compare two replay-grounded opening responses.",
            },
        )
    )

    assert request.request_id == "rank01-opening-study"
    assert request.candidate_ids == ("v000", "v001a")
    assert request.opponent_id == "rank01"
    assert request.roles == ("P0", "P1")
    assert request.seeds == (1, 2)


def test_action_loads_goal_native_rollouts_and_selected_rival(tmp_path: Path) -> None:
    request = MatchRequest.from_path(
        write_request(
            tmp_path / "action.json",
            {
                "action_id": "opening-distillation",
                "rollouts": [{"candidate_id": "v000"}, {"candidate_id": "v001a"}],
                "selected_rival": "rank01",
                "roles": ["P0", "P1"],
                "seeds": [7],
                "rationale": "Test four independent, replay-grounded policy proposals.",
            },
        )
    )

    assert request.request_id == "opening-distillation"
    assert request.candidate_ids == ("v000", "v001a")
    assert request.opponent_id == "rank01"


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("roles", ["P2"], "roles"),
        ("candidate_ids", ["v000", "v000"], "candidate_ids"),
        ("candidate_ids", [], "candidate_ids"),
    ],
)
def test_request_rejects_invalid_match_contract(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    payload: dict[str, object] = {
        "request_id": "bad-request",
        "candidate_ids": ["v000"],
        "opponent_id": "rank01",
        "roles": ["P0"],
        "seeds": [1],
        "rationale": "test",
    }
    payload[field] = value

    with pytest.raises(ValueError, match=error):
        MatchRequest.from_path(write_request(tmp_path / "match_request.json", payload))

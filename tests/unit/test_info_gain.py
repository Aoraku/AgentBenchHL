"""逐轮信息增益度量的离线契约测试。

覆盖三件事（都不需要真实 LLM/对局）：

1. ε-正则化的结果分布 KL 在数学上正确（含"完全相同 ⇒ IG=0"）；
2. 回放分歧位置对文本回放（记录级）与二进制回放（字节级）都能给出结果；
3. 配对不成立时（首轮无基线、无配对样本）诚实返回 None + 原因，
   绝不用胜率差之类的替代量冒充信息增益。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from agentbench_hl.application.info_gain import (
    outcome_counts,
    outcome_ig_nats,
    paired_margin_shift,
    replay_divergence,
)


def test_identical_outcome_distributions_have_zero_information_gain() -> None:
    counts = {"win": 3, "draw": 1, "loss": 2}
    assert outcome_ig_nats(counts, counts, epsilon=0.02) == 0.0


def test_outcome_ig_matches_the_closed_form_kl() -> None:
    parent = {"win": 1, "draw": 0, "loss": 3}
    candidate = {"win": 3, "draw": 0, "loss": 1}
    epsilon = 0.5
    total = 4 + 3 * epsilon
    p = [(1 + epsilon) / total, epsilon / total, (3 + epsilon) / total]
    q = [(3 + epsilon) / total, epsilon / total, (1 + epsilon) / total]
    expected = sum(qi * math.log(qi / pi) for pi, qi in zip(p, q, strict=True))

    assert outcome_ig_nats(parent, candidate, epsilon=epsilon) == expected
    # 方向性：候选与基线分布不同 ⇒ KL > 0。
    # （注意本例的两个分布互为镜像，两个方向的 KL 恰好相等，不能用来验证不对称性。）
    assert outcome_ig_nats(parent, candidate, epsilon=epsilon) > 0


def test_outcome_ig_is_asymmetric() -> None:
    parent = {"win": 4, "draw": 0, "loss": 0}
    candidate = {"win": 2, "draw": 1, "loss": 1}

    forward = outcome_ig_nats(parent, candidate, epsilon=0.02)
    backward = outcome_ig_nats(candidate, parent, epsilon=0.02)

    assert forward > 0 and backward > 0
    assert forward != backward


def test_outcome_counts_ignores_infrastructure_failures() -> None:
    rows = [
        {"status": "complete", "result": "win"},
        {"status": "complete", "result": "loss"},
        {"status": "infra_error", "result": None},
        {"status": "incomplete", "result": "win"},
    ]
    assert outcome_counts(rows) == {"win": 1, "draw": 0, "loss": 1}


def test_replay_divergence_reports_record_index_for_json_replays(tmp_path: Path) -> None:
    left = tmp_path / "parent.json"
    right = tmp_path / "candidate.json"
    left.write_text(json.dumps([{"r": 1}, {"r": 2}, {"r": 3}]), encoding="utf-8")
    right.write_text(json.dumps([{"r": 1}, {"r": 2}, {"r": 9}, {"r": 10}]), encoding="utf-8")

    report = replay_divergence(left, right)

    assert report["available"] is True
    assert report["unit"] == "record"
    assert report["identical"] is False
    assert report["first_divergence"] == 2
    assert report["divergence_frac"] == 0.5  # 2 / max(3, 4)


def test_identical_replays_are_reported_as_no_behaviour_change(tmp_path: Path) -> None:
    payload = json.dumps([{"r": 1}, {"r": 2}])
    left = tmp_path / "a.json"
    right = tmp_path / "b.json"
    left.write_text(payload, encoding="utf-8")
    right.write_text(payload, encoding="utf-8")

    report = replay_divergence(left, right)

    assert report["identical"] is True
    assert report["first_divergence"] is None
    assert report["divergence_frac"] is None


def test_binary_replays_fall_back_to_byte_level_divergence(tmp_path: Path) -> None:
    left = tmp_path / "a.bin"
    right = tmp_path / "b.bin"
    left.write_bytes(b"\x00\x01\x02\x03")
    right.write_bytes(b"\x00\x01\xff\x03")

    report = replay_divergence(left, right)

    assert report["unit"] == "byte"
    assert report["first_divergence"] == 2
    assert report["identical"] is False


def test_missing_replay_is_reported_as_unavailable(tmp_path: Path) -> None:
    report = replay_divergence(tmp_path / "missing.json", tmp_path / "also-missing.json")

    assert report["available"] is False
    assert "missing" in str(report["reason"])


def test_paired_margin_shift_only_uses_matched_cases() -> None:
    def row(seed: int, role: str, margin: float) -> dict[str, object]:
        return {
            "status": "complete",
            "opponent_id": "rank01",
            "role": role,
            "seed": seed,
            "score_margin": margin,
        }

    parent = [row(7, "P0", -2), row(7, "P1", 1)]
    # 第二条未配对（seed 不同）：必须被忽略，否则对手/局面不同的观测会混进来。
    candidate = [row(7, "P0", 3), row(9, "P1", 50)]

    assert paired_margin_shift(parent, candidate) == 5.0
    assert paired_margin_shift(parent, []) is None

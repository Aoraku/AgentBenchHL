"""逐轮信息增益度量 —— 游戏无关、可复现、绝不用近似值冒充真值。

## 为什么不能直接用"上一轮胜率之差"

胜率差（``outcome_shift``）不是信息量，而且**对手不同就不可比**：上一轮打 rank12、
这一轮打 rank03，胜率下降既可能是策略退化，也可能只是对手更强。

## 本模块的度量：配对影子对局（paired shadow matches）

每一轮拿到本轮最优候选后，把**上一轮冠军**（parent 快照）放到**完全相同的**
case 上再跑一遍（同对手、同座次、同 seed），于是得到严格配对的两组观测：

1. ``outcome_ig_nats``（nats/局，全部游戏可得）
   在 {win, draw, loss} 三点支撑上做 ε-正则化的 KL：

   ``IG = Σ_o q(o) · ln(q(o) / p(o))``，其中
   ``p = (n_parent(o) + ε) / (N + 3ε)``，``q`` 同理。

   这是"本轮把可观测结果分布推动了多少"的真实信息量；因为配对，对手强弱被消掉。

2. ``behavior_divergence``（全部游戏可得）
   同 case 下两份回放的**首次分歧位置**。游戏确定 + 对手/seed 相同 ⇒ 分歧点之前
   两者的公开状态逐帧相同，分歧点就是"行为第一次不同的时刻"。
   ``divergence_frac`` 越小表示改动越早地改变了行为；``identical=true`` 表示这一轮
   在该 case 上**行为完全没变**（IG 应当为 0，可用于交叉验证）。

3. ``behavioral_ig``（nats/决策，**需要该游戏的策略探针**）
   决策级 KL 是金标准，但它要求能在冻结状态上重放策略的合法动作分布，属于
   游戏专有能力（见 :mod:`agentbench_hl.ports.policy_probe`）。没有探针时这里
   诚实记 ``null`` 并写明 ``behavioral_ig_reason``，绝不用上面两个代替。

所有度量都只依赖公开信息（回放 + 契约三态），不需要读任何人类源码。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

OUTCOMES = ("win", "draw", "loss")
_BYTE_CHUNK = 1 << 20


def outcome_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """统计一组对局的三态结果（只计入 ``status=complete``）。"""

    counts = dict.fromkeys(OUTCOMES, 0)
    for row in rows:
        if row.get("status") != "complete":
            continue
        result = str(row.get("result") or "")
        if result in counts:
            counts[result] += 1
    return counts


def _smoothed(counts: Mapping[str, int], epsilon: float) -> dict[str, float]:
    total = sum(counts.get(name, 0) for name in OUTCOMES)
    denominator = total + epsilon * len(OUTCOMES)
    return {name: (counts.get(name, 0) + epsilon) / denominator for name in OUTCOMES}


def outcome_ig_nats(
    parent: Mapping[str, int],
    candidate: Mapping[str, int],
    *,
    epsilon: float,
) -> float:
    """ε-正则化的结果分布 KL（nats）：``KL(candidate ‖ parent)``。"""

    if epsilon <= 0:
        raise ValueError("epsilon must be positive to regularize an empirical KL")
    p = _smoothed(parent, epsilon)
    q = _smoothed(candidate, epsilon)
    return sum(q[name] * math.log(q[name] / p[name]) for name in OUTCOMES)


def _records(path: Path) -> list[object] | None:
    """尽量把回放解析成"记录序列"；不是 JSON 数组时返回 None（走字节比较）。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, Mapping):
        for key in ("records", "rounds", "frames", "log"):
            inner = value.get(key)
            if isinstance(inner, list):
                return inner
    return None


def _byte_divergence(left: Path, right: Path) -> dict[str, object]:
    size_left = left.stat().st_size
    size_right = right.stat().st_size
    offset = 0
    with left.open("rb") as first, right.open("rb") as second:
        while True:
            chunk_a = first.read(_BYTE_CHUNK)
            chunk_b = second.read(_BYTE_CHUNK)
            if not chunk_a or not chunk_b:
                break
            limit = min(len(chunk_a), len(chunk_b))
            for index in range(limit):
                if chunk_a[index] != chunk_b[index]:
                    offset += index
                    return {
                        "unit": "byte",
                        "identical": False,
                        "first_divergence": offset,
                        "parent_length": size_left,
                        "candidate_length": size_right,
                    }
            offset += limit
            if len(chunk_a) != len(chunk_b):
                break
    identical = size_left == size_right and offset >= min(size_left, size_right)
    return {
        "unit": "byte",
        "identical": bool(identical and size_left == size_right),
        "first_divergence": None if identical and size_left == size_right else offset,
        "parent_length": size_left,
        "candidate_length": size_right,
    }


def replay_divergence(parent_replay: str | Path, candidate_replay: str | Path) -> dict[str, object]:
    """同 case 下两份回放的首次分歧位置（记录级优先，二进制回放退化为字节级）。"""

    left = Path(parent_replay)
    right = Path(candidate_replay)
    if not left.is_file() or not right.is_file():
        return {"available": False, "reason": "replay file missing"}
    left_records = _records(left)
    right_records = _records(right)
    if left_records is None or right_records is None:
        payload = _byte_divergence(left, right)
    else:
        first: int | None = None
        for index in range(min(len(left_records), len(right_records))):
            if left_records[index] != right_records[index]:
                first = index
                break
        identical = first is None and len(left_records) == len(right_records)
        payload = {
            "unit": "record",
            "identical": identical,
            "first_divergence": (
                None
                if identical
                else (first if first is not None else min(len(left_records), len(right_records)))
            ),
            "parent_length": len(left_records),
            "candidate_length": len(right_records),
        }
    longest = max(int(payload["parent_length"] or 0), int(payload["candidate_length"] or 0))
    divergence = payload.get("first_divergence")
    payload["available"] = True
    payload["divergence_frac"] = (
        None if divergence is None or longest == 0 else round(int(divergence) / longest, 6)
    )
    return payload


def paired_margin_shift(
    parent_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> float | None:
    """配对 case 上的平均分差变化（正=候选更好）；无可配对样本时返回 None。"""

    def index(rows: Sequence[Mapping[str, object]]) -> dict[tuple[str, str, int], float]:
        table: dict[tuple[str, str, int], float] = {}
        for row in rows:
            if row.get("status") != "complete" or row.get("score_margin") is None:
                continue
            key = (str(row.get("opponent_id")), str(row.get("role")), int(row.get("seed") or 0))
            table[key] = float(row["score_margin"])  # type: ignore[arg-type]
        return table

    parent = index(parent_rows)
    candidate = index(candidate_rows)
    shared = sorted(set(parent) & set(candidate))
    if not shared:
        return None
    return sum(candidate[key] - parent[key] for key in shared) / len(shared)

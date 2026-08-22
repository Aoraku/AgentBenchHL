"""Public request contract between one long-lived Goal and the match bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Role = Literal["P0", "P1"]


def _non_empty_strings(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty string list")
    result = tuple(item.strip() for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _positive_integers(value: object, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
    ):
        raise ValueError(f"{field} must be a non-empty non-negative integer list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


@dataclass(frozen=True)
class MatchRequest:
    request_id: str
    candidate_ids: tuple[str, ...]
    opponent_id: str
    roles: tuple[Role, ...]
    seeds: tuple[int, ...]
    rationale: str
    #: agent 自选的**多个**对手（``self`` 策略下的 b 个）。
    #:
    #: 为什么要有复数字段：k=1 之后一轮是"一个策略打 b 个对手"，而 ``self``
    #: 策略要求 agent 自己挑那 b 个。只有单数的 ``selected_rival`` 时，agent
    #: 的自主权被压缩成"只能挑 1 个"，b>1 时框架只能替它补齐——那就不是
    #: "自己决定"了，消融变量也不干净。
    #:
    #: 空元组 = agent 只写了单数字段，此时以 ``opponent_id`` 为唯一对手。
    opponent_ids: tuple[str, ...] = ()

    @property
    def selected_opponents(self) -> tuple[str, ...]:
        """agent 本轮点名的对手（至少一个）。"""

        return self.opponent_ids or (self.opponent_id,)

    @classmethod
    def from_path(cls, path: str | Path) -> MatchRequest:
        source = Path(path)
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read match request: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("match request must be a JSON object")
        request_id = value.get("action_id", value.get("request_id"))
        rationale = value.get("rationale")
        # 复数优先：``selected_rivals`` / ``opponent_ids`` 都接受；
        # 没写复数就回落单数，老配置与老 run 的 action.json 逐字兼容。
        raw_many = value.get("selected_rivals", value.get("opponent_ids"))
        many: tuple[str, ...] = ()
        if raw_many is not None:
            many = _non_empty_strings(raw_many, "selected_rivals")
        opponent_id = value.get("selected_rival", value.get("opponent_id"))
        if opponent_id is None and many:
            opponent_id = many[0]
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(opponent_id, str) or not opponent_id.strip():
            raise ValueError("opponent_id must be a non-empty string")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        if "rollouts" in value:
            rollouts = value["rollouts"]
            if not isinstance(rollouts, list):
                raise ValueError("rollouts must be a non-empty list")
            candidates = _non_empty_strings(
                [item.get("candidate_id") if isinstance(item, dict) else item for item in rollouts],
                "rollouts",
            )
        else:
            candidates = _non_empty_strings(value.get("candidate_ids"), "candidate_ids")
        raw_roles = _non_empty_strings(value.get("roles"), "roles")
        if any(item not in {"P0", "P1"} for item in raw_roles):
            raise ValueError("roles may contain only P0 or P1")
        return cls(
            request_id=request_id.strip(),
            candidate_ids=candidates,
            opponent_id=opponent_id.strip(),
            roles=tuple(raw_roles),  # type: ignore[arg-type]
            seeds=_positive_integers(value.get("seeds"), "seeds"),
            rationale=rationale.strip(),
            opponent_ids=many,
        )

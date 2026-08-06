"""Render grounded Experience records into stable research documents."""

from __future__ import annotations

import json
from collections.abc import Iterable

from agentbench_hl.domain.experience import ExperienceRecord


def render_record(record: ExperienceRecord) -> str:
    evidence = ", ".join(
        f"{window.match_id}[{window.start_state_id}..{window.end_state_id}]"
        for window in record.evidence_windows
    )
    supersedes = ", ".join(record.supersedes) or "无"
    return "\n".join(
        (
            f"### {record.experience_id}",
            "",
            f"- 科研迭代：{record.scientific_iteration}",
            f"- 对手与角色：{record.target_opponent} / {record.role}",
            f"- 结论：{record.verdict}",
            f"- 公开条件：{record.condition}",
            f"- 机制：{record.mechanism}",
            f"- 候选改动：{record.proposed_change}",
            f"- 预期现象：{record.expected_observation}",
            f"- 版本：{record.parent_id} → {record.candidate_id} ({record.selection})",
            f"- 证据：{evidence}",
            "- 实测："
            + json.dumps(dict(record.measured_outcome), ensure_ascii=False, sort_keys=True),
            f"- 取代经验：{supersedes}",
            "",
        )
    )


def render_document(title: str, records: Iterable[ExperienceRecord]) -> str:
    ordered = tuple(
        sorted(records, key=lambda item: (item.scientific_iteration, item.experience_id))
    )
    body = "\n".join(render_record(record) for record in ordered)
    content = body if body else "无记录。\n"
    return f"# {title}\n\n{content}"


def render_context(records: Iterable[ExperienceRecord]) -> str:
    return render_document("相关 Experience 上下文", records)

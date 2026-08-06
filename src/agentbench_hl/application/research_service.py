"""Append, retrieve, and materialize positive and negative Experience."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.experience import ExperienceRecord
from agentbench_hl.ports.event_store import EventStore
from agentbench_hl.reporting.research_report import (
    render_context,
    render_document,
    render_record,
)


@dataclass(frozen=True)
class ResearchArtifacts:
    playbook: Path
    failed_hypotheses: Path
    open_questions: Path
    role_p0: Path
    role_p1: Path
    opponent_notes: Path
    iteration_reports: tuple[Path, ...]


@dataclass(frozen=True)
class ResearchContext:
    target: str
    role: str
    records: tuple[ExperienceRecord, ...]
    markdown: str


class ResearchService:
    def __init__(self, *, event_store: EventStore, artifact_root: str | Path) -> None:
        self.event_store = event_store
        self.artifact_root = Path(artifact_root)
        self._records = self._replay()

    def _replay(self) -> tuple[ExperienceRecord, ...]:
        return tuple(
            ExperienceRecord.from_payload(event.payload)
            for event in self.event_store.read_all()
            if event.event_type == "ExperienceRecorded"
        )

    def read_all(self) -> tuple[ExperienceRecord, ...]:
        return self._records

    def record(self, record: ExperienceRecord) -> bool:
        existing = next(
            (item for item in self._records if item.experience_id == record.experience_id),
            None,
        )
        if existing is not None:
            # Experience ids are the stable replay key.  A resumed provider
            # may regenerate the same id with a richer wording/outcome after
            # a crash; the durable ledger remains authoritative and must not
            # make the whole long-running experiment unrecoverable.
            return False
        unknown = set(record.supersedes) - {item.experience_id for item in self._records}
        if unknown:
            raise ValueError(f"supersedes references unknown Experience: {sorted(unknown)}")
        event = FinalizedEvent.create(
            "ExperienceRecorded",
            record.to_payload(),
            idempotency_key=f"experience-recorded:{record.experience_id}",
        )
        appended = self.event_store.append(event)
        if appended:
            self._records = (*self._records, record)
        return appended

    def context(self, *, target: str, role: str, max_records: int) -> ResearchContext:
        if role not in {"P0", "P1"}:
            raise ValueError("research context role must be P0 or P1")
        if max_records < 1:
            raise ValueError("max_records must be positive")
        relevant = [
            record
            for record in self._records
            if record.target_opponent == target and record.role == role
        ]
        relevant.sort(
            key=lambda item: (item.scientific_iteration, item.experience_id),
            reverse=True,
        )
        positive = next(
            (item for item in relevant if item.verdict in {"supported", "mixed"}),
            None,
        )
        caution = next(
            (
                item
                for item in relevant
                if item.verdict
                in {
                    "refuted",
                    "inconclusive",
                    "integration_failure",
                    "not_activated",
                }
            ),
            None,
        )
        selected: list[ExperienceRecord] = []
        for required in (positive, caution):
            if required is not None and required not in selected and len(selected) < max_records:
                selected.append(required)
        for record in relevant:
            if record not in selected and len(selected) < max_records:
                selected.append(record)
        selected.sort(key=lambda item: (item.scientific_iteration, item.experience_id))
        records = tuple(selected)
        return ResearchContext(target, role, records, render_context(records))

    def materialize(self) -> ResearchArtifacts:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        records = tuple(
            sorted(
                self._records,
                key=lambda item: (item.scientific_iteration, item.experience_id),
            )
        )
        paths = {
            "playbook": self.artifact_root / "PLAYBOOK.md",
            "failed": self.artifact_root / "FAILED_HYPOTHESES.md",
            "open": self.artifact_root / "OPEN_QUESTIONS.md",
            "p0": self.artifact_root / "ROLE_P0.md",
            "p1": self.artifact_root / "ROLE_P1.md",
            "opponents": self.artifact_root / "OPPONENT_NOTES.md",
        }
        paths["playbook"].write_text(
            render_document(
                "可复用策略经验",
                (item for item in records if item.verdict in {"supported", "mixed"}),
            ),
            encoding="utf-8",
        )
        paths["failed"].write_text(
            render_document(
                "被实战证伪的假设",
                (item for item in records if item.verdict == "refuted"),
            ),
            encoding="utf-8",
        )
        paths["open"].write_text(
            render_document(
                "待验证与未激活假设",
                (
                    item
                    for item in records
                    if item.verdict in {"inconclusive", "integration_failure", "not_activated"}
                ),
            ),
            encoding="utf-8",
        )
        paths["p0"].write_text(
            render_document("P0 经验", (item for item in records if item.role == "P0")),
            encoding="utf-8",
        )
        paths["p1"].write_text(
            render_document("P1 经验", (item for item in records if item.role == "P1")),
            encoding="utf-8",
        )
        paths["opponents"].write_text(
            render_document("对手公开行为笔记", records),
            encoding="utf-8",
        )
        report_root = self.artifact_root / "iterations"
        report_root.mkdir(exist_ok=True)
        reports: list[Path] = []
        for record in records:
            report = report_root / (
                f"iteration-{record.scientific_iteration:04d}-{record.experience_id}.md"
            )
            report.write_text(
                f"# 科研迭代 {record.scientific_iteration}\n\n{render_record(record)}",
                encoding="utf-8",
            )
            reports.append(report)
        return ResearchArtifacts(
            playbook=paths["playbook"],
            failed_hypotheses=paths["failed"],
            open_questions=paths["open"],
            role_p0=paths["p0"],
            role_p1=paths["p1"],
            opponent_notes=paths["opponents"],
            iteration_reports=tuple(reports),
        )

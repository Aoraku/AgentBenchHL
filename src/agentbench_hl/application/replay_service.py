"""Durably materialize deterministic public replay evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.ports.replay import ReplayDecoder


@dataclass(frozen=True)
class ReplayArtifacts:
    summary_json: Path
    timeline_jsonl: Path
    critical_windows_json: Path
    narrative_md: Path


class ReplayService:
    def __init__(self, root: str | Path, *, decode: ReplayDecoder) -> None:
        self.root = Path(root)
        self.decode = decode

    def materialize(self, *, match_id: str, replay_path: str | Path) -> ReplayArtifacts:
        source = Path(replay_path)
        raw = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
            raise ValueError("official replay must be a list of round objects")
        report = self.decode(raw, match_id=match_id)
        target = self.root / match_id
        target.mkdir(parents=True, exist_ok=True)
        artifacts = ReplayArtifacts(
            summary_json=target / "summary.json",
            timeline_jsonl=target / "timeline.jsonl",
            critical_windows_json=target / "critical-windows.json",
            narrative_md=target / "narrative.md",
        )
        artifacts.summary_json.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "match_id": match_id,
                    "winner": report.winner,
                    "frame_count": len(report.frames),
                    "atomic_event_count": len(report.timeline),
                    "metrics": dict(report.metrics),
                    "strategic_claims": [claim.to_dict() for claim in report.strategic_claims],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts.timeline_jsonl.write_text(
            "".join(
                json.dumps(
                    event.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for event in report.timeline
            ),
            encoding="utf-8",
        )
        artifacts.critical_windows_json.write_text(
            json.dumps(
                [window.to_dict() for window in report.critical_windows],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifacts.narrative_md.write_text(report.narrative, encoding="utf-8")
        return artifacts

"""Finalize complete-case scientific metrics from append-only observations."""

from __future__ import annotations

import re
from collections.abc import Mapping

from agentbench_hl.domain.events import FinalizedEvent
from agentbench_hl.domain.metrics import (
    EloResult,
    IterationMetrics,
    combine_usage,
    fit_anchored_elo,
)
from agentbench_hl.domain.models import Usage
from agentbench_hl.domain.policy import (
    DecisionSample,
    compare_decisions,
    occupancy_total_variation,
)
from agentbench_hl.ports.event_store import EventStore


class MetricsService:
    def __init__(
        self,
        *,
        event_store: EventStore,
        human_ratings: Mapping[str, float],
        expected_case_ids: tuple[str, ...],
        epsilon: float,
        probe_schema: str = "legacy",
    ) -> None:
        if not expected_case_ids or len(set(expected_case_ids)) != len(expected_case_ids):
            raise ValueError("expected metric cases must be non-empty and unique")
        self.event_store = event_store
        self.human_ratings = {str(key): float(value) for key, value in human_ratings.items()}
        self.expected_case_ids = expected_case_ids
        self.epsilon = float(epsilon)
        if not probe_schema:
            raise ValueError("probe schema must be non-empty")
        self.probe_schema = probe_schema

    def _append_idempotent(
        self,
        event_type: str,
        payload: Mapping[str, object],
        idempotency_key: str,
    ) -> bool:
        existing = next(
            (
                event
                for event in self.event_store.read_all()
                if event.idempotency_key == idempotency_key
            ),
            None,
        )
        if existing is not None:
            if existing.event_type == event_type and dict(existing.payload) == dict(payload):
                return False
            raise ValueError(f"conflicting idempotency key: {idempotency_key}")
        return self.event_store.append(
            FinalizedEvent.create(event_type, payload, idempotency_key=idempotency_key)
        )

    def _decision_event(self, candidate_id: str, sample: DecisionSample) -> FinalizedEvent:
        return FinalizedEvent.create(
            "DecisionSampleRecorded",
            {
                "candidate_id": candidate_id,
                "state_id": sample.state_id,
                "legal_actions": list(sample.legal_actions),
                "parent_action": sample.parent_action,
                "candidate_action": sample.candidate_action,
                "parent_occupancy": sample.parent_occupancy,
                "candidate_occupancy": sample.candidate_occupancy,
                "probe_schema": self.probe_schema,
            },
            idempotency_key=(
                f"decision-sample:{self.probe_schema}:{candidate_id}:{sample.state_id}"
            ),
        )

    def record_decision_sample(self, candidate_id: str, sample: DecisionSample) -> bool:
        return self.event_store.append_many((self._decision_event(candidate_id, sample),))[0]

    def record_decision_samples(
        self, candidate_id: str, samples: tuple[DecisionSample, ...]
    ) -> int:
        outcomes = self.event_store.append_many(
            tuple(self._decision_event(candidate_id, sample) for sample in samples)
        )
        return sum(outcomes)

    def record_match(
        self,
        *,
        candidate_id: str,
        case_id: str,
        opponent_id: str,
        role: str,
        status: str,
        points: float | None,
        score_margin: float | None,
        scope: str = "evaluation",
    ) -> bool:
        if status == "complete" and (points is None or score_margin is None):
            raise ValueError("complete metric match requires points and score margin")
        if status != "complete" and (points is not None or score_margin is not None):
            raise ValueError("incomplete metric match cannot contain strategy outcome")
        return self._append_idempotent(
            "MetricMatchRecorded",
            {
                "candidate_id": candidate_id,
                "case_id": case_id,
                "opponent_id": opponent_id,
                "role": role,
                "status": status,
                "points": points,
                "score_margin": score_margin,
            },
            f"metric-match:{scope}:{candidate_id}:{case_id}",
        )

    def record_occupancy_trace(
        self,
        *,
        candidate_id: str,
        case_id: str,
        parent_state_ids: tuple[str, ...],
        candidate_state_ids: tuple[str, ...],
    ) -> bool:
        if not parent_state_ids or not candidate_state_ids:
            raise ValueError("occupancy traces must contain decision states")
        return self._append_idempotent(
            "OccupancyTraceRecorded",
            {
                "candidate_id": candidate_id,
                "case_id": case_id,
                "parent_state_ids": list(parent_state_ids),
                "candidate_state_ids": list(candidate_state_ids),
                "probe_schema": self.probe_schema,
            },
            f"occupancy-trace:{self.probe_schema}:{candidate_id}:{case_id}",
        )

    def finalize_iteration(
        self,
        candidate_id: str,
        *,
        champion_id: str,
        research_iteration: int | None = None,
        learning_usage: Usage | None = None,
        evaluation_usage: Usage | None = None,
    ) -> IterationMetrics:
        match = re.fullmatch(r"v(\d+)", candidate_id)
        if match is None:
            raise ValueError("candidate_id must have vNNN form")
        iteration = int(match.group(1)) if research_iteration is None else int(research_iteration)
        if iteration < 0:
            raise ValueError("research_iteration cannot be negative")
        events = self.event_store.read_all()
        panel_prefixes = tuple(
            f"{candidate_id}-{opponent}-{role.lower()}-s{seed}:"
            for case_id in self.expected_case_ids
            for opponent, role, seed in (case_id.split(":"),)
        )

        def sample_is_in_panel(state_id: str) -> bool:
            if any(state_id.startswith(prefix) for prefix in panel_prefixes):
                return True
            return not state_id.startswith(f"{candidate_id}-")

        samples = tuple(
            DecisionSample(
                state_id=str(event.payload["state_id"]),
                legal_actions=tuple(str(item) for item in event.payload["legal_actions"]),
                parent_action=str(event.payload["parent_action"]),
                candidate_action=str(event.payload["candidate_action"]),
                parent_occupancy=str(event.payload["parent_occupancy"]),
                candidate_occupancy=str(event.payload["candidate_occupancy"]),
            )
            for event in events
            if event.event_type == "DecisionSampleRecorded"
            and event.payload.get("candidate_id") == candidate_id
            and event.payload.get("probe_schema", "legacy") == self.probe_schema
            and sample_is_in_panel(str(event.payload["state_id"]))
        )
        behavior = compare_decisions(samples, epsilon=self.epsilon)
        occupancy_events = tuple(
            event
            for event in events
            if event.event_type == "OccupancyTraceRecorded"
            and event.payload.get("candidate_id") == candidate_id
            and event.payload.get("probe_schema", "legacy") == self.probe_schema
            and event.payload.get("case_id") in self.expected_case_ids
        )
        parent_occupancy = tuple(
            str(state_id)
            for event in occupancy_events
            for state_id in event.payload["parent_state_ids"]
        )
        candidate_occupancy = tuple(
            str(state_id)
            for event in occupancy_events
            for state_id in event.payload["candidate_state_ids"]
        )
        occupancy_shift = occupancy_total_variation(
            parent_occupancy,
            candidate_occupancy,
        )
        matches = {
            str(event.payload["case_id"]): event.payload
            for event in events
            if event.event_type == "MetricMatchRecorded"
            and event.payload.get("candidate_id") == candidate_id
        }
        complete_panel = set(self.expected_case_ids).issubset(matches) and all(
            matches[case_id]["status"] == "complete" for case_id in self.expected_case_ids
        )
        elo = None
        win_rate = None
        margin = None
        if complete_panel:
            ordered = tuple(matches[case_id] for case_id in self.expected_case_ids)
            elo_results = tuple(
                EloResult(
                    candidate_id=candidate_id,
                    opponent_id=str(item["opponent_id"]),
                    role=str(item["role"]),
                    points=float(item["points"]),
                )
                for item in ordered
            )
            elo = fit_anchored_elo(elo_results, self.human_ratings)
            win_rate = sum(float(item["points"]) for item in ordered) / len(ordered)
            margin = sum(float(item["score_margin"]) for item in ordered) / len(ordered)
        learning = learning_usage or Usage()
        evaluation = evaluation_usage or Usage()
        return IterationMetrics(
            research_iteration=iteration,
            candidate_id=candidate_id,
            champion_id=champion_id,
            behavioral_ig=behavior.mean_kl_nats,
            occupancy_shift=(
                behavior.occupancy_shift if occupancy_shift is None else occupancy_shift
            ),
            action_disagreement=behavior.disagreement_rate,
            elo=elo,
            win_rate=win_rate,
            score_margin=margin,
            learning_usage=learning,
            evaluation_usage=evaluation,
            total_usage=combine_usage(learning, evaluation),
        )

"""Four primary scientific curves plus separate resource reporting."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

from agentbench_hl.domain.metrics import IterationMetrics

_PANELS = ("Behavioral IG", "Occupancy shift", "Fixed-pool Elo", "Win rate")


@dataclass(frozen=True)
class CurveArtifacts:
    primary_png: Path
    score_margin_png: Path
    resources_png: Path
    csv: Path
    panel_names: tuple[str, ...]


def build_curves(
    rows: tuple[IterationMetrics, ...],
    output_dir: str | Path,
) -> CurveArtifacts:
    if not rows:
        raise ValueError("curve rows cannot be empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    ordered = tuple(sorted(rows, key=lambda item: item.research_iteration))
    iterations = [item.research_iteration for item in ordered]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in iterations):
        raise ValueError("research iteration must be integer")
    table = [item.to_row() for item in ordered]
    csv_path = output / "iteration-metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    primary = output / "primary-curves.png"
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    series = (
        ("behavioral_ig", "Mean KL (nats / decision)"),
        ("occupancy_shift", "State occupancy total variation"),
        ("fixed_pool_elo", "Elo"),
        ("win_rate", "Win rate"),
    )
    for axis, title, (key, ylabel) in zip(axes.flat, _PANELS, series, strict=True):
        axis.plot(iterations, [row[key] for row in table], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Research iteration")
        axis.set_ylabel(ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
    figure.savefig(primary, dpi=180)
    plt.close(figure)

    score_margin = output / "score-margin.png"
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot(iterations, [row["score_margin"] for row in table], marker="o")
    axis.set_title("Score margin")
    axis.set_xlabel("Research iteration")
    axis.set_ylabel("Terminal base-HP margin")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    figure.savefig(score_margin, dpi=180)
    plt.close(figure)

    resources = output / "resource-curves.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, key, title, ylabel in (
        (axes[0], "total_tokens", "API token usage", "Tokens"),
        (axes[1], "total_wall_time_s", "Wall time", "Seconds"),
    ):
        axis.plot(iterations, [row[key] for row in table], marker="o")
        axis.set_title(title)
        axis.set_xlabel("Research iteration")
        axis.set_ylabel(ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
    figure.savefig(resources, dpi=180)
    plt.close(figure)
    return CurveArtifacts(primary, score_margin, resources, csv_path, _PANELS)

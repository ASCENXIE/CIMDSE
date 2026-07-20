"""Draw the full-ResNet-18 Tile DSE energy-latency Pareto curve."""

from __future__ import annotations

import csv
import os
from bisect import bisect_left
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/accelforge_mplconfig")

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results" / "tile_dse_aggregate.csv"
FRONT_CSV = ROOT / "experiments" / "results" / "tile_dse_full_resnet_energy_latency_pareto.csv"
FIGURE = ROOT / "experiments" / "results" / "tile_dse_full_resnet_pareto.png"
FIGURE_PDF = ROOT / "experiments" / "results" / "tile_dse_full_resnet_pareto.pdf"
ARCH_SUMMARY_CSV = ROOT / "experiments" / "results" / "tile_dse_architecture_summary.csv"
ARCH_FIGURE = ROOT / "experiments" / "results" / "tile_dse_architecture_summary.png"
ARCH_FIGURE_PDF = ROOT / "experiments" / "results" / "tile_dse_architecture_summary.pdf"
SAMPLES = ROOT / "experiments" / "results" / "tile_dse_bayesian_samples.csv"
MAPPING_COLORS = {
    "k_parallel": "#c75b39",
    "n_parallel": "#2e86ab",
    "mixed": "#3b9d6d",
}
NEAR_FRONT_ENERGY_MARGIN = 0.15


def read_rows() -> list[dict[str, str | float]]:
    string_fields = {
        "design",
        "tile_class",
        "mapping_summary",
        "macro_counts",
        "schedule_trace",
    }
    with RESULTS.open(newline="") as file:
        return [
            {
                key: value
                if key in string_fields
                else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        ]


def energy_latency_front(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    ordered = sorted(rows, key=lambda row: (float(row["latency_ms"]), float(row["fabric_energy_mJ"])))
    front = []
    best_energy = float("inf")
    for row in ordered:
        energy = float(row["fabric_energy_mJ"])
        if energy < best_energy:
            front.append(row)
            best_energy = energy
    return front


def near_front_rows(
    rows: list[dict[str, str | float]], front: list[dict[str, str | float]]
) -> list[dict[str, str | float]]:
    """Keep dominated schedules within a relative energy band over the E-T front."""
    front_latency = [float(row["latency_ms"]) for row in front]
    front_energy = [float(row["fabric_energy_mJ"]) for row in front]
    nearby = []
    for row in rows:
        latency = float(row["latency_ms"])
        if latency < front_latency[0] or latency > front_latency[-1]:
            continue
        upper_index = bisect_left(front_latency, latency)
        if upper_index == 0:
            boundary_energy = front_energy[0]
        elif upper_index == len(front):
            boundary_energy = front_energy[-1]
        else:
            lower_index = upper_index - 1
            span = front_latency[upper_index] - front_latency[lower_index]
            ratio = (latency - front_latency[lower_index]) / span if span else 0.0
            boundary_energy = front_energy[lower_index] + ratio * (
                front_energy[upper_index] - front_energy[lower_index]
            )
        energy = float(row["fabric_energy_mJ"])
        if boundary_energy <= energy <= boundary_energy * (1 + NEAR_FRONT_ENERGY_MARGIN):
            nearby.append(row)
    return nearby


def dominant_mapping(row: dict[str, str | float]) -> str:
    counts = {
        key: int(value)
        for key, value in (
            item.split("=") for item in str(row["mapping_summary"]).split(";")
        )
    }
    names = {"K": "k_parallel", "N": "n_parallel", "M": "mixed"}
    return names[max(counts, key=counts.get)]


def format_design(row: dict[str, str | float]) -> str:
    parts = [part.replace("xm", " x ").replace("x8", "") for part in str(row["macro_counts"]).split("+")]
    first_line = " + ".join(parts[:2])
    return first_line if len(parts) <= 2 else f"{first_line}\n+ {' + '.join(parts[2:])}"


def extreme_points(front: list[dict[str, str | float]]) -> dict[str, dict[str, str | float]]:
    fastest = min(front, key=lambda row: float(row["latency_ms"]))
    lowest_energy = min(front, key=lambda row: float(row["fabric_energy_mJ"]))
    min_latency = float(fastest["latency_ms"])
    max_latency = max(float(row["latency_ms"]) for row in front)
    min_energy = float(lowest_energy["fabric_energy_mJ"])
    max_energy = max(float(row["fabric_energy_mJ"]) for row in front)
    knee = min(
        front,
        key=lambda row: (
            (float(row["latency_ms"]) - min_latency) / (max_latency - min_latency)
        ) ** 2
        + ((float(row["fabric_energy_mJ"]) - min_energy) / (max_energy - min_energy)) ** 2,
    )
    return {"Lowest latency": fastest, "Lowest energy": lowest_energy, "Knee": knee}


def write_front(rows: list[dict[str, str | float]]) -> None:
    with FRONT_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def search_coverage() -> str:
    if not SAMPLES.is_file():
        return "exhaustive search"
    with SAMPLES.open(newline="") as file:
        rows = list(csv.DictReader(file))
    evaluated = len(rows)
    candidate_pool_size = int(rows[0].get("candidate_pool_size", 309)) if rows else 309
    settings = ""
    if rows and "latency_cap_ratio" in rows[0]:
        settings = (
            f", latency cap {float(rows[0]['latency_cap_ratio']):g}x"
            f", local epsilon {float(rows[0]['local_pareto_epsilon']):.1%}"
        )
    return (
        f"Bayesian search, {evaluated}/{candidate_pool_size} Tile designs evaluated"
        f"{settings}"
    )


def baseline(rows: list[dict[str, str | float]], design: str):
    return min(
        (row for row in rows if row["design"] == design),
        key=lambda row: float(row["fabric_energy_mJ"]),
    )


def architecture_summary(
    rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    grouped: dict[str, list[dict[str, str | float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["design"]), []).append(row)
    summary = []
    for design, candidates in grouped.items():
        lowest_energy = min(candidates, key=lambda row: float(row["fabric_energy_mJ"]))
        lowest_latency = min(candidates, key=lambda row: float(row["latency_ms"]))
        min_energy = float(lowest_energy["fabric_energy_mJ"])
        max_energy = max(float(row["fabric_energy_mJ"]) for row in candidates)
        min_latency = float(lowest_latency["latency_ms"])
        max_latency = max(float(row["latency_ms"]) for row in candidates)
        energy_span = max(max_energy - min_energy, 1e-15)
        latency_span = max(max_latency - min_latency, 1e-15)
        knee = min(
            candidates,
            key=lambda row: (
                (float(row["latency_ms"]) - min_latency) / latency_span
            )
            ** 2
            + (
                (float(row["fabric_energy_mJ"]) - min_energy) / energy_span
            )
            ** 2,
        )
        summary.append(
            {
                "design": design,
                "tile_class": lowest_energy["tile_class"],
                "macro_counts": lowest_energy["macro_counts"],
                "schedule_count": len(candidates),
                "min_energy_mJ": min_energy,
                "min_energy_latency_ms": float(lowest_energy["latency_ms"]),
                "min_latency_mJ": float(lowest_latency["fabric_energy_mJ"]),
                "min_latency_ms": min_latency,
                "knee_energy_mJ": float(knee["fabric_energy_mJ"]),
                "knee_latency_ms": float(knee["latency_ms"]),
            }
        )
    return sorted(summary, key=lambda row: str(row["design"]))


def write_architecture_summary(rows: list[dict[str, str | float]]) -> None:
    with ARCH_SUMMARY_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def draw_architecture_summary(rows: list[dict[str, str | float]]) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 7))
    class_colors = {"homogeneous": "#c75b39", "heterogeneous": "#2e86ab"}
    lowest_energy = min(rows, key=lambda row: float(row["min_energy_mJ"]))
    lowest_latency = min(rows, key=lambda row: float(row["min_latency_ms"]))
    min_energy = float(lowest_energy["knee_energy_mJ"])
    max_energy = max(float(row["knee_energy_mJ"]) for row in rows)
    min_latency = float(lowest_latency["knee_latency_ms"])
    max_latency = max(float(row["knee_latency_ms"]) for row in rows)
    knee = min(
        rows,
        key=lambda row: (
            (float(row["knee_energy_mJ"]) - min_energy)
            / max(max_energy - min_energy, 1e-15)
        )
        ** 2
        + (
            (float(row["knee_latency_ms"]) - min_latency)
            / max(max_latency - min_latency, 1e-15)
        )
        ** 2,
    )
    labeled_designs = {
        str(lowest_energy["design"]),
        str(lowest_latency["design"]),
        str(knee["design"]),
        "9xm64x8",
        "1xm64x8+1xm512x8",
        *(
            str(row["design"])
            for row in rows
            if row["tile_class"] == "homogeneous"
        ),
    }
    homogeneous_order = {
        str(row["design"]): index
        for index, row in enumerate(
            sorted(
                (row for row in rows if row["tile_class"] == "homogeneous"),
                key=lambda row: float(row["min_energy_latency_ms"]),
            )
        )
    }
    homogeneous_offsets = ((6, -16), (6, -30), (6, -44))
    for row in rows:
        color = class_colors[str(row["tile_class"])]
        energy_point = (
            float(row["min_energy_latency_ms"]),
            float(row["min_energy_mJ"]),
        )
        latency_point = (
            float(row["min_latency_ms"]),
            float(row["min_latency_mJ"]),
        )
        same_point = energy_point == latency_point
        axis.scatter(
            *energy_point,
            marker="D" if same_point else "o",
            s=76 if same_point else 54,
            color=color,
            edgecolor="#202020",
            linewidth=0.7,
            zorder=3,
        )
        if not same_point:
            axis.scatter(
                *latency_point,
                marker="^",
                s=62,
                color=color,
                edgecolor="#202020",
                linewidth=0.7,
                zorder=3,
            )
        if row["design"] in labeled_designs:
            offset = (
                homogeneous_offsets[
                    homogeneous_order[str(row["design"])] % len(homogeneous_offsets)
                ]
                if row["tile_class"] == "homogeneous"
                else (6, 6)
            )
            axis.annotate(
                format_design(row),
                energy_point,
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
            )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="#202020",
            label="minimum-energy schedule",
        ),
        Line2D(
            [],
            [],
            marker="^",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="#202020",
            label="minimum-latency schedule",
        ),
        Line2D(
            [],
            [],
            marker="D",
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="#202020",
            label="same schedule is both",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="none",
            markerfacecolor=class_colors["homogeneous"],
            label="homogeneous Tile",
        ),
        Line2D(
            [],
            [],
            marker="s",
            color="none",
            markerfacecolor=class_colors["heterogeneous"],
            label="heterogeneous Tile",
        ),
    ]
    axis.legend(handles=handles, frameon=False)
    axis.set_title(
        "Full ResNet-18 Tile Architecture Summary\n"
        f"{search_coverage()}"
    )
    axis.set_xlabel("sequential Macro VMM latency (ms)")
    axis.set_ylabel("Tile fabric energy (mJ)")
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(ARCH_FIGURE, dpi=180)
    figure.savefig(ARCH_FIGURE_PDF)
    plt.close(figure)


def main() -> None:
    rows = read_rows()
    front = energy_latency_front(rows)
    nearby = near_front_rows(rows, front)
    extremes = extreme_points(front)
    summary = architecture_summary(rows)
    write_front(front)
    write_architecture_summary(summary)
    draw_architecture_summary(summary)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11.5, 7))
    axis.scatter(
        [float(row["latency_ms"]) for row in nearby],
        [float(row["fabric_energy_mJ"]) for row in nearby],
        color="#b8b8b8",
        s=9,
        alpha=0.28,
        linewidths=0,
        label=(
            "near-front feasible schedules "
            f"(within {NEAR_FRONT_ENERGY_MARGIN:.0%} energy)"
        ),
        zorder=1,
    )
    for mapping, color in MAPPING_COLORS.items():
        candidates = [row for row in front if dominant_mapping(row) == mapping]
        axis.scatter(
            [float(row["latency_ms"]) for row in candidates],
            [float(row["fabric_energy_mJ"]) for row in candidates],
            color=color,
            s=24,
            alpha=0.82,
            linewidths=0,
            label=f"predominant {mapping.replace('_', ' ')}",
            zorder=2,
        )
    extreme_offsets = {
        "Lowest latency": (10, -28),
        "Lowest energy": (-112, 10),
        "Knee": (10, 8),
    }
    for label, row in extremes.items():
        axis.scatter(
            float(row["latency_ms"]),
            float(row["fabric_energy_mJ"]),
            marker="D",
            s=72,
            facecolor="#f4f4f4",
            edgecolor="#202020",
            linewidth=1.2,
            zorder=4,
        )
        axis.annotate(
            f"{label}\n{format_design(row)}",
            (float(row["latency_ms"]), float(row["fabric_energy_mJ"])),
            xytext=extreme_offsets[label],
            textcoords="offset points",
            fontsize=8,
            zorder=5,
        )
    anchors = (
        ("9xm64x8", "9 x 64"),
        ("1xm64x8+1xm512x8", "64 + 512"),
    )
    anchor_offsets = {"9xm64x8": (7, -18), "1xm64x8+1xm512x8": (7, -16)}
    for design, label in anchors:
        row = baseline(rows, design)
        axis.scatter(
            float(row["latency_ms"]),
            float(row["fabric_energy_mJ"]),
            marker="X",
            s=100,
            color="#202020",
            zorder=3,
        )
        axis.annotate(
            label,
            (float(row["latency_ms"]), float(row["fabric_energy_mJ"])),
            xytext=anchor_offsets[design],
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_title(
        "Full ResNet-18 Tile DSE: Energy-Latency Pareto Front\n"
        f"{search_coverage()}"
    )
    axis.set_xlabel("sequential Macro VMM latency (ms)")
    axis.set_ylabel("Tile fabric energy (mJ)")
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.text(
        0.5,
        0.01,
        "Each marker is a discrete feasible Tile schedule; no line is drawn between Pareto points.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(FIGURE, dpi=180)
    figure.savefig(FIGURE_PDF)
    plt.close(figure)
    print(f"Energy-latency Pareto points: {len(front)}")
    print(FRONT_CSV)
    print(FIGURE)
    print(FIGURE_PDF)
    print(ARCH_SUMMARY_CSV)
    print(ARCH_FIGURE)
    print(ARCH_FIGURE_PDF)


if __name__ == "__main__":
    main()

"""Draw full-network Tile DSE energy-latency Pareto visualizations."""

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


ROOT = Path(__file__).resolve().parents[3]
RESULT_PREFIX = os.environ.get("TILE_DSE_RESULT_PREFIX", "tile_window_dse")
RESULT_SUBDIR = os.environ.get("TILE_DSE_RESULT_SUBDIR", "")
WORKLOAD_LABEL = os.environ.get("TILE_DSE_WORKLOAD_LABEL", "ResNet-18")
RESULT_DIR = ROOT / "experiments" / "results" / "tile_DSE" / RESULT_SUBDIR
RESULTS = RESULT_DIR / f"{RESULT_PREFIX}_aggregate.csv"
ETA_RESULTS = RESULT_DIR / f"{RESULT_PREFIX}_aggregate_pareto_eta.csv"
FRONT_CSV = RESULT_DIR / f"{RESULT_PREFIX}_full_resnet_energy_latency_pareto.csv"
FIGURE = RESULT_DIR / f"{RESULT_PREFIX}_full_resnet_pareto.png"
FIGURE_PDF = RESULT_DIR / f"{RESULT_PREFIX}_full_resnet_pareto.pdf"
ETA_FIGURE = RESULT_DIR / f"{RESULT_PREFIX}_energy_latency_area_pareto.png"
ETA_FIGURE_PDF = RESULT_DIR / f"{RESULT_PREFIX}_energy_latency_area_pareto.pdf"
ARCH_SUMMARY_CSV = RESULT_DIR / f"{RESULT_PREFIX}_architecture_summary.csv"
ARCH_FIGURE = RESULT_DIR / f"{RESULT_PREFIX}_architecture_summary.png"
ARCH_FIGURE_PDF = RESULT_DIR / f"{RESULT_PREFIX}_architecture_summary.pdf"
EDP_AREA_CSV = RESULT_DIR / f"{RESULT_PREFIX}_edp_area_pareto.csv"
EDP_AREA_FIGURE = RESULT_DIR / f"{RESULT_PREFIX}_edp_area_pareto.png"
EDP_AREA_FIGURE_PDF = RESULT_DIR / f"{RESULT_PREFIX}_edp_area_pareto.pdf"
EDAP_RANKING_CSV = RESULT_DIR / f"{RESULT_PREFIX}_edap_ranking.csv"
EDAP_RANKING_FIGURE = RESULT_DIR / f"{RESULT_PREFIX}_edap_ranking.png"
EDAP_RANKING_FIGURE_PDF = RESULT_DIR / f"{RESULT_PREFIX}_edap_ranking.pdf"
SAMPLES = RESULT_DIR / f"{RESULT_PREFIX}_bayesian_samples.csv"
MAPPING_COLORS = {
    "k_parallel": "#c75b39",
    "n_parallel": "#2e86ab",
    "mixed": "#3b9d6d",
}
NEAR_FRONT_ENERGY_MARGIN = 0.15


def read_rows(path: Path = RESULTS) -> list[dict[str, str | float]]:
    string_fields = {
        "design",
        "tile_class",
        "mapping_summary",
        "macro_counts",
        "schedule_trace",
        "workload_name",
        "tile_latency_batch_granularity",
    }
    with path.open(newline="") as file:
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
        if "local_state_cap" in rows[0]:
            settings += f", local state cap {int(rows[0]['local_state_cap'])}"
    fixed_panel = rows and all(
        row.get("selection_stage") == "fixed_panel" for row in rows
    )
    if fixed_panel:
        search_kind = "Bayesian-selected fixed panel"
    else:
        search_kind = (
            "exhaustive search" if evaluated == candidate_pool_size else "Bayesian search"
        )
    return (
        f"{search_kind}, {evaluated}/{candidate_pool_size} Tile designs evaluated"
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
    label_offsets = {
        str(lowest_energy["design"]): (-98, 18),
        str(lowest_latency["design"]): (8, -20),
        str(knee["design"]): (8, 32),
        "9xm64x8": (8, 10),
        "1xm64x8+1xm512x8": (8, 18),
    }
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
        if row["design"] in label_offsets:
            axis.annotate(
                format_design(row),
                energy_point,
                xytext=label_offsets[str(row["design"])],
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
        f"Full {WORKLOAD_LABEL} Tile Architecture Summary\n"
        f"{search_coverage()}"
    )
    axis.set_xlabel("modeled Tile latency (ms)")
    axis.set_ylabel("Tile fabric energy (mJ)")
    axis.margins(x=0.04, y=0.12)
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(ARCH_FIGURE, dpi=180)
    figure.savefig(ARCH_FIGURE_PDF)
    plt.close(figure)


def minimum_edp_by_architecture(
    rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    grouped: dict[str, list[dict[str, str | float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["design"]), []).append(row)
    selected = []
    for design, candidates in grouped.items():
        row = min(
            candidates,
            key=lambda item: float(item["fabric_energy_mJ"])
            * float(item["latency_ms"]),
        )
        selected.append(
            {
                "design": design,
                "tile_class": row["tile_class"],
                "macro_counts": row["macro_counts"],
                "tile_area_mm2": float(row["tile_area_mm2"]),
                "fabric_energy_mJ": float(row["fabric_energy_mJ"]),
                "latency_ms": float(row["latency_ms"]),
                "EDP_mJ_ms": float(row["fabric_energy_mJ"])
                * float(row["latency_ms"]),
            }
        )
    return sorted(selected, key=lambda row: str(row["design"]))


def edp_area_front(
    rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["tile_area_mm2"]),
            float(row["EDP_mJ_ms"]),
        ),
    )
    front = []
    best_edp = float("inf")
    for row in ordered:
        edp = float(row["EDP_mJ_ms"])
        if edp < best_edp:
            front.append(row)
            best_edp = edp
    return front


def draw_edp_area(rows: list[dict[str, str | float]]) -> None:
    front = edp_area_front(rows)
    front_designs = {str(row["design"]) for row in front}
    class_styles = {
        "homogeneous": ("o", "#2e86ab"),
        "heterogeneous": ("s", "#c75b39"),
    }
    figure, axis = plt.subplots(figsize=(9.2, 6.2))
    for tile_class, (marker, color) in class_styles.items():
        for on_front, size, alpha, edgecolor, linewidth, zorder in (
            (False, 48, 0.48, "none", 0.0, 1),
            (True, 86, 0.96, "#202020", 1.0, 3),
        ):
            subset = [
                row
                for row in rows
                if row["tile_class"] == tile_class
                and (str(row["design"]) in front_designs) == on_front
            ]
            axis.scatter(
                [float(row["tile_area_mm2"]) for row in subset],
                [float(row["EDP_mJ_ms"]) for row in subset],
                marker=marker,
                color=color,
                s=size,
                alpha=alpha,
                edgecolors=edgecolor,
                linewidths=linewidth,
                zorder=zorder,
            )
    label_offsets = {
        "15xm32x8": (8, 9),
        "4xm128x8": (8, 20),
        "2xm256x8": (-62, -28),
        "1xm32x8+2xm256x8": (8, 12),
        "1xm64x8+2xm256x8": (-100, 20),
        "2xm32x8+1xm512x8": (-108, -2),
        "1xm64x8+1xm512x8": (8, -14),
        "1xm128x8+1xm512x8": (8, 12),
        "5xm32x8+1xm512x8": (-104, -2),
    }
    for row in front:
        design = str(row["design"])
        axis.annotate(
            format_design(row).replace("\n", " "),
            (float(row["tile_area_mm2"]), float(row["EDP_mJ_ms"])),
            xytext=label_offsets.get(design, (8, 8)),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#6b6b6b", "lw": 0.6},
            zorder=4,
        )
    handles = [
        Line2D([], [], marker="o", linestyle="", color="#2e86ab", label="Homogeneous"),
        Line2D([], [], marker="s", linestyle="", color="#c75b39", label="Heterogeneous"),
        Line2D(
            [], [], marker="o", linestyle="", markerfacecolor="none",
            markeredgecolor="#202020", label="EDP-Area Pareto"
        ),
    ]
    axis.set_title(
        f"{WORKLOAD_LABEL} Tile EDP versus Area\n{search_coverage()}"
    )
    axis.set_title(
        f"{WORKLOAD_LABEL} Tile EDP versus Area\n"
        "48/138 evaluated - fixed Bayesian panel"
    )
    axis.set_xlabel(r"Tile area (mm$^2$)")
    axis.set_ylabel(r"Energy-delay product (mJ$\cdot$ms)")
    axis.set_ylim(bottom=3.65)
    axis.grid(alpha=0.24)
    axis.set_axisbelow(True)
    axis.legend(handles=handles, frameon=False)
    figure.tight_layout()
    figure.savefig(EDP_AREA_FIGURE, dpi=220)
    figure.savefig(EDP_AREA_FIGURE_PDF)
    plt.close(figure)
    with EDP_AREA_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=front[0].keys())
        writer.writeheader()
        writer.writerows(front)


def minimum_edap_by_architecture(
    rows: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    grouped: dict[str, list[dict[str, str | float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["design"]), []).append(row)
    selected = []
    for design, candidates in grouped.items():
        row = min(
            candidates,
            key=lambda item: float(item["fabric_energy_mJ"])
            * float(item["latency_ms"])
            * float(item["tile_area_mm2"]),
        )
        edap = (
            float(row["fabric_energy_mJ"])
            * float(row["latency_ms"])
            * float(row["tile_area_mm2"])
        )
        selected.append(
            {
                "rank": 0,
                "design": design,
                "tile_class": row["tile_class"],
                "macro_counts": row["macro_counts"],
                "tile_area_mm2": float(row["tile_area_mm2"]),
                "fabric_energy_mJ": float(row["fabric_energy_mJ"]),
                "latency_ms": float(row["latency_ms"]),
                "EDAP_mJ_ms_mm2": edap,
            }
        )
    selected.sort(key=lambda row: (float(row["EDAP_mJ_ms_mm2"]), str(row["design"])))
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def draw_edap_ranking(rows: list[dict[str, str | float]]) -> None:
    ordered = rows[:10]
    colors = [
        "#2e86ab"
        if row["tile_class"] == "homogeneous"
        else "#c75b39"
        for row in ordered
    ]
    labels = [
        f"{int(row['rank'])}. {format_design(row).replace(chr(10), ' ')}"
        for row in ordered
    ]
    values = [float(row["EDAP_mJ_ms_mm2"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(11.5, 6.5))
    axis.barh(range(len(ordered)), values, color=colors, alpha=0.88, height=0.72)
    axis.set_yticks(range(len(ordered)), labels)
    axis.invert_yaxis()
    axis.set_xlabel(r"Energy $\times$ Delay $\times$ Area (mJ$\cdot$ms$\cdot$mm$^2$)")
    axis.set_ylabel("Tile architecture rank (best at top)")
    axis.set_title(
        f"{WORKLOAD_LABEL} Tile Energy-Delay-Area Top 10\n"
        "48/138 evaluated - fixed Bayesian panel"
    )
    axis.grid(axis="x", alpha=0.24)
    axis.set_axisbelow(True)
    handles = [
        Line2D([], [], marker="s", linestyle="", color="#2e86ab", label="Homogeneous"),
        Line2D([], [], marker="s", linestyle="", color="#c75b39", label="Heterogeneous"),
    ]
    axis.legend(handles=handles, frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(EDAP_RANKING_FIGURE, dpi=220)
    figure.savefig(EDAP_RANKING_FIGURE_PDF)
    plt.close(figure)
    with EDAP_RANKING_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ordered[0].keys())
        writer.writeheader()
        writer.writerows(ordered)


def draw_energy_latency_area_front(
    rows: list[dict[str, str | float]],
) -> None:
    """Draw the actual three-objective front and an area-colored E-T projection."""

    latency = [float(row["latency_ms"]) for row in rows]
    energy = [float(row["fabric_energy_mJ"]) for row in rows]
    area = [float(row["tile_area_mm2"]) for row in rows]
    figure = plt.figure(figsize=(14, 6.8))
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_2d = figure.add_subplot(1, 2, 2)

    axis_3d.scatter(
        latency,
        energy,
        area,
        c=area,
        cmap="viridis",
        s=10,
        alpha=0.48,
        linewidths=0,
        depthshade=False,
    )
    axis_3d.set_xlabel("latency (ms)")
    axis_3d.set_ylabel("energy (mJ)")
    axis_3d.set_zlabel("Tile area (mm2)")
    axis_3d.set_title("Three-objective Pareto front")
    axis_3d.view_init(elev=24, azim=-58)

    projection = axis_2d.scatter(
        latency,
        energy,
        c=area,
        cmap="viridis",
        s=11,
        alpha=0.58,
        linewidths=0,
    )
    axis_2d.set_xlabel("modeled Tile latency (ms)")
    axis_2d.set_ylabel("Tile fabric energy (mJ)")
    axis_2d.set_title("Energy-latency projection colored by area")
    axis_2d.grid(alpha=0.25)
    axis_2d.set_axisbelow(True)
    colorbar = figure.colorbar(projection, ax=axis_2d, pad=0.02)
    colorbar.set_label("Tile area (mm2)")

    figure.suptitle(
        f"Full {WORKLOAD_LABEL} Tile DSE: Energy-Latency-Area Pareto\n"
        f"{search_coverage()}",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(ETA_FIGURE, dpi=180)
    figure.savefig(ETA_FIGURE_PDF)
    plt.close(figure)


def main() -> None:
    rows = read_rows()
    eta_rows = read_rows(ETA_RESULTS)
    front = energy_latency_front(rows)
    nearby = near_front_rows(rows, front)
    extremes = extreme_points(front)
    summary = architecture_summary(rows)
    edp_area_rows = minimum_edp_by_architecture(rows)
    edap_rows = minimum_edap_by_architecture(rows)
    write_front(front)
    write_architecture_summary(summary)
    draw_architecture_summary(summary)
    draw_edp_area(edp_area_rows)
    draw_edap_ranking(edap_rows)
    draw_energy_latency_area_front(eta_rows)
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
    extreme_designs = {str(row["design"]) for row in extremes.values()}
    for design, label in anchors:
        if design in extreme_designs:
            continue
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
        f"Full {WORKLOAD_LABEL} Tile DSE: Energy-Latency Pareto Front\n"
        f"{search_coverage()}"
    )
    axis.set_xlabel("modeled Tile latency (ms)")
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
    print(f"EDP-area Pareto points: {len(edp_area_front(edp_area_rows))}")
    print(EDP_AREA_CSV)
    print(EDP_AREA_FIGURE)
    print(EDP_AREA_FIGURE_PDF)
    print(EDAP_RANKING_CSV)
    print(EDAP_RANKING_FIGURE)
    print(EDAP_RANKING_FIGURE_PDF)
    print(f"Energy-latency-area Pareto points: {len(eta_rows)}")
    print(ETA_FIGURE)
    print(ETA_FIGURE_PDF)


if __name__ == "__main__":
    main()

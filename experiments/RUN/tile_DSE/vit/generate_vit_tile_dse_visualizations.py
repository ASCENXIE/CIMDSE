"""Generate publication-ready visualizations for a ViT Tile DSE."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable


os.environ.setdefault("MPLCONFIGDIR", "/tmp/accelforge_mplconfig")

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULT_DIR = (
    ROOT / "experiments" / "results" / "tile_DSE" / "vision_transformer_l16_v2"
)
PREFIX = "vit_l16_pipeline_tile_dse"
TARGET_DESIGN = "1xm64x8+1xm512x8"
CLASS_COLORS = {"homogeneous": "#d97706", "heterogeneous": "#2474a6"}
ACCENT = "#b42318"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="directory containing the completed ViT DSE CSV files",
    )
    parser.add_argument(
        "--prefix",
        default=PREFIX,
        help="input and output filename prefix",
    )
    parser.add_argument(
        "--workload-label",
        default="ViT-L/16",
        help="workload name used in figure titles",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required DSE result is missing: {path}")
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def pareto_rows(
    rows: list[dict[str, str]], objectives: tuple[str, ...]
) -> list[dict[str, str]]:
    """Return exact nondominated mappings for minimization objectives."""

    front = []
    for candidate in rows:
        values = tuple(value(candidate, name) for name in objectives)
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_values = tuple(value(other, name) for name in objectives)
            if all(left <= right for left, right in zip(other_values, values)) and any(
                left < right for left, right in zip(other_values, values)
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: tuple(value(row, name) for name in objectives))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict[str, str], name: str) -> float:
    return float(row[name])


def edp(row: dict[str, str]) -> float:
    return value(row, "fabric_energy_mJ") * value(row, "latency_ms")


def design_label(design: str) -> str:
    return design.replace("xm", "x").replace("x8", "")


def best(rows: Iterable[dict[str, str]], objective) -> dict[str, str]:
    return min(rows, key=objective)


def group_by_design(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["design"], []).append(row)
    return groups


def coverage_text(samples: list[dict[str, str]]) -> str:
    completed = sum(row["evaluation_status"] == "completed" for row in samples)
    timed_out = sum(row["evaluation_status"] == "timeout_skipped" for row in samples)
    complexity_skipped = sum(
        row["evaluation_status"] == "complexity_skipped" for row in samples
    )
    timeout = float(samples[0]["design_timeout_s"]) if samples else 0.0
    budget = int(samples[0]["partition_budget"]) if samples else 0
    return (
        f"fixed panel: {completed}/{len(samples)} completed, {timed_out} timed out, "
        f"{complexity_skipped} complexity-skipped; {timeout:g} s timeout, "
        f"partition budget {budget}"
    )


def architecture_summary(
    rows: list[dict[str, str]],
    el_front: list[dict[str, str]],
    eta_front: list[dict[str, str]],
) -> list[dict[str, str | float | int]]:
    el_designs = {row["design"] for row in el_front}
    eta_designs = {row["design"] for row in eta_front}
    groups = group_by_design(rows)
    global_energy = min(value(row, "fabric_energy_mJ") for row in rows)
    global_latency = min(value(row, "latency_ms") for row in rows)
    global_edp = min(edp(row) for row in rows)
    summary: list[dict[str, str | float | int]] = []

    for design, candidates in groups.items():
        energy_row = best(candidates, lambda row: value(row, "fabric_energy_mJ"))
        latency_row = best(candidates, lambda row: value(row, "latency_ms"))
        edp_row = best(candidates, edp)
        summary.append(
            {
                "design": design,
                "tile_class": edp_row["tile_class"],
                "tile_area_mm2": value(edp_row, "tile_area_mm2"),
                "schedule_count": len(candidates),
                "min_energy_mJ": value(energy_row, "fabric_energy_mJ"),
                "min_energy_latency_ms": value(energy_row, "latency_ms"),
                "min_latency_ms": value(latency_row, "latency_ms"),
                "min_latency_energy_mJ": value(latency_row, "fabric_energy_mJ"),
                "min_edp_mJ_ms": edp(edp_row),
                "min_edp_energy_mJ": value(edp_row, "fabric_energy_mJ"),
                "min_edp_latency_ms": value(edp_row, "latency_ms"),
                "min_edp_assignment": edp_row["bank_assignment"],
                "energy_regret_pct": 100.0
                * (value(energy_row, "fabric_energy_mJ") / global_energy - 1.0),
                "latency_regret_pct": 100.0
                * (value(latency_row, "latency_ms") / global_latency - 1.0),
                "edp_regret_pct": 100.0 * (edp(edp_row) / global_edp - 1.0),
                "on_energy_latency_pareto": int(design in el_designs),
                "on_energy_latency_area_pareto": int(design in eta_designs),
            }
        )

    ordered = sorted(summary, key=lambda row: float(row["min_edp_mJ_ms"]))
    for rank, row in enumerate(ordered, start=1):
        row["min_edp_rank"] = rank
    return ordered


def write_summary(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def energy_area_front(
    rows: list[dict[str, str | float | int]],
) -> list[dict[str, str | float | int]]:
    """Return the architecture-level minimum-energy versus area Pareto front."""

    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["tile_area_mm2"]),
            float(row["min_energy_mJ"]),
        ),
    )
    front = []
    best_energy = float("inf")
    for row in ordered:
        energy = float(row["min_energy_mJ"])
        if energy < best_energy:
            front.append(row)
            best_energy = energy
    return front


def edp_area_rows(
    summary: list[dict[str, str | float | int]],
) -> list[dict[str, str | float | int]]:
    rows = []
    for row in summary:
        rows.append(
            {
                **row,
                "EDP_mJ_ms": float(row["min_edp_mJ_ms"]),
                "EDAP_mJ_ms_mm2": float(row["min_edp_mJ_ms"])
                * float(row["tile_area_mm2"]),
            }
        )
    return rows


def edp_area_front(
    rows: list[dict[str, str | float | int]],
) -> list[dict[str, str | float | int]]:
    ordered = sorted(
        rows,
        key=lambda row: (float(row["tile_area_mm2"]), float(row["EDP_mJ_ms"])),
    )
    front = []
    best_edp = float("inf")
    for row in ordered:
        candidate_edp = float(row["EDP_mJ_ms"])
        if candidate_edp < best_edp:
            front.append(row)
            best_edp = candidate_edp
    return front


def draw_edp_area_and_edap_ranking(
    result_dir: Path,
    prefix: str,
    summary: list[dict[str, str | float | int]],
    samples: list[dict[str, str]],
) -> None:
    rows = edp_area_rows(summary)
    front = edp_area_front(rows)
    front_designs = {str(row["design"]) for row in front}
    class_styles = {
        "homogeneous": ("o", "#2e86ab"),
        "heterogeneous": ("s", "#c75b39"),
    }
    subtitle = coverage_text(samples)

    figure, axis = plt.subplots(figsize=(9.2, 6.5))
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
    # Label every architecture on the EDP-area front; dominated points remain
    # intentionally unlabeled so the frontier is readable in a paper figure.
    label_offsets = {
        "3xm32x8+3xm128x8": (8, 12),
        "4xm128x8": (8, -20),
        "2xm128x8+1xm256x8": (8, 16),
        "1xm512x8": (-62, 18),
        "1xm32x8+1xm512x8": (8, 28),
        "2xm32x8+1xm512x8": (-104, 22),
        "1xm64x8+1xm512x8": (12, -12),
    }
    fallback_offsets = ((8, 10), (-72, 10), (8, -20), (-72, -20))
    for index, row in enumerate(front):
        design = str(row["design"])
        offset = label_offsets.get(design, fallback_offsets[index % len(fallback_offsets)])
        axis.annotate(
            design_label(design),
            (float(row["tile_area_mm2"]), float(row["EDP_mJ_ms"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5,
            color="#202020",
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.55},
            zorder=5,
        )
    axis.set_title(f"{WORKLOAD_LABEL} Tile EDP versus Area\n{subtitle}")
    axis.set_xlabel(r"Tile area (mm$^2$)")
    axis.set_ylabel(r"Energy-delay product (mJ$\cdot$ms)")
    axis.grid(alpha=0.24)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color="#2e86ab", label="Homogeneous"),
            Line2D([], [], marker="s", linestyle="", color="#c75b39", label="Heterogeneous"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor="#202020", label="EDP-Area Pareto"),
        ],
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(result_dir / f"{prefix}_edp_area_pareto.png", dpi=220)
    figure.savefig(result_dir / f"{prefix}_edp_area_pareto.pdf")
    plt.close(figure)

    ordered = sorted(
        rows,
        key=lambda row: (float(row["EDAP_mJ_ms_mm2"]), str(row["design"])),
    )[:10]
    figure, axis = plt.subplots(figsize=(11.5, 6.5))
    axis.barh(
        range(len(ordered)),
        [float(row["EDAP_mJ_ms_mm2"]) for row in ordered],
        color=[
            "#2e86ab" if row["tile_class"] == "homogeneous" else "#c75b39"
            for row in ordered
        ],
        alpha=0.88,
        height=0.72,
    )
    axis.set_yticks(
        range(len(ordered)),
        [design_label(str(row["design"])) for row in ordered],
    )
    axis.invert_yaxis()
    axis.set_xlabel(r"Energy $\times$ Delay $\times$ Area (mJ$\cdot$ms$\cdot$mm$^2$)")
    axis.set_ylabel("Tile architecture (best at top)")
    axis.set_title(f"{WORKLOAD_LABEL} Tile Energy-Delay-Area Top 10\n{subtitle}")
    axis.grid(axis="x", alpha=0.24)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Line2D([], [], marker="s", linestyle="", color="#2e86ab", label="Homogeneous"),
            Line2D([], [], marker="s", linestyle="", color="#c75b39", label="Heterogeneous"),
        ],
        frameon=False,
        loc="upper right",
    )
    figure.tight_layout()
    figure.savefig(result_dir / f"{prefix}_edap_ranking.png", dpi=220)
    figure.savefig(result_dir / f"{prefix}_edap_ranking.pdf")
    plt.close(figure)

    write_summary(result_dir / f"{prefix}_edp_area_pareto.csv", front)
    write_summary(result_dir / f"{prefix}_edap_ranking.csv", ordered)


def annotate(
    axis,
    row: dict[str, str],
    label: str,
    offset: tuple[int, int],
    *,
    energy_key: str = "fabric_energy_mJ",
    latency_key: str = "latency_ms",
) -> None:
    axis.annotate(
        label,
        (value(row, latency_key), value(row, energy_key)),
        xytext=offset,
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7},
        zorder=8,
    )


def draw_energy_latency(
    result_dir: Path,
    prefix: str,
    rows: list[dict[str, str]],
    front: list[dict[str, str]],
    samples: list[dict[str, str]],
) -> None:
    target = best(
        (row for row in rows if row["design"] == TARGET_DESIGN),
        edp,
    )
    low_energy = best(rows, lambda row: value(row, "fabric_energy_mJ"))
    low_latency = best(rows, lambda row: value(row, "latency_ms"))
    low_edp = best(rows, edp)

    figure, axis = plt.subplots(figsize=(11.5, 7.2))
    axis.scatter(
        [value(row, "latency_ms") for row in rows],
        [value(row, "fabric_energy_mJ") for row in rows],
        s=7,
        color="#b9bec4",
        alpha=0.18,
        linewidths=0,
        rasterized=True,
        label=f"all feasible mappings ({len(rows):,})",
        zorder=1,
    )
    pareto = axis.scatter(
        [value(row, "latency_ms") for row in front],
        [value(row, "fabric_energy_mJ") for row in front],
        c=[value(row, "tile_area_mm2") for row in front],
        cmap="viridis",
        s=17,
        alpha=0.76,
        linewidths=0,
        label=f"energy-latency Pareto ({len(front):,})",
        zorder=3,
    )
    colorbar = figure.colorbar(pareto, ax=axis, pad=0.015)
    colorbar.set_label("Tile area (mm$^2$)")

    extrema = (
        (low_energy, "minimum energy", "s", "#238636", (-100, 16)),
        (low_latency, "minimum latency", "^", "#0969da", (10, -26)),
        (low_edp, "minimum EDP", "D", "#d97706", (10, 13)),
    )
    for row, label, marker, color, offset in extrema:
        axis.scatter(
            value(row, "latency_ms"),
            value(row, "fabric_energy_mJ"),
            marker=marker,
            s=86,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=6,
        )
        annotate(axis, row, f"{label}\n{design_label(row['design'])}", offset)

    axis.scatter(
        value(target, "latency_ms"),
        value(target, "fabric_energy_mJ"),
        marker="X",
        s=150,
        color=ACCENT,
        edgecolor="white",
        linewidth=1.0,
        zorder=7,
        label="64 + 512 minimum-EDP mapping",
    )
    annotate(axis, target, "64 + 512\n512 projection / 64 attention", (-16, 35))
    figure.suptitle(
        f"{WORKLOAD_LABEL} Tile DSE: Energy-Latency Search Result\n"
        + coverage_text(samples),
        fontsize=14,
    )
    axis.set_xlabel("Modeled Tile latency (ms)")
    axis.set_ylabel("Tile fabric energy (mJ)")
    axis.grid(alpha=0.22)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right")
    figure.text(
        0.5,
        0.012,
        "Points are discrete mappings. The 64 + 512 point is on the three-objective front, not the two-objective front.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.92))
    figure.savefig(result_dir / f"{prefix}_energy_latency_pareto.png", dpi=220)
    figure.savefig(result_dir / f"{prefix}_energy_latency_pareto.pdf")
    plt.close(figure)


def draw_energy_latency_area(
    result_dir: Path,
    prefix: str,
    eta_front: list[dict[str, str]],
    samples: list[dict[str, str]],
) -> None:
    target = best(
        (row for row in eta_front if row["design"] == TARGET_DESIGN),
        edp,
    )
    latency = [value(row, "latency_ms") for row in eta_front]
    energy = [value(row, "fabric_energy_mJ") for row in eta_front]
    area = [value(row, "tile_area_mm2") for row in eta_front]

    figure = plt.figure(figsize=(14, 6.8))
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_2d = figure.add_subplot(1, 2, 2)
    axis_3d.scatter(
        latency,
        energy,
        area,
        c=area,
        cmap="viridis",
        s=9,
        alpha=0.38,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    axis_3d.scatter(
        value(target, "latency_ms"),
        value(target, "fabric_energy_mJ"),
        value(target, "tile_area_mm2"),
        marker="X",
        s=100,
        color=ACCENT,
        edgecolor="white",
        linewidth=0.8,
        depthshade=False,
    )
    axis_3d.set_xlabel("Latency (ms)")
    axis_3d.set_ylabel("Energy (mJ)")
    axis_3d.set_zlabel("Tile area (mm$^2$)")
    axis_3d.set_title("Three-objective Pareto front")
    axis_3d.view_init(elev=23, azim=-57)

    projection = axis_2d.scatter(
        latency,
        energy,
        c=area,
        cmap="viridis",
        s=10,
        alpha=0.48,
        linewidths=0,
        rasterized=True,
    )
    axis_2d.scatter(
        value(target, "latency_ms"),
        value(target, "fabric_energy_mJ"),
        marker="X",
        s=145,
        color=ACCENT,
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
        label="64 + 512 minimum-EDP mapping",
    )
    annotate(axis_2d, target, "64 + 512", (12, -22))
    axis_2d.set_xlabel("Modeled Tile latency (ms)")
    axis_2d.set_ylabel("Tile fabric energy (mJ)")
    axis_2d.set_title("Energy-latency projection, colored by area")
    axis_2d.grid(alpha=0.22)
    axis_2d.set_axisbelow(True)
    axis_2d.legend(frameon=False)
    colorbar = figure.colorbar(projection, ax=axis_2d, pad=0.018)
    colorbar.set_label("Tile area (mm$^2$)")
    figure.suptitle(
        f"{WORKLOAD_LABEL} Tile DSE: Energy-Latency-Area Pareto\n"
        f"{coverage_text(samples)}; {len(eta_front):,} Pareto mappings",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(result_dir / f"{prefix}_energy_latency_area_pareto.png", dpi=220)
    figure.savefig(result_dir / f"{prefix}_energy_latency_area_pareto.pdf")
    plt.close(figure)


def draw_energy_area(
    result_dir: Path,
    prefix: str,
    summary: list[dict[str, str | float | int]],
    front: list[dict[str, str | float | int]],
    samples: list[dict[str, str]],
) -> None:
    target = next(row for row in summary if row["design"] == TARGET_DESIGN)
    figure, axis = plt.subplots(figsize=(11.5, 7.2))
    all_points = axis.scatter(
        [float(row["tile_area_mm2"]) for row in summary],
        [float(row["min_energy_mJ"]) for row in summary],
        c=[float(row["min_energy_latency_ms"]) for row in summary],
        cmap="viridis_r",
        s=48,
        alpha=0.34,
        linewidths=0,
        label=f"completed Tile architectures ({len(summary)})",
        zorder=1,
    )
    axis.scatter(
        [float(row["tile_area_mm2"]) for row in front],
        [float(row["min_energy_mJ"]) for row in front],
        c=[float(row["min_energy_latency_ms"]) for row in front],
        cmap="viridis_r",
        norm=all_points.norm,
        s=78,
        alpha=0.92,
        edgecolor="#202020",
        linewidth=0.7,
        label=f"energy-area Pareto ({len(front)})",
        zorder=3,
    )
    axis.scatter(
        float(target["tile_area_mm2"]),
        float(target["min_energy_mJ"]),
        marker="X",
        s=170,
        color=ACCENT,
        edgecolor="white",
        linewidth=1.0,
        label="64 + 512",
        zorder=6,
    )

    anchors = {
        "1xm512x8": (-70, 26),
        "1xm32x8+1xm512x8": (-54, 49),
        TARGET_DESIGN: (10, 30),
        "1xm128x8+1xm512x8": (10, 24),
    }
    for design, offset in anchors.items():
        candidates = [row for row in front if row["design"] == design]
        if not candidates:
            continue
        row = candidates[0]
        axis.annotate(
            design_label(design),
            (float(row["tile_area_mm2"]), float(row["min_energy_mJ"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7},
            zorder=7,
        )

    colorbar = figure.colorbar(all_points, ax=axis, pad=0.015)
    colorbar.set_label("Latency of minimum-energy mapping (ms)")
    axis.set_title(
        f"{WORKLOAD_LABEL} Tile DSE: Energy-Area Pareto\n" + coverage_text(samples),
        fontsize=14,
    )
    axis.set_xlabel("Tile area (mm$^2$)")
    axis.set_ylabel("Minimum Tile fabric energy (mJ)")
    axis.grid(alpha=0.22)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right")
    figure.text(
        0.5,
        0.012,
        "Each marker represents one completed Tile architecture at its minimum-energy mapping; color preserves latency context.",
        ha="center",
        fontsize=8.5,
        color="#444444",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(result_dir / f"{prefix}_energy_area_pareto.png", dpi=220)
    figure.savefig(result_dir / f"{prefix}_energy_area_pareto.pdf")
    plt.close(figure)


def draw_architecture_summary(
    result_dir: Path,
    prefix: str,
    summary: list[dict[str, str | float | int]],
    samples: list[dict[str, str]],
) -> None:
    target = next(row for row in summary if row["design"] == TARGET_DESIGN)
    min_energy = min(summary, key=lambda row: float(row["min_energy_mJ"]))
    min_latency = min(summary, key=lambda row: float(row["min_latency_ms"]))
    min_edp = min(summary, key=lambda row: float(row["min_edp_mJ_ms"]))
    key_designs = {
        str(target["design"]): (24, 29),
        str(min_energy["design"]): (-24, 44),
        str(min_latency["design"]): (10, -25),
        str(min_edp["design"]): (10, 17),
    }

    figure, (axis, ranking) = plt.subplots(1, 2, figsize=(15, 7.4), gridspec_kw={"width_ratios": [1.25, 1]})
    areas = [float(row["tile_area_mm2"]) for row in summary]
    min_area, max_area = min(areas), max(areas)
    span = max(max_area - min_area, 1e-12)
    for row in summary:
        marker_size = 45 + 180 * (float(row["tile_area_mm2"]) - min_area) / span
        axis.scatter(
            float(row["min_edp_latency_ms"]),
            float(row["min_edp_energy_mJ"]),
            s=marker_size,
            color=CLASS_COLORS[str(row["tile_class"])],
            alpha=0.72,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    axis.scatter(
        float(target["min_edp_latency_ms"]),
        float(target["min_edp_energy_mJ"]),
        marker="X",
        s=190,
        color=ACCENT,
        edgecolor="white",
        linewidth=1.0,
        zorder=6,
    )
    for design, offset in key_designs.items():
        row = next(item for item in summary if item["design"] == design)
        axis.annotate(
            design_label(design),
            (float(row["min_edp_latency_ms"]), float(row["min_edp_energy_mJ"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.7},
        )
    axis.set_title("One minimum-EDP mapping per completed Tile")
    axis.set_xlabel("Latency (ms)")
    axis.set_ylabel("Energy (mJ)")
    axis.grid(alpha=0.22)
    axis.set_axisbelow(True)
    axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS["homogeneous"], label="homogeneous"),
            Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS["heterogeneous"], label="heterogeneous"),
            Line2D([], [], marker="X", linestyle="", color=ACCENT, markersize=9, label="64 + 512"),
        ],
        title="Marker area scales with Tile area",
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )

    shown = list(summary[:15])
    if target not in shown:
        shown.append(target)
    shown = sorted(shown, key=lambda row: float(row["min_edp_mJ_ms"]), reverse=True)
    labels = [f"#{int(row['min_edp_rank'])}  {design_label(str(row['design']))}" for row in shown]
    colors = [ACCENT if row["design"] == TARGET_DESIGN else CLASS_COLORS[str(row["tile_class"])] for row in shown]
    bars = ranking.barh(labels, [float(row["min_edp_mJ_ms"]) for row in shown], color=colors, alpha=0.85)
    for bar, row in zip(bars, shown):
        ranking.text(
            bar.get_width() + max(float(item["min_edp_mJ_ms"]) for item in shown) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{float(row['min_edp_mJ_ms']):.1f}",
            va="center",
            fontsize=7.5,
        )
    ranking_title = "Top 15 architectures by lowest EDP"
    if int(target["min_edp_rank"]) > 15:
        ranking_title += " + 64 + 512"
    ranking.set_title(ranking_title)
    ranking.set_xlabel("Energy-delay product (mJ ms)")
    ranking.grid(axis="x", alpha=0.22)
    ranking.set_axisbelow(True)
    ranking.margins(x=0.14)
    figure.suptitle(
        f"{WORKLOAD_LABEL} Tile Architecture Comparison\n" + coverage_text(samples),
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(result_dir / f"{prefix}_architecture_summary.png", dpi=220)
    figure.savefig(result_dir / f"{prefix}_architecture_summary.pdf")
    plt.close(figure)


def main() -> None:
    global WORKLOAD_LABEL
    args = parse_args()
    WORKLOAD_LABEL = args.workload_label
    result_dir = args.result_dir.resolve()
    aggregate = read_csv(result_dir / f"{args.prefix}_aggregate.csv")
    el_path = result_dir / f"{args.prefix}_aggregate_pareto.csv"
    eta_path = result_dir / f"{args.prefix}_aggregate_pareto_eta.csv"
    el_front = read_csv(el_path) if el_path.is_file() else pareto_rows(
        aggregate, ("fabric_energy_mJ", "latency_ms")
    )
    eta_front = read_csv(eta_path) if eta_path.is_file() else pareto_rows(
        aggregate, ("fabric_energy_mJ", "latency_ms", "tile_area_mm2")
    )
    if not el_path.is_file():
        write_csv(el_path, el_front)
    if not eta_path.is_file():
        write_csv(eta_path, eta_front)
    samples = read_csv(result_dir / f"{args.prefix}_bayesian_samples.csv")
    if TARGET_DESIGN not in {row["design"] for row in aggregate}:
        raise ValueError(f"target design is absent from completed results: {TARGET_DESIGN}")

    summary = architecture_summary(aggregate, el_front, eta_front)
    ea_front = energy_area_front(summary)
    ea_designs = {row["design"] for row in ea_front}
    for row in summary:
        row["on_energy_area_pareto"] = int(row["design"] in ea_designs)
    summary_path = result_dir / f"{args.prefix}_architecture_summary.csv"
    ea_path = result_dir / f"{args.prefix}_energy_area_pareto.csv"
    write_summary(summary_path, summary)
    write_summary(ea_path, ea_front)
    draw_energy_latency(result_dir, args.prefix, aggregate, el_front, samples)
    draw_energy_latency_area(result_dir, args.prefix, eta_front, samples)
    draw_energy_area(result_dir, args.prefix, summary, ea_front, samples)
    draw_architecture_summary(result_dir, args.prefix, summary, samples)
    draw_edp_area_and_edap_ranking(result_dir, args.prefix, summary, samples)

    target = next(row for row in summary if row["design"] == TARGET_DESIGN)
    print(f"Completed architectures: {len(summary)}")
    print(f"64 + 512 minimum-EDP rank: {target['min_edp_rank']}/{len(summary)}")
    print(f"64 + 512 on E-L Pareto: {bool(target['on_energy_latency_pareto'])}")
    print(f"64 + 512 on E-A Pareto: {bool(target['on_energy_area_pareto'])}")
    print(f"64 + 512 on E-L-A Pareto: {bool(target['on_energy_latency_area_pareto'])}")
    print(summary_path)
    print(ea_path)
    for suffix in (
        "energy_latency_pareto.png",
        "energy_latency_pareto.pdf",
        "energy_area_pareto.png",
        "energy_area_pareto.pdf",
        "energy_latency_area_pareto.png",
        "energy_latency_area_pareto.pdf",
        "architecture_summary.png",
        "architecture_summary.pdf",
        "edp_area_pareto.png",
        "edp_area_pareto.pdf",
        "edap_ranking.png",
        "edap_ranking.pdf",
    ):
        print(result_dir / f"{args.prefix}_{suffix}")


if __name__ == "__main__":
    main()

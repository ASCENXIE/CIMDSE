#!/usr/bin/env python3
"""Generate Innovation 1 figures and summary tables.

The script intentionally has one source of truth for architecture metrics:
each design is represented by its lowest-EDP mapping.  The plotting functions
only format that prepared data; they do not recompute DSE results.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/accelforge_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[3]
FIGURE_DIR = ROOT / "paper" / "figures" / "innovation1"
TABLE_DIR = ROOT / "paper" / "tables"

TARGET = "1xm64x8+1xm512x8"
COMPARISONS = ("2xm256x8", "1xm512x8")
HOMOGENEOUS = "#0072B2"
HETEROGENEOUS = "#D55E00"
SUBPLOT_LEGEND_FONT = FontProperties(
    family="Liberation Sans",
    size=9.5,
)

WORKLOAD_FILES = {
    "ResNet-18": ROOT / "experiments/results/tile_DSE/resnet18_v6/tile_window_dse_aggregate.csv",
    "ViT-B/16": ROOT / "experiments/results/tile_DSE/vision_transformer_l16_v2/vit_l16_pipeline_tile_dse_aggregate.csv",
}


def read_architectures(path: Path) -> list[dict[str, Any]]:
    """Collapse mapping rows into one lowest-EDP row per architecture."""
    with path.open(newline="") as file:
        mapping_rows = list(csv.DictReader(file))

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in mapping_rows:
        grouped.setdefault(row["design"], []).append(row)

    architectures = []
    for design, candidates in grouped.items():
        best = min(candidates, key=lambda row: float(row["fabric_energy_mJ"]) * float(row["latency_ms"]))
        energy = float(best["fabric_energy_mJ"])
        latency = float(best["latency_ms"])
        architectures.append(
            {
                "design": design,
                "tile_class": best["tile_class"],
                "area_mm2": float(best["tile_area_mm2"]),
                "energy_mJ": energy,
                "latency_ms": latency,
                "edp_mJ_ms": energy * latency,
            }
        )
    return sorted(architectures, key=lambda row: str(row["design"]))


def edap(row: dict[str, Any]) -> float:
    """Return EDAP for a prepared architecture row."""
    return float(row["edp_mJ_ms"]) * float(row["area_mm2"])


def design_label(design: str) -> str:
    """Convert ``2xm256x8`` into a compact mathtext label ``2x256``."""
    labels = []
    for item in design.split("+"):
        count, shape = item.split("xm")
        rows = shape.removesuffix("x8")
        labels.append(rows if count == "1" else f"{count}$\\times${rows}")
    return "+".join(labels)


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the strict EDP-area Pareto front."""
    ordered = sorted(rows, key=lambda row: (float(row["area_mm2"]), float(row["edp_mJ_ms"])))
    front = []
    best_edp = float("inf")
    for row in ordered:
        if float(row["edp_mJ_ms"]) < best_edp:
            front.append(row)
            best_edp = float(row["edp_mJ_ms"])
    return front


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a non-empty list of dictionaries, creating its parent directory."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_core_tables(workloads: dict[str, list[dict[str, Any]]]) -> None:
    """Export Pareto rows and the three-design cross-workload comparison."""
    pareto_rows = []
    for workload, rows in workloads.items():
        pareto_rows.extend({"workload": workload, "pareto": 1, **row} for row in pareto_front(rows))
    write_csv(TABLE_DIR / "innovation1_core_result_pareto.csv", pareto_rows)

    common = set.intersection(*(set(row["design"] for row in rows) for rows in workloads.values()))
    common_best_edap = {
        workload: min(edap(row) for row in rows if row["design"] in common)
        for workload, rows in workloads.items()
    }
    records = []
    for design in (TARGET, *COMPARISONS):
        regrets = []
        per_workload = []
        for workload, rows in workloads.items():
            row = next(row for row in rows if row["design"] == design)
            best_homogeneous = min(
                (item for item in rows if item["tile_class"] == "homogeneous"),
                key=lambda item: float(item["edp_mJ_ms"]),
            )
            row_edap = edap(row)
            homogeneous_edp = float(best_homogeneous["edp_mJ_ms"])
            homogeneous_edap = edap(best_homogeneous)
            regret = row_edap / common_best_edap[workload] - 1
            regrets.append(regret)
            per_workload.append(
                {
                    "design": design,
                    "workload": workload,
                    "area_mm2": row["area_mm2"],
                    "energy_mJ": row["energy_mJ"],
                    "latency_ms": row["latency_ms"],
                    "edp_mJ_ms": row["edp_mJ_ms"],
                    "edap_mJ_ms_mm2": row_edap,
                    "best_homogeneous_design": best_homogeneous["design"],
                    "edp_reduction_vs_best_homogeneous_pct": 100 * (1 - float(row["edp_mJ_ms"]) / homogeneous_edp),
                    "edap_reduction_vs_best_homogeneous_pct": 100 * (1 - row_edap / homogeneous_edap),
                    "edap_regret_on_common_panel_pct": 100 * regret,
                }
            )
        for record in per_workload:
            record["worst_edap_regret_across_workloads_pct"] = 100 * max(regrets)
            record["common_design_count"] = len(common)
            records.append(record)
    write_csv(TABLE_DIR / "innovation1_core_result_selection.csv", records)


def style_axes(axis: Any) -> None:
    """Apply the shared small-figure grid style."""
    axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.32)
    axis.set_axisbelow(True)


def place_caption(axis: Any, text: str, offset: float = -0.30) -> None:
    """Place every subfigure caption at the same distance below its axes."""
    axis.text(
        0.5,
        offset,
        text,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def add_subplot_legend(axis: Any) -> None:
    """Add the class legend inside the upper-right corner of one subplot."""
    axis.legend(
        handles=_legend_handles(),
        loc="upper right",
        prop=SUBPLOT_LEGEND_FONT,
        frameon=False,
        markerscale=0.78,
        handlelength=1.0,
        handletextpad=0.2,
        borderaxespad=0.15,
        labelspacing=0.12,
    )


def scatter_architectures(axis: Any, rows: list[dict[str, Any]]) -> None:
    """Draw all designs, emphasizing the EDP-area Pareto front."""
    front_designs = {row["design"] for row in pareto_front(rows)}
    for tile_class, marker, color in (
        ("homogeneous", "o", HOMOGENEOUS),
        ("heterogeneous", "s", HETEROGENEOUS),
    ):
        background = [row for row in rows if row["tile_class"] == tile_class and row["design"] not in front_designs]
        foreground = [row for row in rows if row["tile_class"] == tile_class and row["design"] in front_designs]
        for points, size, alpha, edge, width in (
            (background, 12, 0.27, "none", 0),
            (foreground, 23, 0.95, "#202020", 0.65),
        ):
            axis.scatter(
                [row["area_mm2"] for row in points],
                [row["edp_mJ_ms"] for row in points],
                marker=marker,
                s=size,
                color=color,
                alpha=alpha,
                edgecolors=edge,
                linewidths=width,
                zorder=3 if points is foreground else 1,
            )


def draw_core_pareto(workloads: dict[str, list[dict[str, Any]]]) -> None:
    """Write the compact two-panel Pareto-only figure."""
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    for column, (workload, rows) in enumerate(workloads.items()):
        axis = axes[column]
        scatter_architectures(axis, rows)
        target = next(row for row in rows if row["design"] == TARGET)
        axis.scatter(target["area_mm2"], target["edp_mJ_ms"], marker="s", s=52, facecolor="none", edgecolor="#202020", linewidth=1.5, zorder=5)
        axis.annotate("Selected 64+512", (target["area_mm2"], target["edp_mJ_ms"]), xytext=(8, 7), textcoords="offset points", fontsize=6.5, fontweight="bold", arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.4})
        axis.set_title(f"({chr(97 + column)}) {workload}", fontweight="bold", pad=3)
        axis.set_xlabel(r"Tile area (mm$^2$)")
        axis.set_ylabel("Minimum EDP" if column == 0 else "")
        if column == 1:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(3, 3), useMathText=True)
        style_axes(axis)
        axis.margins(x=0.09, y=0.20)
    figure.legend(handles=_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.4)
    save_figure(figure, FIGURE_DIR / "tile_dse_core_result")


def ranking_rows(rows: list[dict[str, Any]], workload: str) -> tuple[list[dict[str, Any]], float]:
    """Select Top 5 plus the workload-specific comparison baselines."""
    ordered = sorted(rows, key=lambda row: (edap(row), row["design"]))
    ranks = {row["design"]: rank for rank, row in enumerate(ordered, 1)}
    required = (TARGET, "1xm32x8+1xm512x8") if workload == "ResNet-18" else (TARGET, *COMPARISONS)
    shown = ordered[:5]
    for design in required:
        if design not in {row["design"] for row in shown}:
            shown.append(next(row for row in ordered if row["design"] == design))
    shown.sort(key=lambda row: ranks[row["design"]])
    minimum = edap(ordered[0])
    return [{"rank": ranks[row["design"]], **row, "edap_mJ_ms_mm2": edap(row), "normalized_edap": edap(row) / minimum} for row in shown], minimum


def draw_ranking(workloads: dict[str, list[dict[str, Any]]], normalized: bool, output_stem: str) -> None:
    """Write the standalone two-panel absolute or normalized EDAP ranking."""
    figure, axes = plt.subplots(1, 2, figsize=(7.16, 3.55))
    exported = []
    for axis, (workload, rows) in zip(axes, workloads.items()):
        shown, _ = ranking_rows(rows, workload)
        values = [row["normalized_edap"] if normalized else row["edap_mJ_ms_mm2"] for row in shown]
        bars = axis.barh(range(len(shown)), values, color=[HOMOGENEOUS if row["tile_class"] == "homogeneous" else HETEROGENEOUS for row in shown], alpha=0.88, height=0.68, edgecolor=["#202020" if row["design"] == TARGET else "none" for row in shown], linewidth=[1.4 if row["design"] == TARGET else 0 for row in shown])
        axis.set_yticks(range(len(shown)), [f"{row['rank']:>2}.  {design_label(row['design'])}" for row in shown])
        axis.invert_yaxis()
        axis.set_title(workload, fontweight="bold", pad=3)
        axis.set_xlabel("Normalized EDAP (best = 1.0)" if normalized else r"EDAP (mJ$\cdot$ms$\cdot$mm$^2$)")
        limit = max(values[:5]) * (1.55 if normalized and workload == "ResNet-18" else 1.42 if normalized else 1.24)
        axis.set_xlim(0, limit)
        for bar, row, value in zip(bars, shown, values):
            text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            y = bar.get_y() + bar.get_height() / 2
            axis.text(min(value, limit) * 0.98 if value > limit else value, y, f"{text} (off scale) >" if value > limit else f"  {text}", ha="right" if value > limit else "left", va="center", fontsize=5.5, fontweight="bold" if row["design"] == TARGET else "normal")
            exported.append({"workload": workload, "shown_in_figure": 1, **row})
        style_axes(axis)
    figure.legend(handles=_ranking_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.5)
    save_figure(figure, FIGURE_DIR / output_stem)
    write_csv(TABLE_DIR / "innovation1_core_result_edap_ranking.csv", exported)


def draw_combined(workloads: dict[str, list[dict[str, Any]]], normalized: bool, output_stem: str) -> None:
    """Write the 2x2 paper figure: Pareto row above EDAP ranking row."""
    figure, axes = plt.subplots(2, 2, figsize=(7.16, 5.05), gridspec_kw={"height_ratios": [1.0, 1.18]})
    for column, (workload, rows) in enumerate(workloads.items()):
        axis = axes[0, column]
        scatter_architectures(axis, rows)
        target = next(row for row in rows if row["design"] == TARGET)
        axis.scatter(target["area_mm2"], target["edp_mJ_ms"], marker="s", s=42, facecolor="none", edgecolor="#202020", linewidth=1.35, zorder=5)
        axis.annotate("Selected 64+512", (target["area_mm2"], target["edp_mJ_ms"]), xytext=(8, 7), textcoords="offset points", fontsize=8.5, fontweight="bold", arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.5})
        axis.set_xlabel(r"Area (mm$^2$)")
        axis.set_ylabel("Minimum EDP" if column == 0 else "")
        if column == 1:
            axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
            axis.text(0.01, 0.98, r"EDP ($10^3$ mJ$\cdot$ms)", transform=axis.transAxes, ha="left", va="top", fontsize=8.5)
        style_axes(axis)
        axis.margins(x=0.08, y=0.17)
        add_subplot_legend(axis)
        place_caption(axis, f"({'a' if column == 0 else 'b'}) {workload}: EDP-area Pareto", offset=-0.38)

        ranking = axes[1, column]
        shown, _ = ranking_rows(rows, workload)
        values = [row["normalized_edap"] if normalized else row["edap_mJ_ms_mm2"] for row in shown]
        bars = ranking.barh(range(len(shown)), values, color=[HOMOGENEOUS if row["tile_class"] == "homogeneous" else HETEROGENEOUS for row in shown], alpha=0.88, height=0.64, edgecolor=["#202020" if row["design"] == TARGET else "none" for row in shown], linewidth=[1.3 if row["design"] == TARGET else 0 for row in shown])
        ranking.set_yticks(range(len(shown)), [f"{row['rank']}. {design_label(row['design'])}" for row in shown])
        ranking.invert_yaxis()
        limit = max(values[:5]) * (1.55 if normalized and workload == "ResNet-18" else 1.42 if normalized else 1.35 if workload == "ResNet-18" else 1.25)
        ranking.set_xlim(0, limit)
        for bar, row, value in zip(bars, shown, values):
            text = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            y = bar.get_y() + bar.get_height() / 2
            ranking.text(min(value, limit) * 0.98 if value > limit else value, y, f"{text} (off scale) >" if value > limit else f"  {text}", ha="right" if value > limit else "left", va="center", fontsize=8, fontweight="bold" if row["design"] == TARGET else "normal")
        ranking.set_xlabel("Normalized EDAP (best = 1.0)" if normalized else r"EDAP (mJ$\cdot$ms$\cdot$mm$^2$)")
        ranking.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.32)
        ranking.set_axisbelow(True)
        place_caption(
            ranking,
            f"({'c' if column == 0 else 'd'}) {workload}: "
            f"{'Normalized ' if normalized else ''}EDAP ranking",
            offset=-0.30,
        )

    # Keep enough horizontal space for ranking labels while using a compact gap
    # between the Pareto and ranking rows.
    figure.subplots_adjust(left=0.13, right=0.985, top=0.97, bottom=0.145, wspace=0.32, hspace=0.50)
    save_figure(figure, FIGURE_DIR / output_stem)


def _legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="o", markersize=4.5, linestyle="", color=HOMOGENEOUS, label="Homogeneous"),
        Line2D([], [], marker="s", markersize=4.5, linestyle="", color=HETEROGENEOUS, label="Heterogeneous"),
    ]


def _ranking_legend_handles() -> list[Line2D]:
    return [
        Line2D([], [], marker="s", linestyle="", color=HOMOGENEOUS, label="Homogeneous"),
        Line2D([], [], marker="s", linestyle="", color=HETEROGENEOUS, label="Heterogeneous"),
        Line2D([], [], marker="s", linestyle="", markerfacecolor=HETEROGENEOUS, markeredgecolor="#202020", markeredgewidth=1.4, label="Selected 64+512"),
    ]


def save_figure(figure: Any, stem: Path) -> None:
    """Save both publication PDF and 600-dpi PNG beside each other."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Load inputs, export tables, and generate all maintained figure variants."""
    plt.rcParams.update({"font.family": "Liberation Sans", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"], "mathtext.fontset": "dejavusans", "font.size": 9.5, "axes.labelsize": 11.5, "axes.titlesize": 10.5, "legend.fontsize": 9.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.2, "axes.linewidth": 0.9})
    workloads = {workload: read_architectures(path) for workload, path in WORKLOAD_FILES.items()}
    write_core_tables(workloads)
    draw_core_pareto(workloads)
    draw_ranking(workloads, normalized=False, output_stem="tile_dse_edap_ranking")
    draw_combined(workloads, normalized=False, output_stem="tile_dse_combined_result")
    draw_combined(workloads, normalized=True, output_stem="tile_dse_combined_result_normalized")


if __name__ == "__main__":
    main()

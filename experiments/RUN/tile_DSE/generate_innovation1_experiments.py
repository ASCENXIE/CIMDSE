#!/usr/bin/env python3
"""Generate the figures and comparison table supporting paper innovation 1."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from math import ceil
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "experiments" / "RUN"
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/accelforge_mplconfig")

import accelforge as af
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from tile_DSE.sram_tile_dse import layer_shapes_from_workload
from tile_DSE.vit.vision_transformer_workload import vit_b16_operator_groups


MACRO_CSV = ROOT / "experiments" / "docs" / "SRAM_DATA" / "cim_macro_efficiency.csv"
DEFAULT_RESNET_DIR = ROOT / "experiments" / "results" / "tile_DSE" / "resnet18_v6"
DEFAULT_VIT_DIR = (
    ROOT / "experiments" / "results" / "tile_DSE" / "vision_transformer_v5"
)
DEFAULT_FIGURE_DIR = ROOT / "paper" / "figures" / "innovation1"
DEFAULT_TABLE_DIR = ROOT / "paper" / "tables"
CNN_PREFIX = "tile_window_dse"
VIT_PREFIX = "vit_b16_pipeline_tile_dse"
TARGET_DESIGN = "1xm64x8+1xm512x8"
MACRO_ROWS = (32, 64, 128, 256, 512)
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#6B7280"
TARGET_COLOR = "#B42318"


@dataclass(frozen=True)
class OperatorShape:
    workload: str
    operator: str
    reduction_length: int
    output_channels: int
    output_positions: int
    repeat: int

    @property
    def macs(self) -> int:
        return (
            self.reduction_length
            * self.output_channels
            * self.output_positions
            * self.repeat
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resnet-result-dir", type=Path, default=DEFAULT_RESNET_DIR)
    parser.add_argument("--vit-result-dir", type=Path, default=DEFAULT_VIT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument(
        "--require-dse-results",
        action="store_true",
        help="Fail instead of skipping the Pareto figure and comparison table.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def macro_characterization_rows() -> list[dict[str, object]]:
    rows = [
        row
        for row in read_csv(MACRO_CSV)
        if int(row["PC"]) == 8 and int(row["WB"]) == 8 and int(row["IB"]) == 8
    ]
    if {int(row["AL"]) for row in rows} != set(MACRO_ROWS):
        raise ValueError("The 8-column Macro table does not contain all DSE row sizes.")
    return [
        {
            "macro": f"{int(row['AL'])}x8",
            "rows": int(row["AL"]),
            "columns": int(row["PC"]),
            "TOPS_per_W": float(row["energy_efficiency"]),
            "TOPS_per_mm2": float(row["area_efficiency"]),
        }
        for row in sorted(rows, key=lambda row: int(row["AL"]))
    ]


def draw_macro_characterization(
    rows: list[dict[str, object]], figure_dir: Path
) -> None:
    x = np.arange(len(rows))
    labels = [str(row["macro"]) for row in rows]
    energy_efficiency = [float(row["TOPS_per_W"]) for row in rows]
    area_efficiency = [float(row["TOPS_per_mm2"]) for row in rows]

    figure, energy_axis = plt.subplots(figsize=(8.2, 4.8))
    area_axis = energy_axis.twinx()
    energy_axis.plot(
        x,
        energy_efficiency,
        color=BLUE,
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="Energy efficiency",
    )
    area_axis.plot(
        x,
        area_efficiency,
        color=ORANGE,
        marker="s",
        linewidth=2.2,
        markersize=6.5,
        label="Area efficiency",
    )
    for index, value in enumerate(energy_efficiency):
        energy_axis.annotate(
            f"{value:.2f}",
            (index, value),
            xytext=(0, 12 if index == 3 else 8),
            textcoords="offset points",
            ha="center",
            color=BLUE,
            fontsize=8,
        )
    for index, value in enumerate(area_efficiency):
        area_axis.annotate(
            f"{value:.2f}",
            (index, value),
            xytext=(0, -22 if index == 3 else -14),
            textcoords="offset points",
            ha="center",
            color=ORANGE,
            fontsize=8,
        )

    energy_axis.set_xticks(x, labels)
    energy_axis.set_xlabel("Macro dimensions (rows x columns)")
    energy_axis.set_ylabel("Energy efficiency (TOPS/W)", color=BLUE)
    area_axis.set_ylabel("Area efficiency (TOPS/mm$^2$)", color=ORANGE)
    energy_axis.tick_params(axis="y", colors=BLUE)
    area_axis.tick_params(axis="y", colors=ORANGE)
    energy_axis.grid(axis="y", alpha=0.22)
    energy_axis.set_axisbelow(True)
    energy_axis.set_title("Compiled Digital SRAM-CIM Macro Characterization")
    handles = [energy_axis.lines[0], area_axis.lines[0]]
    energy_axis.legend(handles, [item.get_label() for item in handles], frameon=False)
    figure.tight_layout()
    save_figure(figure, figure_dir / "macro_efficiency_vs_size")


def workload_shapes() -> list[OperatorShape]:
    workload_path = af.examples.workloads.compute_in_memory.resnet18
    workload = af.Workload.from_yaml(
        workload_path,
        top_key="workload",
        jinja_parse_data={"BATCH_SIZE": 1},
    )
    resnet = [
        OperatorShape(
            "ResNet-18",
            layer.name,
            layer.reduction_length,
            layer.output_channels,
            layer.output_positions,
            1,
        )
        for layer in layer_shapes_from_workload(workload)
    ]
    vit = [
        OperatorShape(
            "ViT-B/16",
            group.name,
            group.shape.reduction_length,
            group.shape.output_channels,
            group.shape.output_positions,
            group.repeat,
        )
        for group in vit_b16_operator_groups()
    ]
    return resnet + vit


def workload_shape_rows(shapes: list[OperatorShape]) -> list[dict[str, object]]:
    total_macs = Counter()
    for shape in shapes:
        total_macs[shape.workload] += shape.macs
    return [
        {
            "workload": shape.workload,
            "operator": shape.operator,
            "K": shape.reduction_length,
            "N": shape.output_channels,
            "P": shape.output_positions,
            "repeat": shape.repeat,
            "MACs": shape.macs,
            "MAC_share_pct": 100.0 * shape.macs / total_macs[shape.workload],
        }
        for shape in shapes
    ]


def row_utilization(shapes: list[OperatorShape], macro_rows: int) -> float:
    useful_macs = sum(shape.macs for shape in shapes)
    padded_macs = sum(
        ceil(shape.reduction_length / macro_rows)
        * macro_rows
        * shape.output_channels
        * shape.output_positions
        * shape.repeat
        for shape in shapes
    )
    return useful_macs / padded_macs


def utilization_rows(shapes: list[OperatorShape]) -> list[dict[str, object]]:
    output = []
    for workload in ("ResNet-18", "ViT-B/16"):
        selected = [shape for shape in shapes if shape.workload == workload]
        for rows in MACRO_ROWS:
            output.append(
                {
                    "workload": workload,
                    "macro_rows": rows,
                    "vmm_weighted_row_utilization_pct": 100.0
                    * row_utilization(selected, rows),
                }
            )
    return output


def draw_reduction_and_utilization(
    shapes: list[OperatorShape], utilization: list[dict[str, object]], figure_dir: Path
) -> None:
    k_values = sorted({shape.reduction_length for shape in shapes})
    x = np.arange(len(k_values))
    width = 0.38
    figure, (distribution_axis, utilization_axis) = plt.subplots(
        1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": (1.5, 1.0)}
    )

    for offset, workload, color in (
        (-width / 2, "ResNet-18", BLUE),
        (width / 2, "ViT-B/16", ORANGE),
    ):
        counts = Counter()
        for shape in shapes:
            if shape.workload == workload:
                counts[shape.reduction_length] += shape.repeat
        total = sum(counts.values())
        distribution_axis.bar(
            x + offset,
            [100.0 * counts[value] / total for value in k_values],
            width,
            color=color,
            alpha=0.88,
            label=workload,
        )
    distribution_axis.set_xticks(x, [str(value) for value in k_values], rotation=45)
    distribution_axis.set_xlabel("Reduction dimension K")
    distribution_axis.set_ylabel("Operator-instance share (%)")
    distribution_axis.set_title("Reduction-Dimension Distribution")
    distribution_axis.grid(axis="y", alpha=0.22)
    distribution_axis.legend(frameon=False)

    for workload, color, marker in (
        ("ResNet-18", BLUE, "o"),
        ("ViT-B/16", ORANGE, "s"),
    ):
        selected = [row for row in utilization if row["workload"] == workload]
        utilization_axis.plot(
            [int(row["macro_rows"]) for row in selected],
            [float(row["vmm_weighted_row_utilization_pct"]) for row in selected],
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6.5,
            label=workload,
        )
    utilization_axis.set_xscale("log", base=2)
    utilization_axis.set_xticks(MACRO_ROWS, [str(value) for value in MACRO_ROWS])
    utilization_axis.set_ylim(0, 105)
    utilization_axis.set_xlabel("Macro rows")
    utilization_axis.set_ylabel("VMM-weighted row utilization (%)")
    utilization_axis.set_title("Ideal Row Utilization")
    utilization_axis.grid(alpha=0.22)
    utilization_axis.legend(frameon=False)
    figure.suptitle("Workload Reduction Shapes and Macro Utilization", fontsize=14)
    figure.tight_layout()
    save_figure(figure, figure_dir / "workload_reduction_dimension_and_utilization")


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def min_edp_architectures(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["design"]].append(row)
    output = []
    for design, candidates in groups.items():
        selected = min(
            candidates,
            key=lambda row: value(row, "fabric_energy_mJ") * value(row, "latency_ms"),
        )
        output.append(
            {
                "design": design,
                "tile_class": selected["tile_class"],
                "energy_mJ": value(selected, "fabric_energy_mJ"),
                "latency_ms": value(selected, "latency_ms"),
                "area_mm2": value(selected, "tile_area_mm2"),
                "EDP_mJ_ms": value(selected, "fabric_energy_mJ")
                * value(selected, "latency_ms"),
                "ADP_mm2_ms": value(selected, "tile_area_mm2")
                * value(selected, "latency_ms"),
                "EAP_mJ_mm2": value(selected, "fabric_energy_mJ")
                * value(selected, "tile_area_mm2"),
            }
        )
    return sorted(output, key=lambda row: float(row["EDP_mJ_ms"]))


def pareto_energy_adp(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (float(row["energy_mJ"]), float(row["ADP_mm2_ms"])),
    )
    front = []
    best_adp = float("inf")
    for row in ordered:
        adp = float(row["ADP_mm2_ms"])
        if adp < best_adp:
            front.append(row)
            best_adp = adp
    return front


def pareto_latency_eap(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (float(row["latency_ms"]), float(row["EAP_mJ_mm2"])),
    )
    front = []
    best_eap = float("inf")
    for row in ordered:
        eap = float(row["EAP_mJ_mm2"])
        if eap < best_eap:
            front.append(row)
            best_eap = eap
    return front


def draw_dse_pareto(
    workloads: dict[str, list[dict[str, object]]], figure_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    class_style = {
        "homogeneous": ("o", BLUE),
        "heterogeneous": ("s", ORANGE),
    }
    for axis, (workload, rows) in zip(axes, workloads.items()):
        front = pareto_energy_adp(rows)
        front_designs = {str(row["design"]) for row in front}
        for tile_class, (marker, color) in class_style.items():
            selected = [row for row in rows if row["tile_class"] == tile_class]
            for on_front, size, alpha, edgecolor, linewidth, zorder in (
                (False, 48, 0.52, "none", 0.0, 1),
                (True, 86, 0.95, "#111827", 1.1, 4),
            ):
                subset = [
                    row
                    for row in selected
                    if (str(row["design"]) in front_designs) == on_front
                ]
                if not subset:
                    continue
                axis.scatter(
                    [float(row["energy_mJ"]) for row in subset],
                    [float(row["ADP_mm2_ms"]) for row in subset],
                    color=color,
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    edgecolors=edgecolor,
                    linewidths=linewidth,
                    zorder=zorder,
                )
        target = next((row for row in rows if row["design"] == TARGET_DESIGN), None)
        if target is not None:
            axis.scatter(
                float(target["energy_mJ"]),
                float(target["ADP_mm2_ms"]),
                marker="X",
                s=175,
                color=TARGET_COLOR,
                edgecolor="white",
                linewidth=0.9,
                zorder=6,
            )
            axis.annotate(
                "64 + 512",
                (float(target["energy_mJ"]), float(target["ADP_mm2_ms"])),
                xytext=(8, 12),
                textcoords="offset points",
                fontsize=8.5,
            )
        axis.set_title(
            f"{workload} ({len(rows)} completed Tiles, {len(front_designs)} E-ADP Pareto)"
        )
        axis.set_xlabel("Energy at minimum-EDP mapping (mJ)")
        axis.set_ylabel(r"Area-delay product (mm$^2\cdot$ms)")
        axis.grid(alpha=0.22)
    figure.suptitle(
        "Tile Energy versus Area-Delay Product",
        fontsize=14,
    )
    legend = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, label="Homogeneous"),
        Line2D([], [], marker="s", linestyle="", color=ORANGE, label="Heterogeneous"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor="#111827",
            label="Energy-ADP Pareto",
        ),
        Line2D(
            [],
            [],
            marker="X",
            linestyle="",
            color=TARGET_COLOR,
            label="64 + 512",
        ),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(figure, figure_dir / "resnet18_vitb_tile_energy_adp_pareto")


def draw_latency_eap_pareto(
    workloads: dict[str, list[dict[str, object]]], figure_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    class_style = {
        "homogeneous": ("o", BLUE),
        "heterogeneous": ("s", ORANGE),
    }
    for axis, (workload, rows) in zip(axes, workloads.items()):
        front = pareto_latency_eap(rows)
        front_designs = {str(row["design"]) for row in front}
        for tile_class, (marker, color) in class_style.items():
            selected = [row for row in rows if row["tile_class"] == tile_class]
            for on_front, size, alpha, edgecolor, linewidth, zorder in (
                (False, 48, 0.52, "none", 0.0, 1),
                (True, 86, 0.95, "#111827", 1.1, 4),
            ):
                subset = [
                    row
                    for row in selected
                    if (str(row["design"]) in front_designs) == on_front
                ]
                if not subset:
                    continue
                axis.scatter(
                    [float(row["latency_ms"]) for row in subset],
                    [float(row["EAP_mJ_mm2"]) for row in subset],
                    color=color,
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    edgecolors=edgecolor,
                    linewidths=linewidth,
                    zorder=zorder,
                )
        target = next((row for row in rows if row["design"] == TARGET_DESIGN), None)
        if target is not None:
            axis.scatter(
                float(target["latency_ms"]),
                float(target["EAP_mJ_mm2"]),
                marker="X",
                s=175,
                color=TARGET_COLOR,
                edgecolor="white",
                linewidth=0.9,
                zorder=6,
            )
            axis.annotate(
                "64 + 512",
                (float(target["latency_ms"]), float(target["EAP_mJ_mm2"])),
                xytext=(8, 12),
                textcoords="offset points",
                fontsize=8.5,
            )
        axis.set_title(
            f"{workload} ({len(rows)} completed Tiles, {len(front_designs)} L-EA Pareto)"
        )
        axis.set_xlabel("Latency at minimum-EDP mapping (ms)")
        axis.set_ylabel(r"Energy-area product (mJ$\cdot$mm$^2$)")
        axis.grid(alpha=0.22)
    figure.suptitle("Tile Latency versus Energy-Area Product", fontsize=14)
    legend = [
        Line2D([], [], marker="o", linestyle="", color=BLUE, label="Homogeneous"),
        Line2D([], [], marker="s", linestyle="", color=ORANGE, label="Heterogeneous"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor="#111827",
            label="Latency-EAP Pareto",
        ),
        Line2D(
            [],
            [],
            marker="X",
            linestyle="",
            color=TARGET_COLOR,
            label="64 + 512",
        ),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(figure, figure_dir / "resnet18_vitb_tile_latency_energy_area_pareto")


def comparison_rows(
    workloads: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    output = []
    roles = (
        ("best_homogeneous", None),
        ("best_heterogeneous", None),
        ("proposed", TARGET_DESIGN),
        ("equal_TOPS_homogeneous", "9xm64x8"),
        ("large_Macro_baseline", "1xm512x8"),
    )
    for workload, rows in workloads.items():
        energy_adp_front = {
            str(row["design"])
            for row in pareto_energy_adp(rows)
        }
        best_by_class = {
            tile_class: min(
                (row for row in rows if row["tile_class"] == tile_class),
                key=lambda row: float(row["EDP_mJ_ms"]),
            )
            for tile_class in ("homogeneous", "heterogeneous")
        }
        for role, fixed_design in roles:
            if role == "best_homogeneous":
                selected = best_by_class["homogeneous"]
            elif role == "best_heterogeneous":
                selected = best_by_class["heterogeneous"]
            else:
                selected = next(
                    (row for row in rows if row["design"] == fixed_design), None
                )
                if selected is None:
                    continue
            output.append(
                {
                    "workload": workload,
                    "comparison_role": role,
                    "design": selected["design"],
                    "tile_class": selected["tile_class"],
                    "energy_mJ": float(selected["energy_mJ"]),
                    "latency_ms": float(selected["latency_ms"]),
                    "area_mm2": float(selected["area_mm2"]),
                    "EDP_mJ_ms": float(selected["EDP_mJ_ms"]),
                    "ADP_mm2_ms": float(selected["ADP_mm2_ms"]),
                    "on_energy_adp_pareto": int(
                        str(selected["design"]) in energy_adp_front
                    ),
                }
            )
    return output


def write_comparison_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    labels = {
        "best_homogeneous": "Best-EDP homogeneous",
        "best_heterogeneous": "Best-EDP heterogeneous",
        "proposed": "Proposed 64+512",
        "equal_TOPS_homogeneous": "Equal-TOPS 9x64",
        "large_Macro_baseline": "Single 512",
    }
    lines = [
        "# Homogeneous and Heterogeneous Tile Comparison",
        "",
        "Each row uses one internally consistent minimum-EDP mapping. The Pareto flag",
        "is computed in the energy versus area-delay-product plane.",
        "",
        "| Workload | Role | Design | Class | Energy (mJ) | Latency (ms) | Area (mm2) | EDP (mJ ms) | ADP (mm2 ms) | E-ADP Pareto |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["workload"]),
                    labels[str(row["comparison_role"])],
                    f"`{row['design']}`",
                    str(row["tile_class"]),
                    f"{float(row['energy_mJ']):.4f}",
                    f"{float(row['latency_ms']):.4f}",
                    f"{float(row['area_mm2']):.4f}",
                    f"{float(row['EDP_mJ_ms']):.4f}",
                    f"{float(row['ADP_mm2_ms']):.4f}",
                    str(row["on_energy_adp_pareto"]),
                )
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def validate_run_metadata(args: argparse.Namespace) -> None:
    sample_paths = {
        "ResNet-18": args.resnet_result_dir / f"{CNN_PREFIX}_bayesian_samples.csv",
        "ViT-B/16": args.vit_result_dir / f"{VIT_PREFIX}_bayesian_samples.csv",
    }
    metadata = {workload: read_csv(path) for workload, path in sample_paths.items()}
    if any(not rows for rows in metadata.values()):
        missing = [workload for workload, rows in metadata.items() if not rows]
        raise FileNotFoundError(f"Missing DSE run metadata for: {missing}")
    versions = {
        workload: {row.get("energy_model_version", "") for row in rows}
        for workload, rows in metadata.items()
    }
    if any(len(values) != 1 or "" in values for values in versions.values()):
        raise ValueError(f"Invalid energy-model metadata: {versions}")
    if len({next(iter(values)) for values in versions.values()}) != 1:
        raise ValueError(f"DSE results use different energy model versions: {versions}")
    panel_hashes = {
        workload: {row.get("design_panel_sha256", "") for row in rows}
        for workload, rows in metadata.items()
    }
    if any(len(values) != 1 or "" in values for values in panel_hashes.values()):
        raise ValueError(f"Invalid design-panel metadata: {panel_hashes}")
    if len({next(iter(values)) for values in panel_hashes.values()}) != 1:
        raise ValueError(f"DSE results use different design panels: {panel_hashes}")


def dse_workloads(args: argparse.Namespace) -> dict[str, list[dict[str, str]]]:
    paths = {
        "ResNet-18": args.resnet_result_dir / f"{CNN_PREFIX}_aggregate.csv",
        "ViT-B/16": args.vit_result_dir / f"{VIT_PREFIX}_aggregate.csv",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        message = "DSE outputs are not ready: " + ", ".join(str(path) for path in missing)
        if args.require_dse_results:
            raise FileNotFoundError(message)
        print(message)
        return {}
    validate_run_metadata(args)
    return {workload: read_csv(path) for workload, path in paths.items()}


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    macro_rows = macro_characterization_rows()
    draw_macro_characterization(macro_rows, args.figure_dir)
    write_csv(args.table_dir / "innovation1_macro_characterization.csv", macro_rows)

    shapes = workload_shapes()
    shape_rows = workload_shape_rows(shapes)
    utilization = utilization_rows(shapes)
    draw_reduction_and_utilization(shapes, utilization, args.figure_dir)
    write_csv(args.table_dir / "innovation1_workload_reduction_dimensions.csv", shape_rows)
    write_csv(args.table_dir / "innovation1_macro_row_utilization.csv", utilization)

    aggregate_workloads = dse_workloads(args)
    if aggregate_workloads:
        edp_workloads = {
            workload: min_edp_architectures(rows)
            for workload, rows in aggregate_workloads.items()
        }
        draw_dse_pareto(edp_workloads, args.figure_dir)
        draw_latency_eap_pareto(edp_workloads, args.figure_dir)
        pareto_points = [
            {
                "workload": workload,
                **row,
                "on_energy_adp_pareto": int(
                    str(row["design"])
                    in {
                        str(item["design"])
                        for item in pareto_energy_adp(edp_workloads[workload])
                    }
                ),
            }
            for workload, rows in edp_workloads.items()
            for row in rows
        ]
        write_csv(
            args.table_dir / "innovation1_tile_energy_adp_pareto_points.csv",
            pareto_points,
        )
        latency_eap_points = [
            {
                "workload": workload,
                **row,
                "on_latency_eap_pareto": int(
                    str(row["design"])
                    in {
                        str(item["design"])
                        for item in pareto_latency_eap(edp_workloads[workload])
                    }
                ),
            }
            for workload, rows in edp_workloads.items()
            for row in rows
        ]
        write_csv(
            args.table_dir / "innovation1_tile_latency_energy_area_pareto_points.csv",
            latency_eap_points,
        )
        comparison = comparison_rows(edp_workloads)
        write_csv(args.table_dir / "innovation1_tile_comparison.csv", comparison)
        write_comparison_markdown(
            args.table_dir / "innovation1_tile_comparison.md", comparison
        )

    print(f"Figures: {args.figure_dir}")
    print(f"Tables: {args.table_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the Tile-only scheduler for the complete ResNet-18 workload."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import accelforge as af

from load_sram_spec import load_sram_spec
from sram_resnet18_weight_stationary_scheduler import evaluate_resnet18_weight_stationary_scheduler
from sram_macro_types import load_sram_macro_library


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "experiments" / "results"
RESULTS = OUTPUT_DIR / "sram_resnet18_tile_scheduler_results.csv"
SUMMARY = OUTPUT_DIR / "sram_resnet18_tile_scheduler_summary.csv"
REPORT = OUTPUT_DIR / "sram_resnet18_tile_scheduler.md"


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["design"])].append(row)
    summary_rows = []
    for design, group in grouped.items():
        macs = sum(int(row["MACs"]) for row in group)
        macro_energy_j = sum(float(row["macro_energy_j"]) for row in group)
        psum_energy_j = sum(float(row["psum_energy_j"]) for row in group)
        fabric_energy_j = macro_energy_j + psum_energy_j
        summary_rows.append(
            {
                "design": design,
                "layers": len(group),
                "tile_macro_area_mm2": group[0]["tile_macro_area_mm2"],
                "system_macro_area_mm2": 4 * float(group[0]["tile_macro_area_mm2"]),
                "MACs": macs,
                "macro_invocations": sum(int(row["macro_invocations"]) for row in group),
                "psum_vectors": sum(int(row["psum_vectors"]) for row in group),
                "local_psum_vectors": sum(int(row["local_psum_vectors"]) for row in group),
                "inter_tile_psum_vectors": sum(
                    int(row["inter_tile_psum_vectors"]) for row in group
                ),
                "macro_energy_mJ": macro_energy_j * 1e3,
                "psum_energy_mJ": psum_energy_j * 1e3,
                "fabric_energy_mJ": fabric_energy_j * 1e3,
                "latency_ms": sum(float(row["latency_s"]) for row in group) * 1e3,
                "effective_TOPS_per_W": 2 * macs / fabric_energy_j / 1e12,
            }
        )
    return summary_rows


def _report(rows: list[dict[str, float | int | str]], summary_rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "# ResNet-18 Tile Scheduler Study",
        "",
        "This study includes Macro VMM energy, Tile-local vector partial-sum energy, and",
        "Macro VMM latency. It excludes line FIFOs, GLB, NoC, DRAM, leakage, and all",
        "data-transfer latency. Partial-sum latency is intentionally ignored.",
        "",
        "A layer is spatially replicated only when its complete weight tensor fits in one",
        "Tile. Otherwise, four Tiles retain distinct K/N weight blocks and execute all spatial",
        "positions for their assigned blocks. Cross-Tile psum is counted with the same vector",
        "add/register proxy as local psum; its communication energy and latency are excluded.",
        "",
        "## Network Summary",
        "",
        "| Design | Tile macro area (mm2) | Macro energy (mJ) | Psum energy (mJ) | Inter-Tile psum vectors | Fabric energy (mJ) | VMM latency (ms) | Effective TOPS/W |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['design']} | {float(row['tile_macro_area_mm2']):.4f} | "
            f"{float(row['macro_energy_mJ']):.4f} | {float(row['psum_energy_mJ']):.4f} | "
            f"{int(row['inter_tile_psum_vectors'])} | {float(row['fabric_energy_mJ']):.4f} | "
            f"{float(row['latency_ms']):.4f} | "
            f"{float(row['effective_TOPS_per_W']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Layer Schedule",
            "",
            "| Layer | Design | K | N | Weight mapping | Replicas | Active Tiles | Row utilization | Local / inter-Tile psum | Macro energy (uJ) | Psum energy (uJ) | VMM latency (us) |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['layer']} | {row['design']} | {row['K']} | {row['N']} | "
            f"{row['weight_mapping']} | {row['weight_replication_factor']} | {row['active_tiles']} | "
            f"{float(row['macro_row_utilization']) * 100:.2f}% | "
            f"{row['local_psum_vectors']} / {row['inter_tile_psum_vectors']} | "
            f"{float(row['macro_energy_j']) * 1e6:.3f} | {float(row['psum_energy_j']) * 1e6:.3f} | "
            f"{float(row['latency_s']) * 1e6:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    spec = load_sram_spec("system_digital_sram_4x4")
    workload_path = af.examples.workloads.compute_in_memory.resnet18
    spec.workload = af.Workload.from_yaml(
        workload_path, top_key="workload", jinja_parse_data={"BATCH_SIZE": 1}
    )
    spec.renames = af.Renames.from_yaml(
        workload_path, top_key="renames", jinja_parse_data={"BATCH_SIZE": 1}
    )
    mappings = spec.map_workload_to_arch(print_progress=False)
    rows = evaluate_resnet18_weight_stationary_scheduler(
        mappings, load_sram_macro_library("system_digital_sram_4x4")
    )
    summary_rows = _summary(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULTS, rows)
    _write_csv(SUMMARY, summary_rows)
    REPORT.write_text(_report(rows, summary_rows), encoding="ascii")

    print("ResNet-18 Tile-only scheduler")
    for row in summary_rows:
        print(
            f"{row['design']}: {float(row['fabric_energy_mJ']):.4f} mJ, "
            f"{float(row['latency_ms']):.4f} ms, {float(row['effective_TOPS_per_W']):.3f} TOPS/W"
        )
    print(f"Per-layer results: {RESULTS}")
    print(f"Summary: {SUMMARY}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()

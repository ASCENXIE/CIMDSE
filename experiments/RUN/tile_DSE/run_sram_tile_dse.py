#!/usr/bin/env python3
"""Exhaustively evaluate a peak-compute window of SRAM-CIM Tile compositions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys


RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import accelforge as af

from load_sram_spec import load_sram_spec
from sram_macro_types import load_sram_macro_library
from tile_DSE.sram_tile_dse import (
    enumerate_tile_designs,
    evaluate_tile_schedules,
    layer_shapes,
    pareto_energy_latency,
)


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experiments" / "results" / "tile_DSE"
DEFAULT_LAYERS = "T0,T1,T7,T12,T17,T20"
RESULT_PREFIX = "tile_window_dse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-peak-compute-tops", type=float, default=0.4608)
    parser.add_argument("--peak-compute-tolerance-ratio", type=float, default=0.20)
    parser.add_argument("--peak-compute-atol-tops", type=float, default=1e-12)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--latency-cap-ratio", type=float, default=4.0)
    parser.add_argument("--local-pareto-epsilon", type=float, default=0.001)
    parser.add_argument("--local-state-cap", type=int, default=4096)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _layer_index(name: str) -> int:
    return int(name.removeprefix("T"))


def _compress_energy_latency_front(
    rows: list[dict[str, float | int | str]],
    state_cap: int,
    initial_epsilon: float,
) -> tuple[list[dict[str, float | int | str]], float]:
    if len(rows) <= state_cap:
        return rows, 0.0
    if initial_epsilon <= 0:
        raise RuntimeError(
            f"Exact aggregate front has {len(rows)} states, exceeding the "
            f"state cap {state_cap}; increase the cap or use epsilon > 0."
        )
    epsilon = initial_epsilon
    ordered = sorted(rows, key=lambda row: float(row["fabric_energy_j"]))
    while True:
        kept = [ordered[0]]
        for row in ordered[1:-1]:
            if float(row["latency_s"]) < float(kept[-1]["latency_s"]) * (1 - epsilon):
                kept.append(row)
        kept.append(ordered[-1])
        if len(kept) <= state_cap:
            return kept, epsilon
        epsilon *= 1.5


def aggregate(
    rows: list[dict[str, float | int | str]],
    state_cap: int = 512,
    epsilon: float = 0.001,
) -> list[dict[str, float | int | str]]:
    grouped: dict[str, dict[str, list[dict[str, float | int | str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[str(row["design"])][str(row["layer"])].append(row)
    output = []
    for design, layers in grouped.items():
        first_layer = next(iter(layers.values()))[0]
        effective_epsilon = 0.0
        states = [
            {
                "macro_energy_j": 0.0,
                "macro_dynamic_energy_j": 0.0,
                "macro_static_energy_j": 0.0,
                "input_register_energy_j": 0.0,
                "output_register_energy_j": 0.0,
                "register_energy_j": 0.0,
                "psum_energy_j": 0.0,
                "cross_macro_psum_energy_j": 0.0,
                "cross_wave_psum_energy_j": 0.0,
                "fabric_energy_j": 0.0,
                "bus_energy_j": 0.0,
                "latency_s": 0.0,
                "compute_only_latency_s": 0.0,
                "weight_load_makespan_s": 0.0,
                "weight_load_added_latency_s": 0.0,
                "bus_added_latency_s": 0.0,
                "mapping_counts": (0, 0, 0),
                "peak_active_output_lanes": 0,
                "schedule_trace": (),
                "input_register_read_entries": 0,
                "input_register_write_entries": 0,
                "output_register_read_entries": 0,
                "output_register_write_entries": 0,
                "weight_load_values": 0,
                "weight_load_rows": 0,
                "weight_load_energy_j": 0.0,
                "weight_load_events": 0,
                "feature_bus_bits": 0,
                "weight_bus_bits": 0,
                "feature_bus_cycles": 0,
                "weight_bus_cycles": 0,
                "total_bus_cycles": 0,
                "cross_macro_psum_vectors": 0,
                "cross_wave_psum_vectors": 0,
                "fully_resident_layers": 0,
                "weight_residency_epochs": 0,
            }
        ]
        for layer_name in sorted(layers, key=_layer_index):
            options = pareto_energy_latency(layers[layer_name])
            candidates = []
            for state in states:
                for option in options:
                    counts = list(state["mapping_counts"])
                    counts[("k_parallel", "n_parallel", "mixed").index(str(option["mapping"]))] += 1
                    candidates.append(
                        {
                            "macro_energy_j": float(state["macro_energy_j"])
                            + float(option["macro_energy_j"]),
                            "macro_dynamic_energy_j": float(
                                state["macro_dynamic_energy_j"]
                            )
                            + float(option["macro_dynamic_energy_j"]),
                            "macro_static_energy_j": float(
                                state["macro_static_energy_j"]
                            )
                            + float(option["macro_static_energy_j"]),
                            "input_register_energy_j": float(
                                state["input_register_energy_j"]
                            )
                            + float(option["input_register_energy_j"]),
                            "output_register_energy_j": float(
                                state["output_register_energy_j"]
                            )
                            + float(option["output_register_energy_j"]),
                            "register_energy_j": float(state["register_energy_j"])
                            + float(option["register_energy_j"]),
                            "psum_energy_j": float(state["psum_energy_j"])
                            + float(option["psum_energy_j"]),
                            "cross_macro_psum_energy_j": float(
                                state["cross_macro_psum_energy_j"]
                            )
                            + float(option["cross_macro_psum_energy_j"]),
                            "cross_wave_psum_energy_j": float(
                                state["cross_wave_psum_energy_j"]
                            )
                            + float(option["cross_wave_psum_energy_j"]),
                            "fabric_energy_j": float(state["fabric_energy_j"])
                            + float(option["fabric_energy_j"]),
                            "bus_energy_j": float(state["bus_energy_j"])
                            + float(option["bus_energy_j"]),
                            "latency_s": float(state["latency_s"]) + float(option["latency_s"]),
                            "compute_only_latency_s": float(
                                state["compute_only_latency_s"]
                            )
                            + float(option["compute_only_latency_s"]),
                            "weight_load_makespan_s": float(
                                state["weight_load_makespan_s"]
                            )
                            + float(option["weight_load_makespan_s"]),
                            "weight_load_added_latency_s": float(
                                state["weight_load_added_latency_s"]
                            )
                            + float(option["weight_load_added_latency_s"]),
                            "bus_added_latency_s": float(
                                state["bus_added_latency_s"]
                            )
                            + float(option["bus_added_latency_s"]),
                            "mapping_counts": tuple(counts),
                            "peak_active_output_lanes": max(
                                int(state["peak_active_output_lanes"]),
                                int(option["active_lane_count"]),
                            ),
                            "schedule_trace": state["schedule_trace"]
                            + (
                                f"{layer_name}:{option['mapping']}[{option['schedule_signature']}]",
                            ),
                            "input_register_read_entries": int(
                                state["input_register_read_entries"]
                            )
                            + int(option["input_register_read_entries"]),
                            "input_register_write_entries": int(
                                state["input_register_write_entries"]
                            )
                            + int(option["input_register_write_entries"]),
                            "output_register_read_entries": int(
                                state["output_register_read_entries"]
                            )
                            + int(option["output_register_read_entries"]),
                            "output_register_write_entries": int(
                                state["output_register_write_entries"]
                            )
                            + int(option["output_register_write_entries"]),
                            "weight_load_values": int(state["weight_load_values"])
                            + int(option["weight_load_values"]),
                            "weight_load_rows": int(state["weight_load_rows"])
                            + int(option["weight_load_rows"]),
                            "weight_load_energy_j": float(
                                state["weight_load_energy_j"]
                            )
                            + float(option["weight_load_energy_j"]),
                            "weight_load_events": int(state["weight_load_events"])
                            + int(option["weight_load_events"]),
                            "feature_bus_bits": int(state["feature_bus_bits"])
                            + int(option["feature_bus_bits"]),
                            "weight_bus_bits": int(state["weight_bus_bits"])
                            + int(option["weight_bus_bits"]),
                            "feature_bus_cycles": int(state["feature_bus_cycles"])
                            + int(option["feature_bus_cycles"]),
                            "weight_bus_cycles": int(state["weight_bus_cycles"])
                            + int(option["weight_bus_cycles"]),
                            "total_bus_cycles": int(state["total_bus_cycles"])
                            + int(option["total_bus_cycles"]),
                            "cross_macro_psum_vectors": int(
                                state["cross_macro_psum_vectors"]
                            )
                            + int(option["cross_macro_psum_vectors"]),
                            "cross_wave_psum_vectors": int(
                                state["cross_wave_psum_vectors"]
                            )
                            + int(option["cross_wave_psum_vectors"]),
                            "fully_resident_layers": int(state["fully_resident_layers"])
                            + int(option["weight_fits_tile"]),
                            "weight_residency_epochs": int(state["weight_residency_epochs"])
                            + int(option["weight_residency_epochs"]),
                        }
                    )
            states, layer_epsilon = _compress_energy_latency_front(
                pareto_energy_latency(candidates), state_cap, epsilon
            )
            effective_epsilon = max(effective_epsilon, layer_epsilon)
        macs = sum(int(options[0]["MACs"]) for options in layers.values())
        tile_class = "homogeneous" if "+" not in design else "heterogeneous"
        for state in states:
            k_count, n_count, mixed_count = state["mapping_counts"]
            energy = float(state["fabric_energy_j"])
            output.append(
                {
                    "design": design,
                    "tile_class": tile_class,
                    "mapping_summary": f"K={k_count};N={n_count};M={mixed_count}",
                    "macro_counts": first_layer["macro_counts"],
                    "tile_capacity_values": first_layer["tile_capacity_values"],
                    "capacity_ratio": first_layer["capacity_ratio"],
                    "tile_macro_area_mm2": first_layer["tile_macro_area_mm2"],
                    "tile_register_area_mm2": first_layer[
                        "tile_register_area_mm2"
                    ],
                    "tile_area_mm2": first_layer["tile_area_mm2"],
                    "area_ratio": first_layer["area_ratio"],
                    "tile_macro_count": first_layer["tile_macro_count"],
                    "tile_macro_type_count": first_layer["tile_macro_type_count"],
                    "tile_peak_compute_tops": first_layer[
                        "tile_peak_compute_tops"
                    ],
                    "tile_peak_compute_ratio": first_layer[
                        "tile_peak_compute_ratio"
                    ],
                    "tile_peak_compute_density_tops_per_mm2": first_layer[
                        "tile_peak_compute_density_tops_per_mm2"
                    ],
                    "max_macro_types_per_tile": first_layer[
                        "max_macro_types_per_tile"
                    ],
                    "tile_output_lanes": first_layer["tile_output_lanes"],
                    "peak_active_output_lanes": state["peak_active_output_lanes"],
                    "macro_row_span": first_layer["macro_row_span"],
                    "layers": len(layers),
                    "MACs": macs,
                    "macro_energy_mJ": float(state["macro_energy_j"]) * 1e3,
                    "macro_dynamic_energy_mJ": float(
                        state["macro_dynamic_energy_j"]
                    )
                    * 1e3,
                    "macro_static_energy_mJ": float(
                        state["macro_static_energy_j"]
                    )
                    * 1e3,
                    "input_register_energy_mJ": float(
                        state["input_register_energy_j"]
                    )
                    * 1e3,
                    "output_register_energy_mJ": float(
                        state["output_register_energy_j"]
                    )
                    * 1e3,
                    "register_energy_mJ": float(state["register_energy_j"]) * 1e3,
                    "psum_energy_mJ": float(state["psum_energy_j"]) * 1e3,
                    "cross_macro_psum_energy_mJ": float(
                        state["cross_macro_psum_energy_j"]
                    )
                    * 1e3,
                    "cross_wave_psum_energy_mJ": float(
                        state["cross_wave_psum_energy_j"]
                    )
                    * 1e3,
                    "bus_energy_mJ": float(state["bus_energy_j"]) * 1e3,
                    "fabric_energy_mJ": energy * 1e3,
                    "latency_ms": float(state["latency_s"]) * 1e3,
                    "compute_only_latency_ms": float(
                        state["compute_only_latency_s"]
                    )
                    * 1e3,
                    "weight_load_makespan_ms": float(
                        state["weight_load_makespan_s"]
                    )
                    * 1e3,
                    "weight_load_added_latency_ms": float(
                        state["weight_load_added_latency_s"]
                    )
                    * 1e3,
                    "bus_added_latency_ms": float(state["bus_added_latency_s"])
                    * 1e3,
                    "input_register_read_entries": state[
                        "input_register_read_entries"
                    ],
                    "input_register_write_entries": state[
                        "input_register_write_entries"
                    ],
                    "output_register_read_entries": state[
                        "output_register_read_entries"
                    ],
                    "output_register_write_entries": state[
                        "output_register_write_entries"
                    ],
                    "weight_storage_values": first_layer["tile_capacity_values"],
                    "weight_load_values": state["weight_load_values"],
                    "weight_load_rows": state["weight_load_rows"],
                    "weight_load_events": state["weight_load_events"],
                    "feature_bus_bits": state["feature_bus_bits"],
                    "weight_bus_bits": state["weight_bus_bits"],
                    "feature_bus_cycles": state["feature_bus_cycles"],
                    "weight_bus_cycles": state["weight_bus_cycles"],
                    "total_bus_cycles": state["total_bus_cycles"],
                    "shared_bus_width_bits": first_layer["shared_bus_width_bits"],
                    "tile_retain_features_across_nblocks": first_layer[
                        "tile_retain_features_across_nblocks"
                    ],
                    "tile_allow_intra_bank_weight_compute_overlap": first_layer[
                        "tile_allow_intra_bank_weight_compute_overlap"
                    ],
                    "vit_allow_cross_bank_weight_compute_overlap": first_layer[
                        "vit_allow_cross_bank_weight_compute_overlap"
                    ],
                    "tile_latency_batch_granularity": first_layer[
                        "tile_latency_batch_granularity"
                    ],
                    "tile_include_pipeline_fill_drain": first_layer[
                        "tile_include_pipeline_fill_drain"
                    ],
                    "bus_energy_per_bit_j": first_layer["bus_energy_per_bit_j"],
                    "cross_macro_psum_vectors": state[
                        "cross_macro_psum_vectors"
                    ],
                    "cross_wave_psum_vectors": state["cross_wave_psum_vectors"],
                    "core_psum_buffer_vectors": 0,
                    "cross_wave_psum_resolved_in_tile": 1,
                    "weight_load_energy_mJ": float(
                        state["weight_load_energy_j"]
                    )
                    * 1e3,
                    "fully_resident_layers": state["fully_resident_layers"],
                    "weight_residency_epochs": state["weight_residency_epochs"],
                    "schedule_trace": ";".join(state["schedule_trace"]),
                    "aggregation_epsilon": effective_epsilon,
                    "aggregation_state_cap": state_cap,
                    "effective_TOPS_per_W": 2 * macs / energy / 1e12,
                }
            )
    return output


def aggregate_energy_latency_area_pareto(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    objectives = ("fabric_energy_mJ", "latency_ms", "tile_area_mm2")
    ordered = sorted(
        rows,
        key=lambda row: (
            *(float(row[key]) for key in objectives),
            str(row.get("design", "")),
        ),
    )
    front: list[dict[str, float | int | str]] = []
    for row in ordered:
        values = tuple(float(row[key]) for key in objectives)
        if any(
            all(float(existing[key]) <= value for key, value in zip(objectives, values))
            for existing in front
        ):
            continue
        front = [
            existing
            for existing in front
            if not all(
                value <= float(existing[key])
                for key, value in zip(objectives, values)
            )
        ]
        front.append(row)
    return sorted(
        front,
        key=lambda row: tuple(float(row[key]) for key in objectives),
    )


def main() -> None:
    args = parse_args()
    if args.tile_peak_compute_tops <= 0:
        raise ValueError("tile-peak-compute-tops must be positive.")
    if not 0 <= args.peak_compute_tolerance_ratio < 1:
        raise ValueError("peak-compute-tolerance-ratio must be in [0, 1).")
    if args.peak_compute_atol_tops < 0:
        raise ValueError("peak-compute-atol-tops cannot be negative.")
    if args.latency_cap_ratio < 1:
        raise ValueError("latency-cap-ratio must be at least 1.")
    if not 0 <= args.local_pareto_epsilon < 1:
        raise ValueError("local-pareto-epsilon must be in [0, 1).")
    if args.local_state_cap < 2:
        raise ValueError("local-state-cap must be at least 2.")
    selected_layers = None if args.all_layers else set(args.layers.split(","))
    spec = load_sram_spec("system_digital_sram_4x4")
    workload_path = af.examples.workloads.compute_in_memory.resnet18
    spec.workload = af.Workload.from_yaml(
        workload_path, top_key="workload", jinja_parse_data={"BATCH_SIZE": 1}
    )
    spec.renames = af.Renames.from_yaml(
        workload_path, top_key="renames", jinja_parse_data={"BATCH_SIZE": 1}
    )
    mappings = spec.map_workload_to_arch(print_progress=False)
    library = load_sram_macro_library("system_digital_sram_4x4")
    reference, designs = enumerate_tile_designs(
        library,
        args.tile_peak_compute_tops,
        args.peak_compute_tolerance_ratio,
        args.peak_compute_atol_tops,
    )
    tech_node = float(mappings.evaluated_specs[mappings.einsum_names[0]].variables.tech_node)
    layers = layer_shapes(mappings, selected_layers)
    rows = []
    schedule_cache: dict[tuple[int, int, int], list[dict[str, float | int | str]]] = {}
    for layer_index, layer in enumerate(layers, start=1):
        shape_key = (
            layer.reduction_length,
            layer.output_channels,
            layer.output_positions,
        )
        cache_hit = shape_key in schedule_cache
        if cache_hit:
            layer_rows = [
                {**row, "layer": layer.name} for row in schedule_cache[shape_key]
            ]
        else:
            layer_rows = [
                row
                for design in designs
                for row in evaluate_tile_schedules(
                    layer,
                    design,
                    library,
                    tech_node,
                    reference,
                    args.latency_cap_ratio,
                    args.local_pareto_epsilon,
                    local_state_cap=args.local_state_cap,
                )
            ]
            schedule_cache[shape_key] = layer_rows
        rows.extend(layer_rows)
        print(
            f"Layer {layer_index}/{len(layers)} {layer.name}: "
            f"K={layer.reduction_length}, N={layer.output_channels}, "
            f"local Pareto schedules={len(layer_rows)}, cache={'hit' if cache_hit else 'miss'}",
            flush=True,
        )
    aggregate_rows = aggregate(rows)
    layer_pareto_rows = [
        row
        for layer in sorted({str(row["layer"]) for row in rows})
        for row in pareto_energy_latency(
            [candidate for candidate in rows if candidate["layer"] == layer]
        )
    ]
    aggregate_pareto_rows = pareto_energy_latency(
        [
            {
                **row,
                "fabric_energy_j": float(row["fabric_energy_mJ"]) * 1e-3,
                "latency_s": float(row["latency_ms"]) * 1e-3,
            }
            for row in aggregate_rows
        ]
    )
    aggregate_pareto_eta_rows = aggregate_energy_latency_area_pareto(
        aggregate_rows
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS_DIR / f"{RESULT_PREFIX}_layers.csv", rows)
    write_csv(RESULTS_DIR / f"{RESULT_PREFIX}_layer_pareto.csv", layer_pareto_rows)
    write_csv(RESULTS_DIR / f"{RESULT_PREFIX}_aggregate.csv", aggregate_rows)
    write_csv(RESULTS_DIR / f"{RESULT_PREFIX}_aggregate_pareto.csv", aggregate_pareto_rows)
    write_csv(
        RESULTS_DIR / f"{RESULT_PREFIX}_aggregate_pareto_eta.csv",
        aggregate_pareto_eta_rows,
    )
    print(f"Reference: {reference.name}")
    lower_tops = args.tile_peak_compute_tops * (1 - args.peak_compute_tolerance_ratio)
    upper_tops = args.tile_peak_compute_tops * (1 + args.peak_compute_tolerance_ratio)
    print(f"Tile peak-compute reference: {args.tile_peak_compute_tops:.6f} TOPS")
    print(f"Tile peak-compute window: [{lower_tops:.6f}, {upper_tops:.6f}] TOPS")
    print(f"Candidates: {len(designs)}")
    print(f"Layer rows: {len(rows)}")
    print(f"Aggregate Pareto points: {len(aggregate_pareto_rows)}")
    print(f"Aggregate E-T-A Pareto points: {len(aggregate_pareto_eta_rows)}")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

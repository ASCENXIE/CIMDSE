#!/usr/bin/env python3
"""Bayesian outer-loop search with exact per-Tile ResNet-18 scheduling."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import accelforge as af
import numpy as np

from load_sram_spec import load_sram_spec
from run_sram_tile_dse import aggregate, write_csv
from sram_macro_types import load_sram_macro_library
from sram_tile_dse import (
    TileDesign,
    enumerate_tile_designs,
    evaluate_tile_schedules,
    layer_shapes,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"
MACRO_NAMES = ("m32x8", "m64x8", "m128x8", "m256x8", "m512x8")
ENERGY_MODEL_VERSION = "macro_psum_aladdin_register_v1"
ARCHITECTURE_MODEL_VERSION = "max_two_macro_types_four_output_lanes_v1"
AREA_MODEL_VERSION = "macro_plus_aladdin_register_area_v1"
LATENCY_MODEL_VERSION = "vmm_plus_parallel_macro_row_weight_load_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-tolerance", type=float, default=0.20)
    parser.add_argument("--area-tolerance", type=float, default=0.20)
    parser.add_argument("--design-budget", type=int, default=40)
    parser.add_argument("--initial-designs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--latency-cap-ratio", type=float, default=2.0)
    parser.add_argument("--local-pareto-epsilon", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def features(designs: list[TileDesign], reference: TileDesign) -> np.ndarray:
    values = []
    for design in designs:
        counts = dict(design.counts)
        values.append(
            [
                *(counts.get(name, 0) for name in MACRO_NAMES),
                design.capacity_values / reference.capacity_values,
                design.area_mm2 / reference.area_mm2,
                design.macro_count,
            ]
        )
    matrix = np.asarray(values, dtype=float)
    span = np.ptp(matrix, axis=0)
    span[span == 0] = 1.0
    return (matrix - matrix.min(axis=0)) / span


def farthest_seeds(
    designs: list[TileDesign], x: np.ndarray, count: int
) -> list[int]:
    forced_names = ("9xm64x8", "1xm64x8+1xm512x8")
    selected = [
        next(index for index, design in enumerate(designs) if design.name == name)
        for name in forced_names
    ]
    selected.extend(
        index for index, design in enumerate(designs) if len(design.counts) == 1
    )
    selected = list(dict.fromkeys(selected))
    while len(selected) < count:
        remaining = [index for index in range(len(designs)) if index not in selected]
        next_index = max(
            remaining,
            key=lambda index: min(
                float(np.linalg.norm(x[index] - x[chosen])) for chosen in selected
            ),
        )
        selected.append(next_index)
    return selected[:count]


def gp_predict(
    train_x: np.ndarray, train_y: np.ndarray, candidate_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(train_x) == 1:
        return (
            np.full(len(candidate_x), train_y[0]),
            np.ones(len(candidate_x)),
        )
    pairwise = np.linalg.norm(train_x[:, None, :] - train_x[None, :, :], axis=2)
    positive = pairwise[pairwise > 0]
    length_scale = max(float(np.median(positive)), 0.15)

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance_sq = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * distance_sq / length_scale**2)

    kernel_train = kernel(train_x, train_x) + np.eye(len(train_x)) * 1e-6
    centered = (train_y - train_y.mean()) / max(float(train_y.std()), 1e-9)
    inverse_target = np.linalg.solve(kernel_train, centered)
    cross = kernel(candidate_x, train_x)
    mean = train_y.mean() + max(float(train_y.std()), 1e-9) * (cross @ inverse_target)
    solved = np.linalg.solve(kernel_train, cross.T)
    variance = np.maximum(1.0 - np.sum(cross * solved.T, axis=1), 1e-9)
    return mean, np.sqrt(variance)


def select_batch(
    x: np.ndarray,
    evaluated: list[int],
    summaries: dict[int, tuple[float, float, float]],
    batch_size: int,
) -> list[int]:
    remaining = [index for index in range(len(x)) if index not in evaluated]
    train_x = x[evaluated]
    targets = np.log(np.asarray([summaries[index] for index in evaluated]))
    means = []
    uncertainties = []
    for column in range(targets.shape[1]):
        mean, uncertainty = gp_predict(train_x, targets[:, column], x[remaining])
        means.append(mean)
        uncertainties.append(uncertainty)
    mean_matrix = np.column_stack(means)
    uncertainty_matrix = np.column_stack(uncertainties)
    normalized = (mean_matrix - mean_matrix.min(axis=0)) / np.maximum(
        np.ptp(mean_matrix, axis=0), 1e-9
    )
    acquisition = normalized.min(axis=1) - 1.5 * uncertainty_matrix.mean(axis=1)
    selected = []
    while remaining and len(selected) < batch_size:
        position = int(np.argmin(acquisition))
        chosen = remaining[position]
        selected.append(chosen)
        distance = np.linalg.norm(x[remaining] - x[chosen], axis=1)
        acquisition += 0.25 * np.exp(-(distance**2) / 0.08)
        acquisition[position] = np.inf
    return selected


def evaluate_design(
    design: TileDesign,
    layers,
    library,
    tech_node: float,
    reference: TileDesign,
    latency_cap_ratio: float,
    local_pareto_epsilon: float,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    layer_rows = []
    shape_cache = {}
    for layer in layers:
        key = (
            layer.reduction_length,
            layer.output_channels,
            layer.output_positions,
        )
        if key not in shape_cache:
            shape_cache[key] = evaluate_tile_schedules(
                layer,
                design,
                library,
                tech_node,
                reference,
                latency_cap_ratio,
                local_pareto_epsilon,
            )
        layer_rows.extend({**row, "layer": layer.name} for row in shape_cache[key])
    return layer_rows, aggregate(layer_rows)


def summarize(rows: list[dict[str, float | int | str]]) -> tuple[float, float, float]:
    energy = np.asarray([float(row["fabric_energy_mJ"]) for row in rows])
    latency = np.asarray([float(row["latency_ms"]) for row in rows])
    return float(energy.min()), float(latency.min()), float((energy * latency).min())


def compact_aggregate_rows(
    rows: list[dict[str, float | int | str]], state_cap: int = 512
) -> list[dict[str, float | int | str]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["design"]), []).append(row)
    output = []
    for candidates in grouped.values():
        ordered = sorted(
            candidates,
            key=lambda row: (
                float(row["fabric_energy_mJ"]),
                float(row["latency_ms"]),
            ),
        )
        front = []
        best_latency = float("inf")
        for row in ordered:
            latency = float(row["latency_ms"])
            if latency < best_latency:
                front.append(row)
                best_latency = latency
        epsilon = 0.001
        compressed = len(front) > state_cap
        while len(front) > state_cap:
            kept = [front[0]]
            for row in front[1:-1]:
                if float(row["latency_ms"]) < float(kept[-1]["latency_ms"]) * (
                    1 - epsilon
                ):
                    kept.append(row)
            kept.append(front[-1])
            front = kept
            epsilon *= 1.5
        effective_epsilon = epsilon / 1.5 if compressed else 0.0
        for row in front:
            row["aggregation_epsilon"] = max(
                float(row.get("aggregation_epsilon", 0.0)), effective_epsilon
            )
            row["aggregation_state_cap"] = state_cap
        output.extend(front)
    return output


def aggregate_resource_pareto(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["design"]), []).append(row)
    output = []
    for own_rows in grouped.values():
        area = float(own_rows[0]["tile_area_mm2"])
        macro_count = float(own_rows[0]["tile_macro_count"])
        eligible = [
            row
            for row in rows
            if float(row["tile_area_mm2"]) <= area
            and float(row["tile_macro_count"]) <= macro_count
        ]
        eligible_front = []
        best_latency = float("inf")
        for row in sorted(
            eligible,
            key=lambda row: (
                float(row["fabric_energy_mJ"]),
                float(row["latency_ms"]),
            ),
        ):
            latency = float(row["latency_ms"])
            if latency < best_latency:
                eligible_front.append(row)
                best_latency = latency
        for candidate in own_rows:
            if any(
                float(other["fabric_energy_mJ"])
                <= float(candidate["fabric_energy_mJ"])
                and float(other["latency_ms"]) <= float(candidate["latency_ms"])
                and (
                    float(other["fabric_energy_mJ"])
                    < float(candidate["fabric_energy_mJ"])
                    or float(other["latency_ms"]) < float(candidate["latency_ms"])
                    or float(other["tile_area_mm2"]) < area
                    or float(other["tile_macro_count"]) < macro_count
                )
                for other in eligible_front
            ):
                continue
            output.append(candidate)
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def checkpoint(
    designs: list[TileDesign],
    evaluated: list[int],
    summaries: dict[int, tuple[float, float, float]],
    layer_rows: list[dict[str, float | int | str]],
    aggregate_rows: list[dict[str, float | int | str]],
    latency_cap_ratio: float,
    local_pareto_epsilon: float,
) -> None:
    write_csv(RESULTS_DIR / "tile_dse_layers.csv", layer_rows)
    write_csv(RESULTS_DIR / "tile_dse_aggregate.csv", aggregate_rows)
    metadata = [
        {
            "design": designs[index].name,
            "selection_order": order,
            "minimum_energy_mJ": summaries[index][0],
            "minimum_latency_ms": summaries[index][1],
            "minimum_EDP_mJ_ms": summaries[index][2],
            "latency_cap_ratio": latency_cap_ratio,
            "local_pareto_epsilon": local_pareto_epsilon,
            "energy_model_version": ENERGY_MODEL_VERSION,
            "architecture_model_version": ARCHITECTURE_MODEL_VERSION,
            "area_model_version": AREA_MODEL_VERSION,
            "latency_model_version": LATENCY_MODEL_VERSION,
            "candidate_pool_size": len(designs),
        }
        for order, index in enumerate(evaluated, start=1)
    ]
    write_csv(RESULTS_DIR / "tile_dse_bayesian_samples.csv", metadata)


def main() -> None:
    args = parse_args()
    if args.design_budget < 1 or args.initial_designs < 1:
        raise ValueError("design-budget and initial-designs must be at least 1")
    if args.initial_designs > args.design_budget:
        raise ValueError("initial-designs cannot exceed design-budget")
    if args.latency_cap_ratio < 1:
        raise ValueError("latency-cap-ratio must be at least 1.")
    if not 0 <= args.local_pareto_epsilon < 1:
        raise ValueError("local-pareto-epsilon must be in [0, 1).")
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")
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
        library, args.capacity_tolerance, args.area_tolerance
    )
    budget = min(args.design_budget, len(designs))
    initial = min(args.initial_designs, budget)
    x = features(designs, reference)
    selected = farthest_seeds(designs, x, initial)
    evaluated = []
    summaries = {}
    all_layer_rows = []
    all_aggregate_rows = []
    layers = layer_shapes(mappings)
    tech_node = float(mappings.evaluated_specs[mappings.einsum_names[0]].variables.tech_node)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume:
        metadata = read_csv(RESULTS_DIR / "tile_dse_bayesian_samples.csv")
        compatible = metadata and all(
            float(row.get("latency_cap_ratio", "nan")) == args.latency_cap_ratio
            and float(row.get("local_pareto_epsilon", "nan"))
            == args.local_pareto_epsilon
            and row.get("energy_model_version") == ENERGY_MODEL_VERSION
            and row.get("architecture_model_version") == ARCHITECTURE_MODEL_VERSION
            and row.get("area_model_version") == AREA_MODEL_VERSION
            and row.get("latency_model_version") == LATENCY_MODEL_VERSION
            for row in metadata
        )
        if metadata and not compatible:
            print("Existing checkpoint uses different scheduler or energy settings; starting fresh.")
            metadata = []
        design_index = {design.name: index for index, design in enumerate(designs)}
        evaluated = [
            design_index[row["design"]]
            for row in sorted(metadata, key=lambda row: int(row["selection_order"]))
            if row["design"] in design_index
        ]
        summaries = {
            design_index[row["design"]]: (
                float(row["minimum_energy_mJ"]),
                float(row["minimum_latency_ms"]),
                float(row["minimum_EDP_mJ_ms"]),
            )
            for row in metadata
            if row["design"] in design_index
        }
        evaluated_names = {designs[index].name for index in evaluated}
        all_layer_rows = [
            row
            for row in read_csv(RESULTS_DIR / "tile_dse_layers.csv")
            if row["design"] in evaluated_names
        ]
        all_aggregate_rows = [
            row
            for row in read_csv(RESULTS_DIR / "tile_dse_aggregate.csv")
            if row["design"] in evaluated_names
        ]
        for row in all_aggregate_rows:
            row.setdefault("aggregation_epsilon", "0.0")
            row.setdefault("aggregation_state_cap", "0")
        all_aggregate_rows = compact_aggregate_rows(all_aggregate_rows)
        print(f"Resuming from {len(evaluated)} evaluated designs", flush=True)

    while len(evaluated) < budget:
        batch = [index for index in selected if index not in evaluated]
        if not batch:
            batch = select_batch(
                x,
                evaluated,
                summaries,
                min(args.batch_size, budget - len(evaluated)),
            )
        pending = batch[: budget - len(evaluated)]
        futures = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            for index in pending:
                design = designs[index]
                print(f"Submitting design: {design.name}", flush=True)
                futures[
                    executor.submit(
                        evaluate_design,
                        design,
                        layers,
                        library,
                        tech_node,
                        reference,
                        args.latency_cap_ratio,
                        args.local_pareto_epsilon,
                    )
                ] = index
            for future in as_completed(futures):
                index = futures[future]
                design = designs[index]
                layer_rows, aggregate_rows = future.result()
                print(
                    f"Completed design {len(evaluated) + 1}/{budget}: {design.name}",
                    flush=True,
                )
                all_layer_rows.extend(layer_rows)
                all_aggregate_rows.extend(aggregate_rows)
                summaries[index] = summarize(aggregate_rows)
                evaluated.append(index)
                checkpoint(
                    designs,
                    evaluated,
                    summaries,
                    all_layer_rows,
                    all_aggregate_rows,
                    args.latency_cap_ratio,
                    args.local_pareto_epsilon,
                )
        selected = []

    all_aggregate_rows = compact_aggregate_rows(all_aggregate_rows)
    aggregate_pareto = aggregate_resource_pareto(all_aggregate_rows)
    write_csv(RESULTS_DIR / "tile_dse_layers.csv", all_layer_rows)
    write_csv(RESULTS_DIR / "tile_dse_aggregate.csv", all_aggregate_rows)
    write_csv(RESULTS_DIR / "tile_dse_aggregate_pareto.csv", aggregate_pareto)
    checkpoint(
        designs,
        evaluated,
        summaries,
        all_layer_rows,
        all_aggregate_rows,
        args.latency_cap_ratio,
        args.local_pareto_epsilon,
    )
    print(f"Evaluated designs: {len(evaluated)}/{len(designs)}")
    print(f"Aggregate Pareto points: {len(aggregate_pareto)}")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

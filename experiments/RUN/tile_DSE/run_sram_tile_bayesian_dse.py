#!/usr/bin/env python3
"""Bayesian Tile search and fixed-panel evaluation across CIM workloads."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import pickle
import sys


RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import accelforge as af
import numpy as np

from load_sram_spec import load_sram_spec
from sram_macro_types import ENERGY_MODEL_VERSION, load_sram_macro_library
from tile_DSE.run_sram_tile_dse import (
    aggregate,
    aggregate_energy_latency_area_pareto,
    write_csv,
)
from tile_DSE.sram_tile_dse import (
    TileDesign,
    enumerate_tile_designs,
    evaluate_tile_schedules,
    layer_shapes_from_workload,
)


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experiments" / "results" / "tile_DSE"
MACRO_NAMES = ("m32x8", "m64x8", "m128x8", "m256x8", "m512x8")
ARCHITECTURE_MODEL_VERSION = "fixed_32_output_registers_shared_512b_bus_four_lanes_v6"
LATENCY_MODEL_VERSION = "nblock_wave_sum_weight_plus_max_feature_compute_v5"
SCHEDULE_SEARCH_VERSION = "reachable_state_bounded_timing_cache_v2"
RESULT_PREFIX = "tile_window_dse"
WORKLOAD_NAMES = ("resnet18", "resnet50", "vgg16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOAD_NAMES, default="resnet18")
    parser.add_argument("--workload-batch-size", type=int, default=1)
    parser.add_argument(
        "--output-subdir",
        default="",
        help="Relative directory below experiments/results/tile_DSE.",
    )
    parser.add_argument(
        "--fixed-designs-csv",
        type=Path,
        help="Evaluate exactly the designs listed in this CSV's design column.",
    )
    parser.add_argument("--tile-peak-compute-tops", type=float, default=0.4608)
    parser.add_argument("--peak-compute-tolerance-ratio", type=float, default=0.20)
    parser.add_argument("--peak-compute-atol-tops", type=float, default=1e-12)
    parser.add_argument("--design-budget", type=int, default=48)
    parser.add_argument("--initial-designs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-designs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--latency-cap-ratio", type=float, default=4.0)
    parser.add_argument("--local-pareto-epsilon", type=float, default=0.001)
    parser.add_argument("--local-state-cap", type=int, default=4096)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent design evaluations; timing caches are bounded per worker.",
    )
    return parser.parse_args()


def workload_path(name: str) -> Path:
    return getattr(af.examples.workloads.compute_in_memory, name)


def result_directory(output_subdir: str) -> Path:
    subdir = Path(output_subdir)
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("output-subdir must be a relative path below the Tile result directory.")
    return RESULTS_DIR / subdir


def fixed_design_names(path: Path) -> list[str]:
    rows = read_csv(path)
    if not rows or "design" not in rows[0]:
        raise ValueError(f"Fixed-design CSV has no design rows: {path}")
    if "selection_order" in rows[0]:
        rows.sort(key=lambda row: int(row["selection_order"]))
    names = list(dict.fromkeys(row["design"] for row in rows))
    if len(names) != len(rows):
        raise ValueError(f"Fixed-design CSV contains duplicate designs: {path}")
    return names


def panel_sha256(names: list[str]) -> str:
    if not names:
        return ""
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def partial_checkpoint_key(
    workload_name: str,
    workload_batch_size: int,
    latency_cap_ratio: float,
    local_pareto_epsilon: float,
    local_state_cap: int,
) -> str:
    payload = "|".join(
        (
            workload_name,
            str(workload_batch_size),
            ENERGY_MODEL_VERSION,
            ARCHITECTURE_MODEL_VERSION,
            LATENCY_MODEL_VERSION,
            SCHEDULE_SEARCH_VERSION,
            str(latency_cap_ratio),
            str(local_pareto_epsilon),
            str(local_state_cap),
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def read_partial_layer_rows(path: Path | None) -> list[dict[str, float | int | str]]:
    if path is None or not path.is_file():
        return []
    with path.open("rb") as file:
        rows = pickle.load(file)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid partial Tile checkpoint: {path}")
    return rows


def write_partial_layer_rows(
    path: Path | None, rows: list[dict[str, float | int | str]]
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(rows, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def features(designs: list[TileDesign], reference: TileDesign) -> np.ndarray:
    values = []
    for design in designs:
        counts = dict(design.counts)
        values.append(
            [
                *(counts.get(name, 0) for name in MACRO_NAMES),
                design.capacity_values / reference.capacity_values,
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
    index_by_name = {design.name: index for index, design in enumerate(designs)}
    selected = [index_by_name[name] for name in forced_names if name in index_by_name]
    selected.extend(
        index for index, design in enumerate(designs) if len(design.counts) == 1
    )
    selected = list(dict.fromkeys(selected))
    if not selected:
        selected.append(0)
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


def predict_remaining(
    x: np.ndarray,
    evaluated: list[int],
    summaries: dict[int, tuple[float, float, float]],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    remaining = [index for index in range(len(x)) if index not in evaluated]
    if not remaining:
        return remaining, np.empty((0, 3)), np.empty((0, 3))
    train_x = x[evaluated]
    targets = np.log(np.asarray([summaries[index] for index in evaluated]))
    means = []
    uncertainties = []
    for column in range(targets.shape[1]):
        mean, uncertainty = gp_predict(train_x, targets[:, column], x[remaining])
        means.append(mean)
        uncertainties.append(uncertainty)
    return remaining, np.column_stack(means), np.column_stack(uncertainties)


def _nondominated_prediction_positions(mean_matrix: np.ndarray) -> list[int]:
    objectives = mean_matrix[:, :2]
    return [
        index
        for index, values in enumerate(objectives)
        if not any(
            other_index != index
            and np.all(other <= values)
            and np.any(other < values)
            for other_index, other in enumerate(objectives)
        )
    ]


def prediction_rows(
    designs: list[TileDesign],
    x: np.ndarray,
    evaluated: list[int],
    summaries: dict[int, tuple[float, float, float]],
) -> list[dict[str, float | int | str]]:
    remaining, mean_matrix, uncertainty_matrix = predict_remaining(
        x, evaluated, summaries
    )
    predicted_front = set(_nondominated_prediction_positions(mean_matrix))
    return [
        {
            "design": designs[index].name,
            "predicted_minimum_energy_mJ": float(np.exp(mean_matrix[position, 0])),
            "predicted_minimum_latency_ms": float(np.exp(mean_matrix[position, 1])),
            "predicted_minimum_EDP_mJ_ms": float(np.exp(mean_matrix[position, 2])),
            "energy_log_uncertainty": float(uncertainty_matrix[position, 0]),
            "latency_log_uncertainty": float(uncertainty_matrix[position, 1]),
            "EDP_log_uncertainty": float(uncertainty_matrix[position, 2]),
            "predicted_energy_latency_pareto": int(position in predicted_front),
            "evaluated_design_count": len(evaluated),
        }
        for position, index in enumerate(remaining)
    ]


def select_validation_batch(
    x: np.ndarray,
    evaluated: list[int],
    summaries: dict[int, tuple[float, float, float]],
    batch_size: int,
) -> list[int]:
    remaining, mean_matrix, uncertainty_matrix = predict_remaining(
        x, evaluated, summaries
    )
    if not remaining:
        return []

    candidate_positions = _nondominated_prediction_positions(mean_matrix)
    if len(candidate_positions) < batch_size:
        normalized = (mean_matrix[:, :2] - mean_matrix[:, :2].min(axis=0)) / np.maximum(
            np.ptp(mean_matrix[:, :2], axis=0), 1e-9
        )
        near_front = np.argsort(normalized.sum(axis=1))
        candidate_positions.extend(
            int(position)
            for position in near_front
            if int(position) not in candidate_positions
        )

    selected: list[int] = []
    available = candidate_positions[:]
    while available and len(selected) < batch_size:
        position = max(
            available,
            key=lambda item: (
                float(uncertainty_matrix[item].mean())
                + (
                    min(
                        float(np.linalg.norm(x[remaining[item]] - x[chosen]))
                        for chosen in selected
                    )
                    if selected
                    else 0.0
                )
            ),
        )
        selected.append(remaining[position])
        available.remove(position)
    return selected


def evaluate_design(
    design: TileDesign,
    layers,
    library,
    tech_node: float,
    reference: TileDesign,
    latency_cap_ratio: float,
    local_pareto_epsilon: float,
    local_state_cap: int,
    partial_path: Path | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    layer_rows = read_partial_layer_rows(partial_path)
    layers_by_shape = {}
    for layer in layers:
        key = (
            layer.reduction_length,
            layer.output_channels,
            layer.output_positions,
        )
        layers_by_shape.setdefault(key, []).append(layer)
    completed_layers = {str(row["layer"]) for row in layer_rows}
    total_shapes = len(layers_by_shape)
    for shape_index, shape_layers in enumerate(layers_by_shape.values(), start=1):
        layer_names = {layer.name for layer in shape_layers}
        if layer_names <= completed_layers:
            continue
        representative = shape_layers[0]
        schedules = evaluate_tile_schedules(
            representative,
            design,
            library,
            tech_node,
            reference,
            latency_cap_ratio,
            local_pareto_epsilon,
            local_state_cap=local_state_cap,
        )
        layer_rows = [
            row for row in layer_rows if str(row["layer"]) not in layer_names
        ]
        layer_rows.extend(
            {**row, "layer": layer.name}
            for layer in shape_layers
            for row in schedules
        )
        completed_layers.update(layer_names)
        write_partial_layer_rows(partial_path, layer_rows)
        print(
            f"Partial design {design.name}: shape {shape_index}/{total_shapes} "
            f"K={representative.reduction_length}, "
            f"N={representative.output_channels}, "
            f"P={representative.output_positions}",
            flush=True,
        )
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


def aggregate_energy_latency_pareto(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    ordered = sorted(
        rows,
        key=lambda row: (float(row["fabric_energy_mJ"]), float(row["latency_ms"])),
    )
    output = []
    best_latency = float("inf")
    for row in ordered:
        latency = float(row["latency_ms"])
        if latency < best_latency:
            output.append(row)
            best_latency = latency
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def checkpoint(
    results_dir: Path,
    designs: list[TileDesign],
    evaluated: list[int],
    summaries: dict[int, tuple[float, float, float]],
    layer_rows: list[dict[str, float | int | str]],
    aggregate_rows: list[dict[str, float | int | str]],
    latency_cap_ratio: float,
    local_pareto_epsilon: float,
    local_state_cap: int,
    tile_peak_compute_tops: float,
    peak_compute_tolerance_ratio: float,
    peak_compute_atol_tops: float,
    x: np.ndarray,
    selection_stages: dict[int, str],
    workload_name: str,
    workload_batch_size: int,
    candidate_pool_size: int,
    design_panel_sha256: str,
) -> None:
    write_csv(results_dir / f"{RESULT_PREFIX}_layers.csv", layer_rows)
    write_csv(results_dir / f"{RESULT_PREFIX}_aggregate.csv", aggregate_rows)
    metadata = [
        {
            "design": designs[index].name,
            "selection_order": order,
            "selection_stage": selection_stages[index],
            "minimum_energy_mJ": summaries[index][0],
            "minimum_latency_ms": summaries[index][1],
            "minimum_EDP_mJ_ms": summaries[index][2],
            "latency_cap_ratio": latency_cap_ratio,
            "local_pareto_epsilon": local_pareto_epsilon,
            "local_state_cap": local_state_cap,
            "tile_peak_compute_reference_tops": tile_peak_compute_tops,
            "peak_compute_tolerance_ratio": peak_compute_tolerance_ratio,
            "peak_compute_atol_tops": peak_compute_atol_tops,
            "energy_model_version": ENERGY_MODEL_VERSION,
            "architecture_model_version": ARCHITECTURE_MODEL_VERSION,
            "latency_model_version": LATENCY_MODEL_VERSION,
            "schedule_search_version": SCHEDULE_SEARCH_VERSION,
            "workload_name": workload_name,
            "workload_batch_size": workload_batch_size,
            "candidate_pool_size": candidate_pool_size,
            "evaluation_pool_size": len(designs),
            "design_panel_sha256": design_panel_sha256,
        }
        for order, index in enumerate(evaluated, start=1)
    ]
    write_csv(results_dir / f"{RESULT_PREFIX}_bayesian_samples.csv", metadata)
    predictions = (
        []
        if design_panel_sha256
        else prediction_rows(designs, x, evaluated, summaries)
    )
    if predictions:
        write_csv(
            results_dir / f"{RESULT_PREFIX}_bayesian_predictions.csv",
            predictions,
        )


def main() -> None:
    args = parse_args()
    if args.design_budget < 1 or args.initial_designs < 1:
        raise ValueError("design-budget and initial-designs must be at least 1")
    if args.initial_designs > args.design_budget:
        raise ValueError("initial-designs cannot exceed design-budget")
    if args.validation_designs < 0:
        raise ValueError("validation-designs cannot be negative")
    if args.latency_cap_ratio < 1:
        raise ValueError("latency-cap-ratio must be at least 1.")
    if not 0 <= args.local_pareto_epsilon < 1:
        raise ValueError("local-pareto-epsilon must be in [0, 1).")
    if args.local_state_cap < 2:
        raise ValueError("local-state-cap must be at least 2.")
    if args.workers < 1:
        raise ValueError("workers must be at least 1.")
    if args.workload_batch_size < 1:
        raise ValueError("workload-batch-size must be at least 1.")
    if args.tile_peak_compute_tops <= 0:
        raise ValueError("tile-peak-compute-tops must be positive.")
    if not 0 <= args.peak_compute_tolerance_ratio < 1:
        raise ValueError("peak-compute-tolerance-ratio must be in [0, 1).")
    if args.peak_compute_atol_tops < 0:
        raise ValueError("peak-compute-atol-tops cannot be negative.")
    results_dir = result_directory(args.output_subdir)
    spec = load_sram_spec("system_digital_sram_4x4")
    selected_workload_path = workload_path(args.workload)
    jinja_data = {"BATCH_SIZE": args.workload_batch_size}
    spec.workload = af.Workload.from_yaml(
        selected_workload_path, top_key="workload", jinja_parse_data=jinja_data
    )
    spec.renames = af.Renames.from_yaml(
        selected_workload_path, top_key="renames", jinja_parse_data=jinja_data
    )
    library = load_sram_macro_library("system_digital_sram_4x4")
    reference, candidate_designs = enumerate_tile_designs(
        library,
        args.tile_peak_compute_tops,
        args.peak_compute_tolerance_ratio,
        args.peak_compute_atol_tops,
    )
    candidate_pool_size = len(candidate_designs)
    fixed_names = fixed_design_names(args.fixed_designs_csv) if args.fixed_designs_csv else []
    design_panel_sha256 = panel_sha256(fixed_names)
    if fixed_names:
        design_by_name = {design.name: design for design in candidate_designs}
        missing = [name for name in fixed_names if name not in design_by_name]
        if missing:
            raise ValueError(f"Fixed-design CSV contains unknown designs: {missing}")
        designs = [design_by_name[name] for name in fixed_names]
    else:
        designs = candidate_designs
    budget = min(args.design_budget, len(designs))
    if fixed_names:
        budget = len(designs)
    initial = min(args.initial_designs, budget)
    validation = min(args.validation_designs, max(budget - initial, 0))
    acquisition_limit = budget - validation
    x = features(designs, reference)
    selected = list(range(len(designs))) if fixed_names else farthest_seeds(designs, x, initial)
    evaluated = []
    summaries = {}
    selection_stages: dict[int, str] = {}
    all_layer_rows = []
    all_aggregate_rows = []
    layers = layer_shapes_from_workload(spec.workload)
    tech_node = library.tech_node
    results_dir.mkdir(parents=True, exist_ok=True)
    partial_key = partial_checkpoint_key(
        args.workload,
        args.workload_batch_size,
        args.latency_cap_ratio,
        args.local_pareto_epsilon,
        args.local_state_cap,
    )
    partial_dir = results_dir / ".partial" / partial_key
    if args.resume:
        metadata = read_csv(results_dir / f"{RESULT_PREFIX}_bayesian_samples.csv")
        compatible = metadata and all(
            float(row.get("latency_cap_ratio", "nan")) == args.latency_cap_ratio
            and float(row.get("local_pareto_epsilon", "nan"))
            == args.local_pareto_epsilon
            and int(row.get("local_state_cap", 0)) == args.local_state_cap
            and float(row.get("tile_peak_compute_reference_tops", "nan"))
            == args.tile_peak_compute_tops
            and float(row.get("peak_compute_tolerance_ratio", "nan"))
            == args.peak_compute_tolerance_ratio
            and float(row.get("peak_compute_atol_tops", "nan"))
            == args.peak_compute_atol_tops
            and row.get("energy_model_version") == ENERGY_MODEL_VERSION
            and row.get("architecture_model_version") == ARCHITECTURE_MODEL_VERSION
            and row.get("latency_model_version") == LATENCY_MODEL_VERSION
            and row.get("schedule_search_version") == SCHEDULE_SEARCH_VERSION
            and row.get("workload_name", "resnet18") == args.workload
            and int(row.get("workload_batch_size", 1)) == args.workload_batch_size
            and row.get("design_panel_sha256", "") == design_panel_sha256
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
        selection_stages = {
            design_index[row["design"]]: row.get("selection_stage", "acquisition")
            for row in metadata
            if row["design"] in design_index
        }
        evaluated_names = {designs[index].name for index in evaluated}
        all_layer_rows = [
            row
            for row in read_csv(results_dir / f"{RESULT_PREFIX}_layers.csv")
            if row["design"] in evaluated_names
        ]
        all_aggregate_rows = [
            row
            for row in read_csv(results_dir / f"{RESULT_PREFIX}_aggregate.csv")
            if row["design"] in evaluated_names
        ]
        for row in all_aggregate_rows:
            row.setdefault("aggregation_epsilon", "0.0")
            row.setdefault("aggregation_state_cap", "0")
        all_aggregate_rows = compact_aggregate_rows(all_aggregate_rows)
        print(f"Resuming from {len(evaluated)} evaluated designs", flush=True)

    while len(evaluated) < budget:
        batch = [index for index in selected if index not in evaluated]
        selection_stage = "fixed_panel" if fixed_names else "initial"
        if not batch:
            if len(evaluated) < acquisition_limit:
                selection_stage = "acquisition"
                batch = select_batch(
                    x,
                    evaluated,
                    summaries,
                    min(
                        args.batch_size,
                        acquisition_limit - len(evaluated),
                    ),
                )
            else:
                selection_stage = "predicted_front_validation"
                batch = select_validation_batch(
                    x,
                    evaluated,
                    summaries,
                    min(args.batch_size, budget - len(evaluated)),
                )
        pending = (
            batch
            if fixed_names
            else batch[: min(args.batch_size, budget - len(evaluated))]
        )
        for index in pending:
            selection_stages[index] = selection_stage
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
                        args.local_state_cap,
                        partial_dir / f"{index:03d}_{design.name}.pkl",
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
                for row in layer_rows:
                    row["workload_name"] = args.workload
                    row["workload_batch_size"] = args.workload_batch_size
                for row in aggregate_rows:
                    row["workload_name"] = args.workload
                    row["workload_batch_size"] = args.workload_batch_size
                summaries[index] = summarize(aggregate_rows)
                evaluated.append(index)
                checkpoint(
                    results_dir,
                    designs,
                    evaluated,
                    summaries,
                    all_layer_rows,
                    all_aggregate_rows,
                    args.latency_cap_ratio,
                    args.local_pareto_epsilon,
                    args.local_state_cap,
                    args.tile_peak_compute_tops,
                    args.peak_compute_tolerance_ratio,
                    args.peak_compute_atol_tops,
                    x,
                    selection_stages,
                    args.workload,
                    args.workload_batch_size,
                    candidate_pool_size,
                    design_panel_sha256,
                )
        selected = [index for index in selected if index not in evaluated]

    all_aggregate_rows = compact_aggregate_rows(all_aggregate_rows)
    aggregate_pareto = aggregate_energy_latency_pareto(all_aggregate_rows)
    aggregate_pareto_eta = aggregate_energy_latency_area_pareto(
        all_aggregate_rows
    )
    write_csv(results_dir / f"{RESULT_PREFIX}_layers.csv", all_layer_rows)
    write_csv(results_dir / f"{RESULT_PREFIX}_aggregate.csv", all_aggregate_rows)
    write_csv(results_dir / f"{RESULT_PREFIX}_aggregate_pareto.csv", aggregate_pareto)
    write_csv(
        results_dir / f"{RESULT_PREFIX}_aggregate_pareto_eta.csv",
        aggregate_pareto_eta,
    )
    checkpoint(
        results_dir,
        designs,
        evaluated,
        summaries,
        all_layer_rows,
        all_aggregate_rows,
        args.latency_cap_ratio,
        args.local_pareto_epsilon,
        args.local_state_cap,
        args.tile_peak_compute_tops,
        args.peak_compute_tolerance_ratio,
        args.peak_compute_atol_tops,
        x,
        selection_stages,
        args.workload,
        args.workload_batch_size,
        candidate_pool_size,
        design_panel_sha256,
    )
    lower_tops = args.tile_peak_compute_tops * (1 - args.peak_compute_tolerance_ratio)
    upper_tops = args.tile_peak_compute_tops * (1 + args.peak_compute_tolerance_ratio)
    print(f"Tile peak-compute reference: {args.tile_peak_compute_tops:.6f} TOPS")
    print(f"Tile peak-compute window: [{lower_tops:.6f}, {upper_tops:.6f}] TOPS")
    print(f"Workload: {args.workload}, batch size {args.workload_batch_size}")
    print(f"Workload layers: {len(layers)}")
    print(f"Workload MACs: {sum(layer.macs for layer in layers)}")
    print(f"Evaluated designs: {len(evaluated)}/{len(designs)}")
    print(f"Aggregate Pareto points: {len(aggregate_pareto)}")
    print(f"Aggregate E-T-A Pareto points: {len(aggregate_pareto_eta)}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a 48-point Bayesian Tile DSE for a pipelined ViT workload."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from math import inf
from pathlib import Path
import signal
import sys
from time import perf_counter


RUN_DIR = Path(__file__).resolve().parents[2]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from sram_macro_types import load_sram_macro_library
from tile_DSE.run_sram_tile_bayesian_dse import (
    aggregate_energy_latency_area_pareto,
    aggregate_energy_latency_pareto,
    farthest_seeds,
    features,
    fixed_design_names,
    panel_sha256,
    prediction_rows,
    select_batch,
    select_validation_batch,
    summarize,
)
from tile_DSE.sram_tile_dse import enumerate_tile_designs
from tile_DSE.vit.vision_transformer_workload import (
    vit_b16_macs,
    vit_b16_operator_groups,
    vit_l16_macs,
    vit_l16_operator_groups,
)
from tile_DSE.vit.vit_tile_dse import (
    ENERGY_MODEL_VERSION,
    PIPELINE_MODEL_VERSION,
    evaluate_vit_design,
    workload_version,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RESULTS_DIR = (
    ROOT / "experiments" / "results" / "tile_DSE" / "vision_transformer_v5"
)
DEFAULT_VIT_L_RESULTS_DIR = (
    ROOT / "experiments" / "results" / "tile_DSE" / "vision_transformer_l16_v2"
)
DEFAULT_FIXED_PANEL = (
    ROOT
    / "experiments"
    / "docs"
    / "tile_DSE"
    / "frozen_panels"
    / "resnet18_tile_panel_48.csv"
)
RESULT_PREFIX = "vit_b16_pipeline_tile_dse"


class DesignEvaluationTimeout(TimeoutError):
    """Raised when one physical Tile exceeds its wall-clock evaluation budget."""


@contextmanager
def _design_timeout(timeout_s: float):
    if timeout_s <= 0:
        yield
        return

    def handle_timeout(_signum, _frame):
        raise DesignEvaluationTimeout(
            f"Design evaluation exceeded {timeout_s:g} seconds"
        )

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-model", choices=("vit_b16", "vit_l16"), default="vit_b16")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--result-prefix")
    parser.add_argument("--fixed-designs-csv", type=Path, default=DEFAULT_FIXED_PANEL)
    parser.add_argument(
        "--select-new-panel",
        action="store_true",
        help="Run an independent Bayesian selection instead of the frozen cross-network panel.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--tile-peak-compute-tops", type=float, default=0.4608)
    parser.add_argument("--peak-compute-tolerance-ratio", type=float, default=0.20)
    parser.add_argument("--peak-compute-atol-tops", type=float, default=1e-12)
    parser.add_argument("--design-budget", type=int, default=48)
    parser.add_argument("--initial-designs", type=int, default=16)
    parser.add_argument("--acquisition-batch-size", type=int, default=4)
    parser.add_argument("--validation-designs", type=int, default=8)
    parser.add_argument("--latency-cap-ratio", type=float, default=4.0)
    parser.add_argument("--local-pareto-epsilon", type=float, default=0.001)
    parser.add_argument("--local-state-cap", type=int, default=4096)
    parser.add_argument("--aggregate-state-cap", type=int, default=256)
    parser.add_argument(
        "--design-timeout-s",
        type=float,
        default=600.0,
        help="Skip one Tile after this many wall-clock seconds; 0 disables the timeout.",
    )
    parser.add_argument(
        "--partition-budget",
        type=int,
        default=0,
        help="0 exhausts all bank partitions; a positive value enables an explicit heuristic cap.",
    )
    parser.add_argument(
        "--evaluate-dense-small-macro-designs",
        action="store_true",
        help="Evaluate ViT designs that match the empirical small-Macro complexity skip rule.",
    )
    parser.add_argument(
        "--completed-design-target",
        type=int,
        default=0,
        help="stop after this many completed designs; 0 evaluates the full panel",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint(
    results_dir: Path,
    designs,
    evaluated: list[int],
    summaries: dict[int, tuple[float, float, float]],
    selection_stages: dict[int, str],
    evaluation_statuses: dict[int, str],
    evaluation_elapsed_s: dict[int, float],
    evaluation_errors: dict[int, str],
    layer_rows: list[dict],
    aggregate_rows: list[dict],
    x,
    args: argparse.Namespace,
    design_panel_sha256: str,
) -> None:
    sample_rows = [
        {
            "selection_order": order,
            "selection_stage": selection_stages[index],
            "design": designs[index].name,
            "minimum_energy_mJ": summaries[index][0],
            "minimum_latency_ms": summaries[index][1],
            "minimum_EDP_mJ_ms": summaries[index][2],
            "evaluation_status": evaluation_statuses[index],
            "evaluation_elapsed_s": evaluation_elapsed_s[index],
            "evaluation_error": evaluation_errors[index],
            "batch_size": args.batch_size,
            "partition_budget": args.partition_budget,
            "design_timeout_s": args.design_timeout_s,
            "latency_cap_ratio": args.latency_cap_ratio,
            "local_pareto_epsilon": args.local_pareto_epsilon,
            "local_state_cap": args.local_state_cap,
            "aggregate_state_cap": args.aggregate_state_cap,
            "dense_small_macro_policy": (
                "evaluate" if args.evaluate_dense_small_macro_designs else "skip"
            ),
            "completed_design_target": args.completed_design_target,
            "energy_model_version": ENERGY_MODEL_VERSION,
            "pipeline_model_version": PIPELINE_MODEL_VERSION,
            "workload_model_version": workload_version(args.workload_model),
            "design_panel_sha256": design_panel_sha256,
        }
        for order, index in enumerate(evaluated, start=1)
    ]
    _write_csv(results_dir / f"{RESULT_PREFIX}_bayesian_samples.csv", sample_rows)
    timeout_rows = [
        row for row in sample_rows if row["evaluation_status"] == "timeout_skipped"
    ]
    _write_csv(results_dir / f"{RESULT_PREFIX}_timeouts.csv", timeout_rows)
    skipped_rows = [
        row for row in sample_rows if row["evaluation_status"] == "complexity_skipped"
    ]
    _write_csv(results_dir / f"{RESULT_PREFIX}_complexity_skips.csv", skipped_rows)
    _write_csv(results_dir / f"{RESULT_PREFIX}_layers.csv", layer_rows)
    _write_csv(results_dir / f"{RESULT_PREFIX}_aggregate.csv", aggregate_rows)
    prediction_path = results_dir / f"{RESULT_PREFIX}_bayesian_predictions.csv"
    successful = [
        index for index in evaluated if evaluation_statuses[index] == "completed"
    ]
    if design_panel_sha256:
        prediction_path.unlink(missing_ok=True)
    elif successful:
        successful_summaries = {index: summaries[index] for index in successful}
        predictions = prediction_rows(
            designs, x, successful, successful_summaries
        )
        _write_csv(prediction_path, predictions)


def _validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.design_budget < 1 or args.initial_designs < 1:
        raise ValueError("design-budget and initial-designs must be positive")
    if args.initial_designs > args.design_budget:
        raise ValueError("initial-designs cannot exceed design-budget")
    if args.validation_designs < 0:
        raise ValueError("validation-designs cannot be negative")
    if args.acquisition_batch_size < 1:
        raise ValueError("acquisition-batch-size must be positive")
    if args.latency_cap_ratio < 1:
        raise ValueError("latency-cap-ratio must be at least 1")
    if not 0 <= args.local_pareto_epsilon < 1:
        raise ValueError("local-pareto-epsilon must be in [0, 1)")
    if args.local_state_cap < 2 or args.aggregate_state_cap < 2:
        raise ValueError("state caps must be at least 2")
    if args.partition_budget < 0:
        raise ValueError("partition-budget cannot be negative")
    if args.design_timeout_s < 0:
        raise ValueError("design-timeout-s cannot be negative")
    if args.completed_design_target < 0:
        raise ValueError("completed-design-target cannot be negative")


def dense_small_macro_skip_reason(design) -> str | None:
    """Return the empirical ViT complexity rule matched by a Tile design."""

    counts = dict(design.counts)
    m32_count = counts.get("m32x8", 0)
    m256_count = counts.get("m256x8", 0)
    m512_count = counts.get("m512x8", 0)
    if m32_count >= 9:
        return "m32_count>=9"
    if (
        len(design.counts) > 1
        and design.macro_count >= 12
        and m512_count == 0
        and m256_count <= 1
    ):
        return "heterogeneous_macro_count>=12_without_m512_or_multiple_m256"
    return None


def main() -> None:
    global RESULT_PREFIX
    args = parse_args()
    _validate_args(args)
    RESULT_PREFIX = args.result_prefix or (
        "vit_l16_pipeline_tile_dse"
        if args.workload_model == "vit_l16"
        else "vit_b16_pipeline_tile_dse"
    )
    library = load_sram_macro_library("system_digital_sram_4x4")
    reference, candidates = enumerate_tile_designs(
        library,
        args.tile_peak_compute_tops,
        args.peak_compute_tolerance_ratio,
        args.peak_compute_atol_tops,
    )
    fixed_panel_path = None if args.select_new_panel else args.fixed_designs_csv
    if fixed_panel_path:
        design_by_name = {design.name: design for design in candidates}
        fixed_names = fixed_design_names(fixed_panel_path)
        missing = [name for name in fixed_names if name not in design_by_name]
        if missing:
            raise ValueError(f"Unknown fixed designs: {missing}")
        designs = [design_by_name[name] for name in fixed_names]
    else:
        designs = candidates
        fixed_names = []
    design_panel_hash = panel_sha256(fixed_names)
    budget = len(designs) if fixed_panel_path else min(args.design_budget, len(designs))
    validation = min(args.validation_designs, max(budget - args.initial_designs, 0))
    acquisition_limit = budget - validation
    x = features(designs, reference)
    default_results_dir = (
        DEFAULT_VIT_L_RESULTS_DIR
        if args.workload_model == "vit_l16"
        else DEFAULT_RESULTS_DIR
    )
    results_dir = (args.output_dir or default_results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluated: list[int] = []
    summaries: dict[int, tuple[float, float, float]] = {}
    selection_stages: dict[int, str] = {}
    evaluation_statuses: dict[int, str] = {}
    evaluation_elapsed_s: dict[int, float] = {}
    evaluation_errors: dict[int, str] = {}
    all_layer_rows: list[dict] = []
    all_aggregate_rows: list[dict] = []
    if args.resume:
        metadata = _read_csv(results_dir / f"{RESULT_PREFIX}_bayesian_samples.csv")
        compatible = metadata and all(
            row.get("energy_model_version") == ENERGY_MODEL_VERSION
            and row.get("pipeline_model_version") == PIPELINE_MODEL_VERSION
            and row.get("workload_model_version") == workload_version(args.workload_model)
            and int(row.get("batch_size", 0)) == args.batch_size
            and int(row.get("partition_budget", -1)) == args.partition_budget
            and float(row.get("latency_cap_ratio", "nan"))
            == args.latency_cap_ratio
            and float(row.get("local_pareto_epsilon", "nan"))
            == args.local_pareto_epsilon
            and int(row.get("local_state_cap", 0)) == args.local_state_cap
            and int(row.get("aggregate_state_cap", 0))
            == args.aggregate_state_cap
            and row.get("dense_small_macro_policy", "evaluate")
            == ("evaluate" if args.evaluate_dense_small_macro_designs else "skip")
            and int(row.get("completed_design_target", 0))
            == args.completed_design_target
            and float(row.get("design_timeout_s", args.design_timeout_s))
            == args.design_timeout_s
            and row.get("design_panel_sha256", "") == design_panel_hash
            for row in metadata
        )
        if compatible:
            index_by_name = {design.name: index for index, design in enumerate(designs)}
            metadata.sort(key=lambda row: int(row["selection_order"]))
            evaluated = [index_by_name[row["design"]] for row in metadata]
            summaries = {
                index_by_name[row["design"]]: (
                    float(row["minimum_energy_mJ"]),
                    float(row["minimum_latency_ms"]),
                    float(row["minimum_EDP_mJ_ms"]),
                )
                for row in metadata
            }
            selection_stages = {
                index_by_name[row["design"]]: row["selection_stage"]
                for row in metadata
            }
            evaluation_statuses = {
                index_by_name[row["design"]]: row.get(
                    "evaluation_status", "completed"
                )
                for row in metadata
            }
            evaluation_elapsed_s = {
                index_by_name[row["design"]]: float(
                    row.get("evaluation_elapsed_s", 0.0)
                )
                for row in metadata
            }
            evaluation_errors = {
                index_by_name[row["design"]]: row.get("evaluation_error", "")
                for row in metadata
            }
            all_layer_rows = _read_csv(results_dir / f"{RESULT_PREFIX}_layers.csv")
            all_aggregate_rows = _read_csv(results_dir / f"{RESULT_PREFIX}_aggregate.csv")
            print(f"Resuming from {len(evaluated)} evaluated designs", flush=True)

    seeds = (
        list(range(len(designs)))
        if fixed_panel_path
        else farthest_seeds(designs, x, min(args.initial_designs, budget))
    )
    completed_design_target = args.completed_design_target or budget
    while len(evaluated) < budget:
        if (
            sum(status == "completed" for status in evaluation_statuses.values())
            >= completed_design_target
        ):
            break
        pending = [index for index in seeds if index not in evaluated]
        stage = "fixed_panel" if fixed_panel_path else "initial"
        if not pending:
            if len(evaluated) < acquisition_limit:
                stage = "acquisition"
                pending = select_batch(
                    x,
                    evaluated,
                    summaries,
                    min(args.acquisition_batch_size, acquisition_limit - len(evaluated)),
                )
            else:
                stage = "predicted_front_validation"
                pending = select_validation_batch(
                    x,
                    evaluated,
                    summaries,
                    min(args.acquisition_batch_size, budget - len(evaluated)),
                )
        if not fixed_panel_path:
            pending = pending[: min(args.acquisition_batch_size, budget - len(evaluated))]
        for index in pending:
            if (
                sum(status == "completed" for status in evaluation_statuses.values())
                >= completed_design_target
            ):
                break
            design = designs[index]
            selection_stages[index] = stage
            print(f"Evaluating {len(evaluated) + 1}/{budget}: {design.name}", flush=True)
            started_at = perf_counter()
            skip_reason = (
                None
                if args.evaluate_dense_small_macro_designs
                else dense_small_macro_skip_reason(design)
            )
            if skip_reason:
                evaluation_statuses[index] = "complexity_skipped"
                evaluation_errors[index] = skip_reason
                evaluation_elapsed_s[index] = 0.0
                summaries[index] = (inf, inf, inf)
                print(
                    f"Complexity skipped {design.name}: {skip_reason}",
                    flush=True,
                )
            else:
                try:
                    with _design_timeout(args.design_timeout_s):
                        layer_rows, aggregate_rows = evaluate_vit_design(
                            design,
                            library,
                            reference,
                            batch_size=args.batch_size,
                            latency_cap_ratio=args.latency_cap_ratio,
                            local_pareto_epsilon=args.local_pareto_epsilon,
                            local_state_cap=args.local_state_cap,
                            aggregate_state_cap=args.aggregate_state_cap,
                            partition_budget=args.partition_budget,
                            workload_model=args.workload_model,
                            partial_path=results_dir / ".partial" / f"{design.name}.pkl",
                        )
                except DesignEvaluationTimeout as error:
                    evaluation_statuses[index] = "timeout_skipped"
                    evaluation_errors[index] = str(error)
                    evaluation_elapsed_s[index] = perf_counter() - started_at
                    summaries[index] = (inf, inf, inf)
                    print(
                        f"Timeout skipped {design.name} after "
                        f"{evaluation_elapsed_s[index]:.1f}s",
                        flush=True,
                    )
                else:
                    all_layer_rows.extend(layer_rows)
                    all_aggregate_rows.extend(aggregate_rows)
                    summaries[index] = summarize(aggregate_rows)
                    evaluation_statuses[index] = "completed"
                    evaluation_errors[index] = ""
                    evaluation_elapsed_s[index] = perf_counter() - started_at
            evaluated.append(index)
            _checkpoint(
                results_dir,
                designs,
                evaluated,
                summaries,
                selection_stages,
                evaluation_statuses,
                evaluation_elapsed_s,
                evaluation_errors,
                all_layer_rows,
                all_aggregate_rows,
                x,
                args,
                design_panel_hash,
            )

    energy_latency = aggregate_energy_latency_pareto(all_aggregate_rows)
    energy_latency_area = aggregate_energy_latency_area_pareto(all_aggregate_rows)
    _write_csv(results_dir / f"{RESULT_PREFIX}_aggregate_pareto.csv", energy_latency)
    _write_csv(
        results_dir / f"{RESULT_PREFIX}_aggregate_pareto_eta.csv",
        energy_latency_area,
    )
    if args.workload_model == "vit_l16":
        workload_label = "ViT-L/16"
        groups = vit_l16_operator_groups(args.batch_size)
        macs = vit_l16_macs(args.batch_size)
    else:
        workload_label = "ViT-B/16"
        groups = vit_b16_operator_groups(args.batch_size)
        macs = vit_b16_macs(args.batch_size)
    print(f"{workload_label} operator groups: {len(groups)}")
    print(f"{workload_label} MACs: {macs}")
    completed_count = sum(
        status == "completed" for status in evaluation_statuses.values()
    )
    timeout_count = sum(
        status == "timeout_skipped" for status in evaluation_statuses.values()
    )
    complexity_skip_count = sum(
        status == "complexity_skipped" for status in evaluation_statuses.values()
    )
    print(f"Resolved designs: {len(evaluated)}/{len(designs)}")
    print(f"Completed designs: {completed_count}")
    print(f"Timeout-skipped designs: {timeout_count}")
    print(f"Complexity-skipped designs: {complexity_skip_count}")
    print(f"Results: {results_dir}")


if __name__ == "__main__":
    main()

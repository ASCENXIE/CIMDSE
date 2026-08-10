"""Projection/attention pipeline mapping for ViT Tile DSE.

The existing CNN evaluator remains the source of single-bank schedules. This
module owns all ViT-specific resource partitioning and network aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import pickle

from sram_macro_types import ENERGY_MODEL_VERSION, MacroLibrary
from tile_DSE.sram_tile_dse import (
    TileDesign,
    _macro_input_register_area_mm2,
    _tile_output_register_area_mm2,
    evaluate_tile_schedules,
)
from tile_DSE.vit.vision_transformer_workload import (
    VitOperatorGroup,
    vit_b16_operator_groups,
)


PIPELINE_MODEL_VERSION = "vit_pipeline_nblock_wave_timing_v4"
WORKLOAD_MODEL_VERSION = "vit_b16_192_tokens_corrected_attention_axis_v2"

_ENERGY_KEYS = (
    "macro_dynamic_energy_j",
    "input_register_energy_j",
    "output_register_energy_j",
    "register_energy_j",
    "psum_energy_j",
    "cross_macro_psum_energy_j",
    "cross_wave_psum_energy_j",
    "weight_load_energy_j",
    "bus_energy_j",
)


@dataclass(frozen=True)
class BankAssignment:
    projection: TileDesign
    attention: TileDesign
    projection_lanes: int
    attention_lanes: int
    concurrent: bool = True

    @property
    def name(self) -> str:
        if not self.concurrent:
            return f"sequential:{self.projection.name}"
        return (
            f"P[{self.projection.name}]@L{self.projection_lanes};"
            f"A[{self.attention.name}]@L{self.attention_lanes}"
        )


def _subdesign(
    counts: tuple[tuple[str, int], ...],
    lane_count: int,
    library: MacroLibrary,
) -> TileDesign:
    active = tuple((name, count) for name, count in counts if count)
    macros = tuple(
        library.macro_types[name]
        for name, count in active
        for _ in range(count)
    )
    if not macros:
        raise ValueError("A pipeline bank cannot be empty")
    input_register_area = sum(
        _macro_input_register_area_mm2(macro, library.tech_node)
        for macro in macros
    )
    output_register_area = _tile_output_register_area_mm2(
        library.tech_node,
        library.output_register_entry_bits,
        lane_count,
        macros[0].columns,
    )
    return TileDesign(
        counts=active,
        capacity_values=sum(macro.rows * macro.columns for macro in macros),
        macro_area_mm2=sum(macro.area_mm2 for macro in macros),
        register_area_mm2=input_register_area + output_register_area,
        peak_compute_tops=sum(macro.peak_tops for macro in macros),
    )


def _partition_score(
    projection: TileDesign,
    attention: TileDesign,
    projection_macs: int,
    attention_macs: int,
) -> float:
    return max(
        projection_macs / projection.peak_compute_tops,
        attention_macs / attention.peak_compute_tops,
    )


def enumerate_bank_assignments(
    design: TileDesign,
    library: MacroLibrary,
    groups: tuple[VitOperatorGroup, ...] | None = None,
    partition_budget: int = 0,
) -> list[BankAssignment]:
    """Enumerate disjoint Macro banks and legal splits of the four lanes.

    ``partition_budget=0`` keeps every physical partition. A positive budget
    retains the analytically best partitions plus type-pure specialization
    points. The budget applies before the six legal lane allocations.
    """

    if partition_budget < 0:
        raise ValueError("partition_budget cannot be negative")
    groups = groups or vit_b16_operator_groups()
    if design.macro_count == 1:
        return [
            BankAssignment(
                design,
                design,
                library.tile_output_lanes,
                library.tile_output_lanes,
                concurrent=False,
            )
        ]

    projection_macs = sum(group.macs for group in groups if group.stream == "projection")
    attention_macs = sum(group.macs for group in groups if group.stream == "attention")
    count_ranges = [range(count + 1) for _, count in design.counts]
    partitions: list[tuple[TileDesign, TileDesign, bool]] = []
    for projection_counts in product(*count_ranges):
        if not any(projection_counts):
            continue
        attention_counts = tuple(
            total - projection
            for (_, total), projection in zip(design.counts, projection_counts)
        )
        if not any(attention_counts):
            continue
        projection = _subdesign(
            tuple(
                (name, count)
                for (name, _), count in zip(design.counts, projection_counts)
            ),
            1,
            library,
        )
        attention = _subdesign(
            tuple(
                (name, count)
                for (name, _), count in zip(design.counts, attention_counts)
            ),
            1,
            library,
        )
        type_pure = (
            len(design.counts) == 2
            and len(projection.counts) == len(attention.counts) == 1
            and projection.counts[0][0] != attention.counts[0][0]
        )
        partitions.append((projection, attention, type_pure))

    partitions.sort(
        key=lambda item: (
            _partition_score(item[0], item[1], projection_macs, attention_macs),
            item[0].name,
            item[1].name,
        )
    )
    if partition_budget and len(partitions) > partition_budget:
        retained = partitions[:partition_budget]
        retained_names = {(item[0].name, item[1].name) for item in retained}
        retained.extend(
            item
            for item in partitions
            if item[2] and (item[0].name, item[1].name) not in retained_names
        )
        partitions = retained

    assignments = []
    for projection, attention, _ in partitions:
        for projection_lanes in range(1, library.tile_output_lanes):
            for attention_lanes in range(1, library.tile_output_lanes):
                if projection_lanes + attention_lanes <= library.tile_output_lanes:
                    assignments.append(
                        BankAssignment(
                            _subdesign(projection.counts, projection_lanes, library),
                            _subdesign(attention.counts, attention_lanes, library),
                            projection_lanes,
                            attention_lanes,
                        )
                    )
    return assignments


def _dynamic_energy(row: dict[str, float | int | str]) -> float:
    return float(row["fabric_energy_j"]) - float(row["macro_static_energy_j"])


def _macro_critical_latency(row: dict[str, float | int | str]) -> float:
    return float(row["latency_s"]) - float(row["bus_added_latency_s"])


def _state_from_schedule(
    row: dict[str, float | int | str], repeat: int
) -> dict[str, float | int | str]:
    state: dict[str, float | int | str] = {
        "dynamic_energy_j": _dynamic_energy(row) * repeat,
        "macro_critical_latency_s": _macro_critical_latency(row) * repeat,
        "serial_latency_s": float(row["latency_s"]) * repeat,
        "total_bus_cycles": int(row["total_bus_cycles"]) * repeat,
        "MACs": int(row["MACs"]) * repeat,
        "peak_active_output_lanes": int(row["active_lane_count"]),
        "schedule_trace": f"{row['layer']}x{repeat}:{row['schedule_signature']}",
    }
    for key in _ENERGY_KEYS:
        state[key] = float(row.get(key, 0.0)) * repeat
    return state


def _combine(left: dict, right: dict) -> dict:
    output = {
        "dynamic_energy_j": float(left["dynamic_energy_j"])
        + float(right["dynamic_energy_j"]),
        "macro_critical_latency_s": float(left["macro_critical_latency_s"])
        + float(right["macro_critical_latency_s"]),
        "serial_latency_s": float(left["serial_latency_s"])
        + float(right["serial_latency_s"]),
        "total_bus_cycles": int(left["total_bus_cycles"])
        + int(right["total_bus_cycles"]),
        "MACs": int(left["MACs"]) + int(right["MACs"]),
        "peak_active_output_lanes": max(
            int(left["peak_active_output_lanes"]),
            int(right["peak_active_output_lanes"]),
        ),
        "schedule_trace": f"{left['schedule_trace']};{right['schedule_trace']}".strip(";"),
    }
    for key in _ENERGY_KEYS:
        output[key] = float(left.get(key, 0.0)) + float(right.get(key, 0.0))
    return output


def _front(
    rows: list[dict],
    objectives: tuple[str, ...],
    state_cap: int,
) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: tuple(float(row[key]) for key in objectives),
    )
    front = []
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
    if len(front) <= state_cap:
        return front
    indices = {
        round(index * (len(front) - 1) / (state_cap - 1))
        for index in range(state_cap)
    }
    return [row for index, row in enumerate(front) if index in indices]


def _aggregate_groups(
    groups: tuple[VitOperatorGroup, ...],
    schedules_by_group: dict[str, list[dict[str, float | int | str]]],
    state_cap: int,
) -> list[dict]:
    states = [
        {
            "dynamic_energy_j": 0.0,
            "macro_critical_latency_s": 0.0,
            "serial_latency_s": 0.0,
            "total_bus_cycles": 0,
            "MACs": 0,
            "peak_active_output_lanes": 0,
            "schedule_trace": "",
            **{key: 0.0 for key in _ENERGY_KEYS},
        }
    ]
    for group in groups:
        options = [
            _state_from_schedule(row, group.repeat)
            for row in schedules_by_group[group.name]
        ]
        states = _front(
            [_combine(state, option) for state in states for option in options],
            ("dynamic_energy_j", "macro_critical_latency_s", "total_bus_cycles"),
            state_cap,
        )
    return states


def _schedule_cache_key(
    design: TileDesign, lane_count: int, group: VitOperatorGroup
) -> tuple:
    shape = group.shape
    return (
        design.counts,
        lane_count,
        shape.reduction_length,
        shape.output_channels,
        shape.output_positions,
    )


def evaluate_vit_design(
    design: TileDesign,
    library: MacroLibrary,
    reference: TileDesign,
    batch_size: int = 1,
    latency_cap_ratio: float = 4.0,
    local_pareto_epsilon: float = 0.001,
    local_state_cap: int = 4096,
    aggregate_state_cap: int = 256,
    partition_budget: int = 0,
    partial_path: Path | None = None,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Evaluate one physical Tile under the ViT two-stream pipeline model."""

    groups = vit_b16_operator_groups(batch_size)
    projection_groups = tuple(group for group in groups if group.stream == "projection")
    attention_groups = tuple(group for group in groups if group.stream == "attention")
    endpoint_groups = tuple(group for group in groups if group.stream == "endpoint")
    assignments = enumerate_bank_assignments(
        design, library, groups, partition_budget=partition_budget
    )
    checkpoint_signature = (
        ENERGY_MODEL_VERSION,
        PIPELINE_MODEL_VERSION,
        WORKLOAD_MODEL_VERSION,
        design.name,
        batch_size,
        latency_cap_ratio,
        local_pareto_epsilon,
        local_state_cap,
        aggregate_state_cap,
        partition_budget,
    )
    cycle_period_s = next(iter(library.macro_types.values())).cycle_period_s
    full_static_power_w = sum(
        count * library.macro_types[name].static_power_w
        for name, count in design.counts
    )
    cache: dict[tuple, list[dict[str, float | int | str]]] = {}
    layer_rows: list[dict[str, float | int | str]] = []

    if partial_path and partial_path.is_file():
        with partial_path.open("rb") as file:
            payload = pickle.load(file)
        if payload.get("signature") == checkpoint_signature:
            return payload["layer_rows"], payload["aggregate_rows"]

    def schedules(group: VitOperatorGroup, bank: TileDesign, lanes: int, assignment: str):
        key = _schedule_cache_key(bank, lanes, group)
        if key not in cache:
            bank_library = replace(library, tile_output_lanes=lanes)
            cache[key] = evaluate_tile_schedules(
                group.shape,
                bank,
                bank_library,
                library.tech_node,
                reference,
                latency_cap_ratio,
                local_pareto_epsilon,
                local_state_cap,
            )
        enriched = []
        for row in cache[key]:
            item = {
                **row,
                "design": design.name,
                "operator_group": group.name,
                "operator_repeat": group.repeat,
                "pipeline_stream": group.stream,
                "bank_design": bank.name,
                "bank_lane_budget": lanes,
                "bank_assignment": assignment,
                "pipeline_model_version": PIPELINE_MODEL_VERSION,
                "workload_model_version": WORKLOAD_MODEL_VERSION,
            }
            enriched.append(item)
            layer_rows.append(item)
        return enriched

    endpoint_schedules = {
        group.name: schedules(group, design, library.tile_output_lanes, "full_tile_endpoint")
        for group in endpoint_groups
    }
    endpoint_states = _aggregate_groups(
        endpoint_groups, endpoint_schedules, aggregate_state_cap
    )
    aggregate_rows = []
    for assignment in assignments:
        projection_schedules = {
            group.name: schedules(
                group,
                assignment.projection,
                assignment.projection_lanes,
                assignment.name,
            )
            for group in projection_groups
        }
        attention_schedules = {
            group.name: schedules(
                group,
                assignment.attention,
                assignment.attention_lanes,
                assignment.name,
            )
            for group in attention_groups
        }
        projection_states = _aggregate_groups(
            projection_groups, projection_schedules, aggregate_state_cap
        )
        attention_states = _aggregate_groups(
            attention_groups, attention_schedules, aggregate_state_cap
        )
        pipeline_candidates = []
        for projection, attention in product(projection_states, attention_states):
            combined = _combine(projection, attention)
            shared_bus_latency_s = combined["total_bus_cycles"] * cycle_period_s
            if assignment.concurrent:
                transformer_latency_s = max(
                    float(projection["macro_critical_latency_s"]),
                    float(attention["macro_critical_latency_s"]),
                    shared_bus_latency_s,
                )
            else:
                transformer_latency_s = max(
                    float(projection["macro_critical_latency_s"])
                    + float(attention["macro_critical_latency_s"]),
                    shared_bus_latency_s,
                )
            combined["transformer_latency_s"] = transformer_latency_s
            combined["projection_compute_latency_s"] = projection[
                "macro_critical_latency_s"
            ]
            combined["attention_compute_latency_s"] = attention[
                "macro_critical_latency_s"
            ]
            combined["transformer_bus_latency_s"] = shared_bus_latency_s
            pipeline_candidates.append(combined)
        pipeline_states = _front(
            pipeline_candidates,
            ("dynamic_energy_j", "transformer_latency_s"),
            aggregate_state_cap,
        )
        for endpoint, pipeline in product(endpoint_states, pipeline_states):
            dynamic_energy_j = float(endpoint["dynamic_energy_j"]) + float(
                pipeline["dynamic_energy_j"]
            )
            endpoint_latency_s = float(endpoint["serial_latency_s"])
            latency_s = endpoint_latency_s + float(pipeline["transformer_latency_s"])
            static_energy_j = full_static_power_w * latency_s
            fabric_energy_j = dynamic_energy_j + static_energy_j
            aggregate_rows.append(
                {
                    "design": design.name,
                    "tile_class": "homogeneous" if len(design.counts) == 1 else "heterogeneous",
                    "bank_assignment": assignment.name,
                    "pipeline_concurrent": int(assignment.concurrent),
                    "projection_bank": assignment.projection.name,
                    "attention_bank": assignment.attention.name,
                    "projection_lane_budget": assignment.projection_lanes,
                    "attention_lane_budget": assignment.attention_lanes,
                    "tile_output_lanes": library.tile_output_lanes,
                    "tile_capacity_values": design.capacity_values,
                    "tile_macro_area_mm2": design.macro_area_mm2,
                    "tile_register_area_mm2": design.register_area_mm2,
                    "tile_area_mm2": design.area_mm2,
                    "tile_macro_count": design.macro_count,
                    "tile_macro_type_count": len(design.counts),
                    "tile_peak_compute_tops": design.peak_compute_tops,
                    "MACs": int(endpoint["MACs"]) + int(pipeline["MACs"]),
                    "fabric_energy_mJ": fabric_energy_j * 1e3,
                    "dynamic_energy_mJ": dynamic_energy_j * 1e3,
                    "macro_static_energy_mJ": static_energy_j * 1e3,
                    "latency_ms": latency_s * 1e3,
                    "endpoint_latency_ms": endpoint_latency_s * 1e3,
                    "transformer_latency_ms": float(pipeline["transformer_latency_s"]) * 1e3,
                    "projection_compute_latency_ms": float(pipeline["projection_compute_latency_s"]) * 1e3,
                    "attention_compute_latency_ms": float(pipeline["attention_compute_latency_s"]) * 1e3,
                    "transformer_bus_latency_ms": float(pipeline["transformer_bus_latency_s"]) * 1e3,
                    "total_bus_cycles": int(endpoint["total_bus_cycles"]) + int(pipeline["total_bus_cycles"]),
                    "transformer_bus_cycles": int(pipeline["total_bus_cycles"]),
                    "shared_bus_width_bits": library.tile_shared_bus_width_bits,
                    "feature_bus_width_bits": library.tile_shared_bus_width_bits,
                    "weight_bus_width_bits": library.tile_weight_bus_width_bits,
                    "tile_retain_features_across_nblocks": int(library.tile_retain_features_across_nblocks),
                    "tile_allow_intra_bank_weight_compute_overlap": int(library.tile_allow_intra_bank_weight_compute_overlap),
                    "vit_allow_cross_bank_weight_compute_overlap": int(library.vit_allow_cross_bank_weight_compute_overlap),
                    "tile_latency_batch_granularity": library.tile_latency_batch_granularity,
                    "tile_include_pipeline_fill_drain": int(library.tile_include_pipeline_fill_drain),
                    "full_tile_static_power_w": full_static_power_w,
                    "effective_TOPS_per_W": 2 * (int(endpoint["MACs"]) + int(pipeline["MACs"])) / fabric_energy_j / 1e12,
                    "schedule_trace": f"{endpoint['schedule_trace']}|{pipeline['schedule_trace']}",
                    "partition_budget": partition_budget,
                    "bank_assignment_count": len(assignments),
                    "energy_model_version": ENERGY_MODEL_VERSION,
                    "pipeline_model_version": PIPELINE_MODEL_VERSION,
                    "workload_model_version": WORKLOAD_MODEL_VERSION,
                }
            )
    aggregate_rows = _front(
        aggregate_rows,
        ("fabric_energy_mJ", "latency_ms"),
        aggregate_state_cap,
    )
    if partial_path:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = partial_path.with_suffix(partial_path.suffix + ".tmp")
        with temporary.open("wb") as file:
            pickle.dump(
                {
                    "signature": checkpoint_signature,
                    "layer_rows": layer_rows,
                    "aggregate_rows": aggregate_rows,
                },
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary.replace(partial_path)
    return layer_rows, aggregate_rows

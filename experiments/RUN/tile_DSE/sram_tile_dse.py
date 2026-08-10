"""Single-Tile Macro-composition enumeration and weight-stationary scheduling."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import ceil, log2, prod
from pathlib import Path
import sys
from time import perf_counter


RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from hwcomponents_library.library.aladdin import AladdinRegister

from sram_component_models import DigitalSramShiftAccumulator
from sram_macro_types import MacroLibrary, MacroType
from sram_workload import rank_variable_sizes, tensor_for_role


@dataclass(frozen=True)
class LayerShape:
    name: str
    reduction_length: int
    output_channels: int
    output_positions: int

    @property
    def output_blocks(self) -> int:
        return ceil(self.output_channels / 8)

    @property
    def weight_values(self) -> int:
        return self.reduction_length * self.output_channels

    @property
    def macs(self) -> int:
        return self.weight_values * self.output_positions


@dataclass(frozen=True)
class TileDesign:
    counts: tuple[tuple[str, int], ...]
    capacity_values: int
    macro_area_mm2: float
    register_area_mm2: float
    peak_compute_tops: float = 0.0

    @property
    def name(self) -> str:
        return "+".join(f"{count}x{name}" for name, count in self.counts)

    @property
    def macro_count(self) -> int:
        return sum(count for _, count in self.counts)

    @property
    def area_mm2(self) -> float:
        return self.macro_area_mm2 + self.register_area_mm2

    def macros(self, library: MacroLibrary) -> tuple[MacroType, ...]:
        return tuple(
            library.macro_types[name]
            for name, count in self.counts
            for _ in range(count)
        )


@dataclass(frozen=True)
class LaneProfile:
    signature: tuple[int, ...]
    tasks: tuple[MacroType, ...]
    task_rows: tuple[int, ...]
    wave_macro_counts: tuple[int, ...]
    latency_s: float
    weight_load_latency_s: float
    wave_task_rows: tuple[tuple[int, ...], ...] = ()
    wave_latencies_s: tuple[float, ...] = ()

    @property
    def wave_count(self) -> int:
        return len(self.wave_macro_counts)

    @property
    def cross_macro_reductions_per_output(self) -> int:
        return sum(count - 1 for count in self.wave_macro_counts)

    @property
    def cross_wave_reductions_per_output(self) -> int:
        return max(self.wave_count - 1, 0)


@dataclass(frozen=True)
class ScheduleState:
    energy_j: float
    latency_s: float
    lanes: tuple[tuple[tuple[int, ...], int], ...]


@dataclass(frozen=True)
class ScheduleTiming:
    compute_only_cycles: int
    macro_critical_cycles: int
    weight_load_critical_cycles: int
    feature_bus_bits: int
    weight_bus_bits: int
    feature_bus_cycles: int
    weight_bus_cycles: int
    total_bus_cycles: int
    latency_cycles: int


@dataclass
class ScheduleSearchMetrics:
    raw_group_count: int = 0
    nondominated_group_count: int = 0
    candidate_transitions: int = 0
    lane_limit_prunes: int = 0
    incomplete_lane_prunes: int = 0
    infeasible_completion_prunes: int = 0
    latency_limit_prunes: int = 0
    accepted_transitions: int = 0
    peak_transient_states: int = 0
    state_cap_applications: int = 0
    resource_front_calls: int = 0
    timing_cache_hits: int = 0
    timing_cache_misses: int = 0
    elapsed_s: float = 0.0


_INTERMEDIATE_STATE_PRUNE_THRESHOLD = 256
_TIMING_CACHE_MAX_ENTRIES = 131072


def layer_shapes(mappings, names: set[str] | None = None) -> list[LayerShape]:
    shapes = []
    for einsum_name in mappings.einsum_names:
        if names is not None and einsum_name not in names:
            continue
        spec = mappings.evaluated_specs[einsum_name]
        einsum = spec.workload.einsums[einsum_name]
        shapes.append(_layer_shape(einsum, spec.workload))
    return shapes


def layer_shapes_from_workload(
    workload, names: set[str] | None = None
) -> list[LayerShape]:
    """Extract P/K/N shapes without requiring a system-level hardware mapping."""
    return [
        _layer_shape(einsum, workload)
        for einsum in workload.einsums
        if names is None or einsum.name in names
    ]


def _layer_shape(einsum, workload) -> LayerShape:
    sizes = rank_variable_sizes(einsum, workload)
    input_tensor = tensor_for_role(einsum, "input")
    weight_tensor = tensor_for_role(einsum, "weight")
    output_tensor = tensor_for_role(einsum, "output")
    input_vars = set(einsum.tensor2rank_variables[input_tensor.name])
    weight_vars = set(einsum.tensor2rank_variables[weight_tensor.name])
    output_vars = set(einsum.tensor2rank_variables[output_tensor.name])
    output_channel_vars = output_vars & weight_vars - input_vars
    return LayerShape(
        name=einsum.name,
        reduction_length=prod(sizes[var] for var in input_vars & weight_vars - output_vars),
        output_channels=prod(sizes[var] for var in output_channel_vars),
        output_positions=prod(sizes[var] for var in output_vars - output_channel_vars),
    )


def enumerate_tile_designs(
    library: MacroLibrary,
    peak_compute_reference_tops: float,
    peak_compute_tolerance_ratio: float = 0.20,
    peak_compute_atol_tops: float = 1e-12,
) -> tuple[TileDesign, list[TileDesign]]:
    """Enumerate Tile compositions within a relative peak-compute window.

    Physical area and weight capacity remain reported diagnostics, but neither
    determines whether a composition enters the candidate pool.
    """
    if peak_compute_reference_tops <= 0:
        raise ValueError("peak_compute_reference_tops must be positive.")
    if not 0 <= peak_compute_tolerance_ratio < 1:
        raise ValueError("peak_compute_tolerance_ratio must be in [0, 1).")
    if peak_compute_atol_tops < 0:
        raise ValueError("peak_compute_atol_tops cannot be negative.")
    peak_compute_lower_tops = peak_compute_reference_tops * (
        1 - peak_compute_tolerance_ratio
    )
    peak_compute_upper_tops = peak_compute_reference_tops * (
        1 + peak_compute_tolerance_ratio
    )
    macro_types = tuple(sorted(library.macro_types.values(), key=lambda macro: macro.rows))
    m64 = library.macro_types["m64x8"]
    fixed_output_register_area_mm2 = _tile_output_register_area_mm2(
        library.tech_node,
        library.output_register_entry_bits,
        library.tile_output_lanes,
        m64.columns,
    )
    m64_input_register_area_mm2 = _macro_input_register_area_mm2(
        m64, library.tech_node
    )
    reference = TileDesign(
        counts=(("m64x8", 9),),
        capacity_values=9 * m64.rows * m64.columns,
        macro_area_mm2=9 * m64.area_mm2,
        register_area_mm2=(
            9 * m64_input_register_area_mm2
            + fixed_output_register_area_mm2
        ),
        peak_compute_tops=9 * m64.peak_tops,
    )
    candidates: list[TileDesign] = []

    def visit(
        index: int,
        counts: list[int],
        capacity: int,
        macro_area: float,
        register_area: float,
        peak_compute_tops: float,
    ) -> None:
        if peak_compute_tops > peak_compute_upper_tops + peak_compute_atol_tops:
            return
        if index == len(macro_types):
            if (
                peak_compute_lower_tops - peak_compute_atol_tops
                <= peak_compute_tops
                <= peak_compute_upper_tops + peak_compute_atol_tops
                and any(counts)
                and sum(count > 0 for count in counts)
                <= library.max_macro_types_per_tile
            ):
                candidates.append(
                    TileDesign(
                        counts=tuple(
                            (macro.name, count)
                            for macro, count in zip(macro_types, counts)
                            if count
                        ),
                        capacity_values=capacity,
                        macro_area_mm2=macro_area,
                        register_area_mm2=(
                            register_area + fixed_output_register_area_mm2
                        ),
                        peak_compute_tops=peak_compute_tops,
                    )
                )
            return
        macro = macro_types[index]
        macro_register_area = _macro_input_register_area_mm2(
            macro, library.tech_node
        )
        max_count = int(
            (peak_compute_upper_tops + peak_compute_atol_tops - peak_compute_tops)
            // macro.peak_tops
        )
        for count in range(max_count + 1):
            visit(
                index + 1,
                counts + [count],
                capacity + count * macro.rows * macro.columns,
                macro_area + count * macro.area_mm2,
                register_area + count * macro_register_area,
                peak_compute_tops + count * macro.peak_tops,
            )

    visit(0, [], 0, 0.0, 0.0, 0.0)
    candidates.sort(key=lambda design: (design.macro_count, design.capacity_values, design.name))
    return reference, candidates


def _vmm_latency_s(macro: MacroType) -> float:
    return macro.bit_cycles_per_full_vmm * macro.cycle_period_s


@lru_cache(maxsize=None)
def _register_access_energy_j(tech_node: float, width: int) -> tuple[float, float]:
    """Return Aladdin register read/write energy for one value of ``width`` bits."""

    register = AladdinRegister(tech_node=tech_node, width=width)
    return float(register.read().energy), float(register.write().energy)


@lru_cache(maxsize=None)
def _register_area_mm2(tech_node: float, width: int) -> float:
    return float(AladdinRegister(tech_node=tech_node, width=width).area) * 1e6


def _macro_input_register_area_mm2(
    macro: MacroType, tech_node: float
) -> float:
    return 2 * macro.rows * _register_area_mm2(tech_node, macro.input_bits)


def _tile_output_register_area_mm2(
    tech_node: float,
    output_register_entry_bits: int,
    output_lanes: int,
    outputs_per_lane: int,
) -> float:
    return (
        output_lanes
        * outputs_per_lane
        * _register_area_mm2(tech_node, output_register_entry_bits)
    )


def _macro_register_energy_per_invocation_j(
    macro: MacroType,
    active_rows: int,
    tech_node: float,
    output_register_entry_bits: int,
) -> tuple[float, float]:
    """Return input/output register dynamic energy for one physical VMM.

    One active ping-pong bank is written and read per invocation. Only rows used by
    this VMM are accessed. Each Macro output is written to and read from its single
    local partial-sum register bank once.
    """

    input_read, input_write = _register_access_energy_j(tech_node, macro.input_bits)
    output_read, output_write = _register_access_energy_j(
        tech_node, output_register_entry_bits
    )
    return (
        active_rows * (input_read + input_write),
        macro.columns * (output_read + output_write),
    )


def _inventory(design: TileDesign, macro_types: tuple[MacroType, ...]) -> tuple[int, ...]:
    counts = dict(design.counts)
    return tuple(counts.get(macro.name, 0) for macro in macro_types)


def _signature_macros(
    signature: tuple[int, ...], macro_types: tuple[MacroType, ...]
) -> tuple[MacroType, ...]:
    return tuple(
        macro
        for macro, count in zip(macro_types, signature)
        for _ in range(count)
    )


def _productive_group(
    signature: tuple[int, ...], macro_types: tuple[MacroType, ...], reduction_length: int
) -> bool:
    """Discard groups with a Macro that cannot improve one output-block execution."""

    macros = _signature_macros(signature, macro_types)
    if not macros:
        return False
    total_rows = sum(macro.rows for macro in macros)
    if total_rows < reduction_length:
        return True
    return total_rows - min(macro.rows for macro in macros) < reduction_length


@lru_cache(maxsize=None)
def _group_signatures(
    inventory: tuple[int, ...], macro_types: tuple[MacroType, ...], reduction_length: int
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        signature
        for signature in product(*(range(count + 1) for count in inventory))
        if any(signature) and _productive_group(signature, macro_types, reduction_length)
    )


def _group_profile(
    signature: tuple[int, ...],
    macro_types: tuple[MacroType, ...],
    reduction_length: int,
    weight_load_rows_per_cycle: int,
    weight_lines_per_bus_cycle: int,
) -> LaneProfile:
    """Return one output block's active Macro invocations and VMM makespan."""

    tasks = []
    task_rows = []
    wave_macro_counts = []
    wave_task_rows = []
    wave_latencies_s = []
    latency_s = 0.0
    weight_load_latency_s = 0.0
    remaining = reduction_length
    ordered = tuple(
        sorted(_signature_macros(signature, macro_types), key=lambda macro: macro.rows, reverse=True)
    )
    while remaining > 0:
        active = []
        active_rows = []
        for macro in ordered:
            if remaining <= 0:
                break
            rows = min(remaining, macro.rows)
            active.append(macro)
            active_rows.append(rows)
            remaining -= rows
        tasks.extend(active)
        task_rows.extend(active_rows)
        wave_macro_counts.append(len(active))
        wave_task_rows.append(tuple(active_rows))
        wave_latency_s = max(_vmm_latency_s(macro) for macro in active)
        latency_s += wave_latency_s
        wave_latencies_s.append(wave_latency_s)
        weight_load_cycles = max(
            ceil(max(active_rows) / weight_load_rows_per_cycle),
            ceil(sum(active_rows) / weight_lines_per_bus_cycle),
        )
        weight_load_latency_s += weight_load_cycles * active[0].cycle_period_s
    return LaneProfile(
        signature=signature,
        tasks=tuple(tasks),
        task_rows=tuple(task_rows),
        wave_macro_counts=tuple(wave_macro_counts),
        latency_s=latency_s,
        weight_load_latency_s=weight_load_latency_s,
        wave_task_rows=tuple(wave_task_rows),
        wave_latencies_s=tuple(wave_latencies_s),
    )


def _weight_load_cycles(
    row_loads: list[int],
    rows_per_cycle_per_macro: int,
    lines_per_bus_cycle: int,
) -> int:
    if not row_loads:
        return 0
    return max(
        ceil(max(row_loads) / rows_per_cycle_per_macro),
        ceil(sum(row_loads) / lines_per_bus_cycle),
    )


def _wave_feature_rows(
    profiles_wave_task_rows: tuple[tuple[tuple[int, ...], ...], ...],
    wave_index: int,
) -> int:
    """Return the union of K rows required by the active wave batch."""

    intervals = []
    for waves in profiles_wave_task_rows:
        if wave_index >= len(waves):
            continue
        start = sum(sum(rows) for rows in waves[:wave_index])
        intervals.append((start, start + sum(waves[wave_index])))
    if not intervals:
        return 0
    total = 0
    merged_start, merged_end = sorted(intervals)[0]
    for start, end in sorted(intervals)[1:]:
        if start > merged_end:
            total += merged_end - merged_start
            merged_start, merged_end = start, end
        else:
            merged_end = max(merged_end, end)
    return total + merged_end - merged_start


def _profile_wave_latency_s(profile: LaneProfile, wave_index: int) -> float:
    if profile.wave_latencies_s:
        return profile.wave_latencies_s[wave_index]
    return profile.latency_s / profile.wave_count


def _schedule_timing(
    layer: LayerShape,
    lanes: tuple[tuple[tuple[int, ...], int], ...],
    profiles: dict[tuple[int, ...], LaneProfile],
    cycle_period_s: float,
    input_bits: int,
    weight_line_bits: int,
    feature_bus_width_bits: int,
    weight_bus_width_bits: int,
    weight_load_rows_per_cycle: int,
) -> ScheduleTiming:
    if not lanes:
        return ScheduleTiming(*(0 for _ in range(9)))

    weight_lines_per_bus_cycle = max(weight_bus_width_bits // weight_line_bits, 1)
    jobs_per_lane = [jobs for _, jobs in lanes]
    lane_profiles = [profiles[signature] for signature, _ in lanes]
    weight_bus_cycles = 0
    feature_bus_cycles = 0
    feature_bus_bits = 0
    weight_bus_bits = 0
    compute_only_cycles = 0
    latency_cycles = 0
    # Active lanes change only at a lane's assigned-job count. Each interval
    # repeats an identical set of N-block/K-wave batches.
    previous_jobs = 0
    for job_boundary in sorted(set(jobs_per_lane)):
        active_profiles = tuple(
            profile
            for jobs, profile in zip(jobs_per_lane, lane_profiles)
            if jobs >= job_boundary
        )
        repetitions = job_boundary - previous_jobs
        wave_count = max(profile.wave_count for profile in active_profiles)
        for wave_index in range(wave_count):
            row_loads = [
                rows
                for profile in active_profiles
                if wave_index < profile.wave_count
                for rows in profile.wave_task_rows[wave_index]
            ]
            wave_weight_cycles = _weight_load_cycles(
                row_loads,
                weight_load_rows_per_cycle,
                weight_lines_per_bus_cycle,
            )
            wave_feature_rows = _wave_feature_rows(
                tuple(profile.wave_task_rows for profile in active_profiles),
                wave_index,
            )
            wave_feature_cycles = layer.output_positions * ceil(
                wave_feature_rows * input_bits / feature_bus_width_bits
            )
            wave_compute_cycles = max(
                round(
                    layer.output_positions
                    * _profile_wave_latency_s(profile, wave_index)
                    / cycle_period_s
                )
                for profile in active_profiles
                if wave_index < profile.wave_count
            )
            weight_bus_cycles += repetitions * wave_weight_cycles
            feature_bus_cycles += repetitions * wave_feature_cycles
            feature_bus_bits += (
                repetitions
                * layer.output_positions
                * wave_feature_rows
                * input_bits
            )
            weight_bus_bits += repetitions * sum(row_loads) * weight_line_bits
            compute_only_cycles += repetitions * wave_compute_cycles
            latency_cycles += repetitions * (
                wave_weight_cycles
                + max(wave_feature_cycles, wave_compute_cycles)
            )
        previous_jobs = job_boundary

    total_bus_cycles = feature_bus_cycles + weight_bus_cycles
    macro_critical_cycles = weight_bus_cycles + compute_only_cycles
    return ScheduleTiming(
        compute_only_cycles=compute_only_cycles,
        macro_critical_cycles=macro_critical_cycles,
        weight_load_critical_cycles=weight_bus_cycles,
        feature_bus_bits=feature_bus_bits,
        weight_bus_bits=weight_bus_bits,
        feature_bus_cycles=feature_bus_cycles,
        weight_bus_cycles=weight_bus_cycles,
        total_bus_cycles=total_bus_cycles,
        latency_cycles=latency_cycles,
    )


def _energy_latency_state_front(
    states: list[ScheduleState], epsilon: float
) -> list[ScheduleState]:
    ordered = sorted(states, key=lambda state: (state.energy_j, state.latency_s, state.lanes))
    front = []
    best_latency = float("inf")
    for state in ordered:
        if state.latency_s < best_latency * (1 - epsilon):
            front.append(state)
            best_latency = state.latency_s
    fastest = min(ordered, key=lambda state: state.latency_s)
    if fastest not in front:
        front.append(fastest)
    return sorted(front, key=lambda state: (state.energy_j, state.latency_s, state.lanes))


def _state_front(
    states: list[ScheduleState],
    epsilon: float = 0.0,
    preserve_lane_count: bool = False,
) -> list[ScheduleState]:
    if not preserve_lane_count:
        return _energy_latency_state_front(states, epsilon)

    # A state using fewer lanes may remain extensible even when it is currently
    # slower. Keep the E-T front under every active-lane budget.
    front = {
        state
        for lane_limit in sorted({len(state.lanes) for state in states})
        for state in _energy_latency_state_front(
            [state for state in states if len(state.lanes) <= lane_limit], epsilon
        )
    }
    return sorted(front, key=lambda state: (state.energy_j, state.latency_s, state.lanes))


def _extend_exact_state_front(
    states: list[ScheduleState], candidate: ScheduleState
) -> None:
    """Bound transient DP storage without changing the exact local Pareto front."""

    states.append(candidate)
    if len(states) >= _INTERMEDIATE_STATE_PRUNE_THRESHOLD:
        states[:] = _state_front(states, preserve_lane_count=True)


@lru_cache(maxsize=8192)
def _strict_resource_subsets(used: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for subset in product(*(range(value + 1) for value in used))
        if subset != used
    )


def _canonical_lane_key(
    lanes: tuple[tuple[tuple[int, ...], int], ...],
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Canonical timing-cache key; physical lane order does not affect timing."""

    return tuple(sorted(lanes))


def _resource_front(
    states: dict[tuple[tuple[int, ...], int], list[ScheduleState]],
    epsilon: float,
) -> dict[tuple[tuple[int, ...], int], list[ScheduleState]]:
    """Prune states dominated in energy, latency, Macro resources, and lanes."""

    exact_fronts = {
        key: _state_front(candidates, epsilon, preserve_lane_count=True)
        for key, candidates in states.items()
    }
    by_completed_jobs: dict[int, list[tuple[int, ...]]] = {}
    for used, completed_jobs in exact_fronts:
        by_completed_jobs.setdefault(completed_jobs, []).append(used)

    output: dict[tuple[tuple[int, ...], int], list[ScheduleState]] = {}
    for completed_jobs, resource_states in by_completed_jobs.items():
        for used in sorted(resource_states, key=lambda value: (sum(value), value)):
            key = (used, completed_jobs)
            kept = []
            dominating_states = [
                state
                for subset in _strict_resource_subsets(used)
                for state in output.get((subset, completed_jobs), ())
            ]
            for state in exact_fronts[key]:
                if any(
                    existing.energy_j <= state.energy_j
                    and existing.latency_s <= state.latency_s
                    and len(existing.lanes) <= len(state.lanes)
                    for existing in dominating_states
                ):
                    continue
                kept.append(state)
            if kept:
                output[key] = kept
    return output


def _cap_resource_states(
    states: dict[tuple[tuple[int, ...], int], list[ScheduleState]],
    state_cap: int,
) -> tuple[dict[tuple[tuple[int, ...], int], list[ScheduleState]], bool]:
    """Keep a deterministic, progress-stratified sample of a large DP frontier."""

    state_count = sum(len(candidates) for candidates in states.values())
    if state_count <= state_cap:
        return states, False

    by_completed_jobs: dict[
        int, list[tuple[tuple[tuple[int, ...], int], ScheduleState]]
    ] = {}
    for key, candidates in states.items():
        by_completed_jobs.setdefault(key[1], []).extend(
            (key, state) for state in candidates
        )

    progress_values = sorted(by_completed_jobs)
    base_budget, extra = divmod(state_cap, len(progress_values))
    selected: list[tuple[tuple[tuple[int, ...], int], ScheduleState]] = []
    for progress_index, completed_jobs in enumerate(progress_values):
        candidates = by_completed_jobs[completed_jobs]
        budget = base_budget + (progress_index < extra)
        if len(candidates) <= budget:
            selected.extend(candidates)
            continue

        orderings = [
            sorted(candidates, key=lambda item: (item[1].energy_j, item[1].latency_s)),
            sorted(candidates, key=lambda item: (item[1].latency_s, item[1].energy_j)),
            sorted(candidates, key=lambda item: (len(item[1].lanes), item[1].energy_j)),
            sorted(candidates, key=lambda item: (sum(item[0][0]), item[1].energy_j)),
        ]
        orderings.extend(
            sorted(candidates, key=lambda item, index=index: (item[0][0][index], item[1].energy_j))
            for index in range(len(candidates[0][0][0]))
        )
        chosen: dict[
            tuple[tuple[tuple[int, ...], int], ScheduleState], None
        ] = {}
        samples_per_ordering = max(1, (budget + len(orderings) - 1) // len(orderings))
        for ordered in orderings:
            for sample_index in range(samples_per_ordering):
                position = round(
                    sample_index * (len(ordered) - 1) / max(samples_per_ordering - 1, 1)
                )
                chosen[ordered[position]] = None
                if len(chosen) == budget:
                    break
            if len(chosen) == budget:
                break
        if len(chosen) < budget:
            for item in orderings[0]:
                chosen[item] = None
                if len(chosen) == budget:
                    break
        selected.extend(chosen)

    output: dict[tuple[tuple[int, ...], int], list[ScheduleState]] = {}
    for key, state in selected:
        output.setdefault(key, []).append(state)
    return output, True


def _nondominated_groups(
    groups: tuple[tuple[int, ...], ...],
    macro_types: tuple[MacroType, ...],
    reduction_length: int,
    weight_load_rows_per_cycle: int,
    weight_lines_per_bus_cycle: int,
) -> tuple[tuple[int, ...], ...]:
    profiles = {
        group: _group_profile(
            group,
            macro_types,
            reduction_length,
            weight_load_rows_per_cycle,
            weight_lines_per_bus_cycle,
        )
        for group in groups
    }
    energies = {
        group: sum(
            macro.dynamic_energy_for_active_rows_j(rows)
            for macro, rows in zip(profile.tasks, profile.task_rows)
        )
        for group, profile in profiles.items()
    }
    return tuple(
        group
        for group in groups
        if not any(
            other != group
            and all(left <= right for left, right in zip(other, group))
            and energies[other] <= energies[group]
            and profiles[other].cross_wave_reductions_per_output
            <= profiles[group].cross_wave_reductions_per_output
            and (
                profiles[other].latency_s
                + profiles[other].weight_load_latency_s
                <= profiles[group].latency_s
                + profiles[group].weight_load_latency_s
            )
            for other in groups
        )
    )


def _schedule_front(
    layer: LayerShape,
    inventory: tuple[int, ...],
    macro_types: tuple[MacroType, ...],
    groups: tuple[tuple[int, ...], ...],
    psum_vector_energy_j: float,
    cross_wave_psum_energy_j: float,
    tech_node: float,
    output_register_entry_bits: int,
    max_output_lanes: int,
    weight_load_rows_per_cycle: int,
    feature_bus_width_bits: int,
    weight_bus_width_bits: int,
    bus_energy_per_bit_j: float,
    latency_cap_ratio: float,
    local_epsilon: float,
    local_state_cap: int,
    metrics: ScheduleSearchMetrics | None = None,
) -> tuple[list[ScheduleState], float, bool]:
    """Solve Macro grouping and N-block assignment with bounded Pareto DP."""

    zero_inventory = (0,) * len(inventory)
    states: dict[tuple[tuple[int, ...], int], list[ScheduleState]] = {
        (zero_inventory, 0): [ScheduleState(0.0, 0.0, ())]
    }
    profiles = {
        signature: _group_profile(
            signature,
            macro_types,
            layer.reduction_length,
            weight_load_rows_per_cycle,
            max(
                weight_bus_width_bits
                // (macro_types[0].columns * macro_types[0].weight_bits),
                1,
            ),
        )
        for signature in groups
    }
    cycle_period_s = macro_types[0].cycle_period_s
    input_bits = macro_types[0].input_bits
    weight_line_bits = macro_types[0].columns * macro_types[0].weight_bits
    tile_static_power_w = sum(
        count * macro.static_power_w
        for count, macro in zip(inventory, macro_types)
    )
    timing_cache: OrderedDict[
        tuple[tuple[tuple[int, ...], int], ...], ScheduleTiming
    ] = OrderedDict()

    def timing(
        lanes: tuple[tuple[tuple[int, ...], int], ...]
    ) -> ScheduleTiming:
        key = _canonical_lane_key(lanes)
        try:
            result = timing_cache.pop(key)
        except KeyError:
            if metrics is not None:
                metrics.timing_cache_misses += 1
            result = _schedule_timing(
                layer,
                key,
                profiles,
                cycle_period_s,
                input_bits,
                weight_line_bits,
                feature_bus_width_bits,
                weight_bus_width_bits,
                weight_load_rows_per_cycle,
            )
            if len(timing_cache) >= _TIMING_CACHE_MAX_ENTRIES:
                timing_cache.popitem(last=False)
        else:
            if metrics is not None:
                metrics.timing_cache_hits += 1
        timing_cache[key] = result
        return result

    feasible_latency_s = float("inf")
    for signature in groups:
        copies = min(
            max_output_lanes,
            layer.output_blocks,
            min(
                (
                    available // required
                    for available, required in zip(inventory, signature)
                    if required
                ),
                default=1,
            ),
        )
        jobs, extra = divmod(layer.output_blocks, copies)
        lanes = tuple(
            (signature, jobs + (index < extra)) for index in range(copies)
        )
        feasible_latency_s = min(
            feasible_latency_s, timing(lanes).latency_cycles * cycle_period_s
        )
    latency_cap_s = feasible_latency_s * latency_cap_ratio

    state_cap_applied = False
    state_count = 1
    if metrics is not None:
        metrics.peak_transient_states = max(metrics.peak_transient_states, state_count)
    for signature_index, signature in enumerate(groups):
        future_groups = groups[signature_index + 1 :]

        @lru_cache(maxsize=None)
        def future_group_fits(used: tuple[int, ...]) -> bool:
            return any(
                all(
                    occupied + required <= available
                    for occupied, required, available in zip(
                        used, future_signature, inventory
                    )
                )
                for future_signature in future_groups
            )

        def can_still_complete(
            used: tuple[int, ...], completed_jobs: int, lane_count: int
        ) -> bool:
            return completed_jobs == layer.output_blocks or (
                lane_count < max_output_lanes and future_group_fits(used)
            )

        profile = profiles[signature]
        macro_energy_per_job = sum(
            task.dynamic_energy_for_active_rows_j(rows)
            for task, rows in zip(profile.tasks, profile.task_rows)
        )
        weight_load_energy_per_job = sum(
            rows * task.weight_row_write_energy_j
            for task, rows in zip(profile.tasks, profile.task_rows)
        )
        input_register_energy_per_job = sum(
            _macro_register_energy_per_invocation_j(
                task, rows, tech_node, output_register_entry_bits
            )[0]
            for task, rows in zip(profile.tasks, profile.task_rows)
        )
        output_read_energy_j, output_write_energy_j = _register_access_energy_j(
            tech_node, output_register_entry_bits
        )
        output_register_energy_per_output = (
            profile.wave_count
            * macro_types[0].columns
            * (output_read_energy_j + output_write_energy_j)
        )
        energy_per_job = weight_load_energy_per_job + layer.output_positions * (
            macro_energy_per_job
            + input_register_energy_per_job
            + output_register_energy_per_output
            + profile.cross_macro_reductions_per_output * psum_vector_energy_j
            + profile.cross_wave_reductions_per_output
            * cross_wave_psum_energy_j
        )
        updated = {}
        for key, candidates in states.items():
            viable = [
                state
                for state in candidates
                if can_still_complete(key[0], key[1], len(state.lanes))
            ]
            if metrics is not None:
                metrics.infeasible_completion_prunes += len(candidates) - len(viable)
            if viable:
                updated[key] = viable
        state_count = sum(len(candidates) for candidates in updated.values())
        for (used, completed_jobs), current_states in states.items():
            prepared_states = []
            for state in current_states:
                state_timing = timing(state.lanes)
                state_bus_energy_j = (
                    state_timing.feature_bus_bits + state_timing.weight_bus_bits
                ) * bus_energy_per_bit_j
                prepared_states.append(
                    (
                        state,
                        state.energy_j
                        - tile_static_power_w * state.latency_s
                        - state_bus_energy_j,
                    )
                )
            max_copies = min(
                (
                    (available - occupied) // required
                    for available, occupied, required in zip(inventory, used, signature)
                    if required
                ),
                default=0,
            )
            max_copies = min(
                max_copies,
                layer.output_blocks - completed_jobs,
                max_output_lanes,
            )
            for copies in range(1, max_copies + 1):
                next_used = tuple(
                    occupied + copies * required
                    for occupied, required in zip(used, signature)
                )
                has_future_resource_path = future_group_fits(next_used)
                flexible_states = tuple(
                    item
                    for item in prepared_states
                    if len(item[0].lanes) + copies < max_output_lanes
                    and has_future_resource_path
                )
                finishing_states = tuple(
                    item
                    for item in prepared_states
                    if len(item[0].lanes) + copies == max_output_lanes
                )
                resource_finishing_states = tuple(
                    item
                    for item in prepared_states
                    if len(item[0].lanes) + copies < max_output_lanes
                    and not has_future_resource_path
                )
                lane_rejected_state_count = (
                    len(prepared_states)
                    - len(flexible_states)
                    - len(finishing_states)
                    - len(resource_finishing_states)
                )
                max_assigned_jobs = layer.output_blocks - completed_jobs
                assignment_count = max_assigned_jobs - copies + 1
                if metrics is not None:
                    metrics.candidate_transitions += (
                        lane_rejected_state_count * assignment_count
                        + len(finishing_states) * max(assignment_count - 1, 0)
                    )
                    metrics.lane_limit_prunes += (
                        lane_rejected_state_count * assignment_count
                    )
                    metrics.incomplete_lane_prunes += (
                        len(finishing_states) * max(assignment_count - 1, 0)
                    )
                    metrics.candidate_transitions += len(
                        resource_finishing_states
                    ) * max(assignment_count - 1, 0)
                    metrics.infeasible_completion_prunes += len(
                        resource_finishing_states
                    ) * max(assignment_count - 1, 0)
                for assigned_jobs in range(copies, max_assigned_jobs + 1):
                    candidate_states = flexible_states
                    if assigned_jobs == max_assigned_jobs:
                        candidate_states += finishing_states + resource_finishing_states
                    if not candidate_states:
                        continue
                    jobs_per_lane, extra = divmod(assigned_jobs, copies)
                    lane_jobs = [jobs_per_lane + 1] * extra + [jobs_per_lane] * (
                        copies - extra
                    )
                    key = (next_used, completed_jobs + assigned_jobs)
                    candidates = updated.get(key)
                    previous_candidate_count = len(candidates) if candidates else 0
                    for state, state_dynamic_energy_j in candidate_states:
                        if metrics is not None:
                            metrics.candidate_transitions += 1
                        candidate_lanes = state.lanes + tuple(
                            (signature, jobs) for jobs in lane_jobs
                        )
                        candidate_timing = timing(candidate_lanes)
                        candidate_latency_s = (
                            candidate_timing.latency_cycles * cycle_period_s
                        )
                        if candidate_latency_s > latency_cap_s:
                            if metrics is not None:
                                metrics.latency_limit_prunes += 1
                            continue
                        candidate_bus_energy_j = (
                            candidate_timing.feature_bus_bits
                            + candidate_timing.weight_bus_bits
                        ) * bus_energy_per_bit_j
                        if candidates is None:
                            candidates = []
                            updated[key] = candidates
                        _extend_exact_state_front(
                            candidates,
                            ScheduleState(
                                energy_j=(
                                    state_dynamic_energy_j
                                    + assigned_jobs * energy_per_job
                                    + candidate_bus_energy_j
                                    + tile_static_power_w * candidate_latency_s
                                ),
                                latency_s=candidate_latency_s,
                                lanes=candidate_lanes,
                            ),
                        )
                        if metrics is not None:
                            metrics.accepted_transitions += 1
                    state_count += (
                        len(candidates) - previous_candidate_count if candidates else 0
                    )
                    if metrics is not None:
                        metrics.peak_transient_states = max(
                            metrics.peak_transient_states, state_count
                        )
                    if state_count > 2 * local_state_cap:
                        updated, capped = _cap_resource_states(updated, local_state_cap)
                        state_cap_applied |= capped
                        if capped and metrics is not None:
                            metrics.state_cap_applications += 1
                        state_count = sum(
                            len(candidate_states)
                            for candidate_states in updated.values()
                        )
        if metrics is not None:
            metrics.resource_front_calls += 1
        states = _resource_front(updated, local_epsilon)
        states, capped = _cap_resource_states(states, local_state_cap)
        state_cap_applied |= capped
        if capped and metrics is not None:
            metrics.state_cap_applications += 1
        state_count = sum(len(candidates) for candidates in states.values())
    completed = [
        state
        for (_, completed_jobs), candidates in states.items()
        if completed_jobs == layer.output_blocks
        for state in candidates
    ]
    return _state_front(completed, local_epsilon), latency_cap_s, state_cap_applied


def _mapping_class(lanes: tuple[tuple[tuple[int, ...], int], ...]) -> str:
    if len(lanes) == 1:
        return "k_parallel"
    if all(sum(group) == 1 for group, _ in lanes):
        return "n_parallel"
    return "mixed"


def _group_label(signature: tuple[int, ...], macro_types: tuple[MacroType, ...]) -> str:
    return "+".join(
        f"{count}x{macro.name}"
        for macro, count in zip(macro_types, signature)
        if count
    )


def _psum_vector_energy_j(layer: LayerShape, macros: tuple[MacroType, ...], tech_node: float) -> tuple[int, float]:
    macro = macros[0]
    width = macro.input_bits + macro.weight_bits + ceil(log2(max(layer.reduction_length, 1)))
    energy = 8 * float(
        DigitalSramShiftAccumulator(tech_node=tech_node, width=width).read().energy
    )
    return width, energy


def _evaluate_partition(
    layer: LayerShape,
    design: TileDesign,
    macros: tuple[MacroType, ...],
    macro_types: tuple[MacroType, ...],
    state: ScheduleState,
    psum_width_bits: int,
    psum_vector_energy_j: float,
    cross_wave_psum_energy_j: float,
    tech_node: float,
    output_register_entry_bits: int,
    max_macro_types_per_tile: int,
    tile_output_lanes: int,
    weight_load_rows_per_cycle: int,
    feature_bus_width_bits: int,
    weight_bus_width_bits: int,
    bus_energy_per_bit_j: float,
    retain_features_across_nblocks: bool,
    allow_intra_bank_weight_compute_overlap: bool,
    allow_cross_bank_weight_compute_overlap: bool,
    latency_batch_granularity: str,
    include_pipeline_fill_drain: bool,
    latency_cap_s: float,
    latency_cap_ratio: float,
    local_epsilon: float,
    local_state_cap: int,
    local_state_cap_applied: bool,
    reference: TileDesign,
) -> dict[str, float | int | str]:
    weight_line_bits = macro_types[0].columns * macro_types[0].weight_bits
    weight_lines_per_bus_cycle = max(weight_bus_width_bits // weight_line_bits, 1)
    profile_by_signature = {
        group: _group_profile(
            group,
            macro_types,
            layer.reduction_length,
            weight_load_rows_per_cycle,
            weight_lines_per_bus_cycle,
        )
        for group, _ in state.lanes
    }
    profiles = tuple(profile_by_signature[group] for group, _ in state.lanes)
    timing = _schedule_timing(
        layer,
        state.lanes,
        profile_by_signature,
        macro_types[0].cycle_period_s,
        macro_types[0].input_bits,
        weight_line_bits,
        feature_bus_width_bits,
        weight_bus_width_bits,
        weight_load_rows_per_cycle,
    )
    jobs_per_lane = [jobs for _, jobs in state.lanes]
    invocations_by_type = {macro.name: 0 for macro in macro_types if macro in macros}
    cross_macro_psum_vectors = 0
    cross_wave_psum_vectors = 0
    for jobs, profile in zip(jobs_per_lane, profiles):
        cross_macro_psum_vectors += (
            jobs
            * layer.output_positions
            * profile.cross_macro_reductions_per_output
        )
        cross_wave_psum_vectors += (
            jobs
            * layer.output_positions
            * profile.cross_wave_reductions_per_output
        )
        for macro in macro_types:
            if macro.name in invocations_by_type:
                invocations_by_type[macro.name] += (
                    jobs * layer.output_positions * profile.tasks.count(macro)
                )
    macro_dynamic_energy_j = sum(
        jobs
        * layer.output_positions
        * sum(
            task.dynamic_energy_for_active_rows_j(rows)
            for task, rows in zip(profile.tasks, profile.task_rows)
        )
        for jobs, profile in zip(jobs_per_lane, profiles)
    )
    macro_static_power_w = sum(macro.static_power_w for macro in macros)
    macro_static_energy_j = macro_static_power_w * state.latency_s
    macro_energy_j = macro_dynamic_energy_j + macro_static_energy_j
    input_register_read_entries = sum(
        jobs * layer.output_positions * sum(profile.task_rows)
        for jobs, profile in zip(jobs_per_lane, profiles)
    )
    output_register_read_entries = sum(
        jobs
        * layer.output_positions
        * profile.wave_count
        * macro_types[0].columns
        for jobs, profile in zip(jobs_per_lane, profiles)
    )
    input_register_energy_j = sum(
        jobs
        * layer.output_positions
        * sum(
            _macro_register_energy_per_invocation_j(
                task, rows, tech_node, output_register_entry_bits
            )[0]
            for task, rows in zip(profile.tasks, profile.task_rows)
        )
        for jobs, profile in zip(jobs_per_lane, profiles)
    )
    output_read_energy_j, output_write_energy_j = _register_access_energy_j(
        tech_node, output_register_entry_bits
    )
    output_register_energy_j = output_register_read_entries * (
        output_read_energy_j + output_write_energy_j
    )
    register_energy_j = input_register_energy_j + output_register_energy_j
    psum_vectors = cross_macro_psum_vectors + cross_wave_psum_vectors
    cross_macro_psum_energy_j = cross_macro_psum_vectors * psum_vector_energy_j
    cross_wave_psum_energy_j_total = (
        cross_wave_psum_vectors * cross_wave_psum_energy_j
    )
    psum_energy_j = cross_macro_psum_energy_j + cross_wave_psum_energy_j_total
    compute_only_latency_s = (
        timing.compute_only_cycles * macro_types[0].cycle_period_s
    )
    input_broadcast_rounds = max(jobs_per_lane)
    input_broadcast_values = (
        input_broadcast_rounds
        * layer.reduction_length
        * layer.output_positions
    )
    core_psum_buffer_vectors = 0
    weight_load_makespan_s = (
        timing.weight_load_critical_cycles * macro_types[0].cycle_period_s
    )
    weight_load_added_latency_s = state.latency_s - compute_only_latency_s
    weight_load_rows = sum(
        jobs * sum(profile.task_rows)
        for jobs, profile in zip(jobs_per_lane, profiles)
    )
    weight_load_energy_j = sum(
        jobs
        * sum(
            rows * task.weight_row_write_energy_j
            for task, rows in zip(profile.tasks, profile.task_rows)
        )
        for jobs, profile in zip(jobs_per_lane, profiles)
    )
    bus_energy_j = (
        timing.feature_bus_bits + timing.weight_bus_bits
    ) * bus_energy_per_bit_j
    fabric_energy_j = (
        macro_energy_j
        + register_energy_j
        + psum_energy_j
        + weight_load_energy_j
        + bus_energy_j
    )
    macro_capacity_macs = sum(
        invocations_by_type[macro.name] * macro.rows * macro.columns
        for macro in macro_types
        if macro.name in invocations_by_type
    )
    macro_busy_time_s = sum(
        invocations_by_type[macro.name] * _vmm_latency_s(macro)
        for macro in macro_types
        if macro.name in invocations_by_type
    )
    macro_time_utilization = macro_busy_time_s / (design.macro_count * state.latency_s)
    macro_row_utilization = layer.macs / macro_capacity_macs
    row_sizes = [macro.rows for macro in macros]
    partition_labels = [_group_label(profile.signature, macro_types) for profile in profiles]
    schedule_signature = "/".join(
        f"{label}@{jobs}" for label, jobs in zip(partition_labels, jobs_per_lane)
    )
    return {
        "layer": layer.name,
        "design": design.name,
        "mapping": _mapping_class(state.lanes),
        "schedule_signature": schedule_signature,
        "K": layer.reduction_length,
        "N": layer.output_channels,
        "output_positions": layer.output_positions,
        "MACs": layer.macs,
        "macro_counts": "+".join(f"{count}x{name}" for name, count in design.counts),
        "tile_capacity_values": design.capacity_values,
        "capacity_ratio": design.capacity_values / reference.capacity_values,
        "tile_macro_area_mm2": design.macro_area_mm2,
        "tile_register_area_mm2": design.register_area_mm2,
        "tile_area_mm2": design.area_mm2,
        "area_ratio": design.area_mm2 / reference.area_mm2,
        "tile_macro_count": design.macro_count,
        "tile_macro_type_count": len(design.counts),
        "tile_peak_compute_tops": design.peak_compute_tops,
        "tile_peak_compute_ratio": design.peak_compute_tops
        / reference.peak_compute_tops,
        "tile_peak_compute_density_tops_per_mm2": design.peak_compute_tops
        / design.area_mm2,
        "max_macro_types_per_tile": max_macro_types_per_tile,
        "tile_output_lanes": tile_output_lanes,
        "macro_row_span": max(row_sizes) / min(row_sizes),
        "lane_count": len(state.lanes),
        "active_lane_count": len(state.lanes),
        "k_partitions_by_lane": "|".join(
            "+".join(macro.name for macro in profile.tasks) for profile in profiles
        ),
        "macro_invocations": sum(invocations_by_type.values()),
        "input_broadcast_rounds": input_broadcast_rounds,
        "input_broadcast_values": input_broadcast_values,
        "macro_row_utilization": macro_row_utilization,
        "macro_time_utilization": macro_time_utilization,
        "effective_macro_utilization": macro_row_utilization * macro_time_utilization,
        "psum_vectors": psum_vectors,
        "cross_macro_psum_vectors": cross_macro_psum_vectors,
        "cross_wave_psum_vectors": cross_wave_psum_vectors,
        "core_psum_buffer_vectors": core_psum_buffer_vectors,
        "cross_wave_psum_resolved_in_tile": 1,
        "max_k_wave_count": max(profile.wave_count for profile in profiles),
        "k_wave_counts_by_lane": "|".join(
            str(profile.wave_count) for profile in profiles
        ),
        "psum_width_bits": psum_width_bits,
        "psum_vector_energy_j": psum_vector_energy_j,
        "cross_macro_psum_energy_j": cross_macro_psum_energy_j,
        "cross_wave_psum_energy_j": cross_wave_psum_energy_j_total,
        "macro_energy_j": macro_energy_j,
        "macro_dynamic_energy_j": macro_dynamic_energy_j,
        "macro_static_energy_j": macro_static_energy_j,
        "macro_static_power_w": macro_static_power_w,
        "macro_static_power_fraction": macros[0].static_power_fraction,
        "macro_partial_vmm_fixed_energy_by_type_j": "|".join(
            f"{macro.name}:{macro.partial_vmm_fixed_energy_j:.12g}"
            for macro in sorted(set(macros), key=lambda item: item.rows)
        ),
        "macro_partial_vmm_energy_per_active_row_by_type_j": "|".join(
            f"{macro.name}:{macro.partial_vmm_energy_per_active_row_j:.12g}"
            for macro in sorted(set(macros), key=lambda item: item.rows)
        ),
        "register_model": "aladdin_register_22nm_read_write_v1",
        "input_register_entry_bits": macros[0].input_bits,
        "output_register_entry_bits": output_register_entry_bits,
        "input_register_read_entries": input_register_read_entries,
        "input_register_write_entries": input_register_read_entries,
        "output_register_read_entries": output_register_read_entries,
        "output_register_write_entries": output_register_read_entries,
        "input_register_energy_j": input_register_energy_j,
        "output_register_energy_j": output_register_energy_j,
        "register_energy_j": register_energy_j,
        "psum_energy_j": psum_energy_j,
        "fabric_energy_j": fabric_energy_j,
        "compute_only_latency_s": compute_only_latency_s,
        "weight_load_makespan_s": weight_load_makespan_s,
        "weight_load_added_latency_s": weight_load_added_latency_s,
        "feature_bus_bits": timing.feature_bus_bits,
        "weight_bus_bits": timing.weight_bus_bits,
        "feature_bus_cycles": timing.feature_bus_cycles,
        "weight_bus_cycles": timing.weight_bus_cycles,
        "total_bus_cycles": timing.total_bus_cycles,
        "shared_bus_width_bits": feature_bus_width_bits,
        "feature_bus_width_bits": feature_bus_width_bits,
        "weight_bus_width_bits": weight_bus_width_bits,
        "tile_retain_features_across_nblocks": int(
            retain_features_across_nblocks
        ),
        "tile_allow_intra_bank_weight_compute_overlap": int(
            allow_intra_bank_weight_compute_overlap
        ),
        "vit_allow_cross_bank_weight_compute_overlap": int(
            allow_cross_bank_weight_compute_overlap
        ),
        "tile_latency_batch_granularity": latency_batch_granularity,
        "tile_include_pipeline_fill_drain": int(include_pipeline_fill_drain),
        "bus_energy_per_bit_j": bus_energy_per_bit_j,
        "bus_energy_j": bus_energy_j,
        "bus_added_latency_s": (
            timing.latency_cycles - timing.macro_critical_cycles
        ) * macro_types[0].cycle_period_s,
        "latency_s": state.latency_s,
        "latency_cap_s": latency_cap_s,
        "latency_cap_ratio": latency_cap_ratio,
        "local_pareto_epsilon": local_epsilon,
        "local_schedule_state_cap": local_state_cap,
        "local_schedule_state_cap_applied": int(local_state_cap_applied),
        "weight_values": layer.weight_values,
        "weight_load_values": layer.weight_values,
        "weight_load_rows": weight_load_rows,
        "weight_load_events": sum(
            jobs * len(profile.tasks) for jobs, profile in zip(jobs_per_lane, profiles)
        ),
        "weight_load_rows_per_cycle_per_macro": weight_load_rows_per_cycle,
        "weight_load_energy_j": weight_load_energy_j,
        "weight_reuse_per_load": layer.output_positions,
        "weight_fits_tile": int(layer.weight_values <= design.capacity_values),
        "weight_residency_epochs": ceil(layer.weight_values / design.capacity_values),
        "TOPS_per_W": 2 * layer.macs / fabric_energy_j / 1e12,
    }


def evaluate_tile_schedules(
    layer: LayerShape,
    design: TileDesign,
    library: MacroLibrary,
    tech_node: float,
    reference: TileDesign,
    latency_cap_ratio: float = 4.0,
    local_epsilon: float = 0.001,
    local_state_cap: int = 4096,
    cross_wave_psum_energy_j: float | None = None,
    metrics: ScheduleSearchMetrics | None = None,
) -> list[dict[str, float | int | str]]:
    """Enumerate local weight-stationary schedules and retain their E-T front."""

    macro_types = tuple(sorted(library.macro_types.values(), key=lambda macro: macro.rows))
    macros = design.macros(library)
    inventory = _inventory(design, macro_types)
    weight_line_bits = macro_types[0].columns * macro_types[0].weight_bits
    weight_lines_per_bus_cycle = max(
        library.tile_weight_bus_width_bits // weight_line_bits, 1
    )
    started_at = perf_counter()
    raw_groups = _group_signatures(inventory, macro_types, layer.reduction_length)
    groups = _nondominated_groups(
        raw_groups,
        macro_types,
        layer.reduction_length,
        library.weight_load_rows_per_cycle,
        weight_lines_per_bus_cycle,
    )
    psum_width_bits, psum_vector_energy_j = _psum_vector_energy_j(
        layer, macros, tech_node
    )
    if cross_wave_psum_energy_j is None:
        cross_wave_psum_energy_j = psum_vector_energy_j
    states, latency_cap_s, local_state_cap_applied = _schedule_front(
        layer,
        inventory,
        macro_types,
        groups,
        psum_vector_energy_j,
        cross_wave_psum_energy_j,
        tech_node,
        library.output_register_entry_bits,
        library.tile_output_lanes,
        library.weight_load_rows_per_cycle,
        library.tile_shared_bus_width_bits,
        library.tile_weight_bus_width_bits,
        library.tile_bus_energy_per_bit_j,
        latency_cap_ratio,
        local_epsilon,
        local_state_cap,
        metrics,
    )
    rows = [
        _evaluate_partition(
            layer,
            design,
            macros,
            macro_types,
            state,
            psum_width_bits,
            psum_vector_energy_j,
            cross_wave_psum_energy_j,
            tech_node,
            library.output_register_entry_bits,
            library.max_macro_types_per_tile,
            library.tile_output_lanes,
            library.weight_load_rows_per_cycle,
            library.tile_shared_bus_width_bits,
            library.tile_weight_bus_width_bits,
            library.tile_bus_energy_per_bit_j,
            library.tile_retain_features_across_nblocks,
            library.tile_allow_intra_bank_weight_compute_overlap,
            library.vit_allow_cross_bank_weight_compute_overlap,
            library.tile_latency_batch_granularity,
            library.tile_include_pipeline_fill_drain,
            latency_cap_s,
            latency_cap_ratio,
            local_epsilon,
            local_state_cap,
            local_state_cap_applied,
            reference,
        )
        for state in states
    ]
    result = pareto_tile_core_resources(rows)
    if metrics is not None:
        metrics.raw_group_count = len(raw_groups)
        metrics.nondominated_group_count = len(groups)
        metrics.elapsed_s = perf_counter() - started_at
    return result


def dominates(left: dict[str, float | int | str], right: dict[str, float | int | str]) -> bool:
    objectives = ("fabric_energy_j", "latency_s", "tile_area_mm2", "tile_macro_count")
    return all(float(left[key]) <= float(right[key]) for key in objectives) and any(
        float(left[key]) < float(right[key]) for key in objectives
    )


def pareto(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    objectives = ("fabric_energy_j", "latency_s", "tile_area_mm2", "tile_macro_count")
    ordered = sorted(
        rows,
        key=lambda row: (
            *(float(row[key]) for key in objectives),
            str(row.get("design", "")),
            str(row.get("schedule_signature", "")),
        ),
    )
    front = []
    for row in ordered:
        equivalent = any(
            all(float(row[key]) == float(existing[key]) for key in objectives)
            for existing in front
        )
        if not equivalent and not any(dominates(existing, row) for existing in front):
            front.append(row)
    return front


def pareto_energy_latency(
    rows: list[dict[str, float | int | str]]
) -> list[dict[str, float | int | str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["fabric_energy_j"]),
            float(row["latency_s"]),
            str(row.get("schedule_signature", "")),
        ),
    )
    front = []
    best_latency_s = float("inf")
    for row in ordered:
        latency_s = float(row["latency_s"])
        if latency_s < best_latency_s:
            front.append(row)
            best_latency_s = latency_s
    return front


def pareto_tile_core_resources(
    rows: list[dict[str, float | int | str]]
) -> list[dict[str, float | int | str]]:
    """Keep the Tile energy-latency front after all psums are resolved locally."""

    objectives = ("fabric_energy_j", "latency_s")
    ordered = sorted(
        rows,
        key=lambda row: (
            *(float(row[key]) for key in objectives),
            str(row.get("schedule_signature", "")),
        ),
    )
    front = []
    for row in ordered:
        values = tuple(float(row[key]) for key in objectives)
        equivalent = any(
            values == tuple(float(existing[key]) for key in objectives)
            for existing in front
        )
        dominated = any(
            all(float(existing[key]) <= value for key, value in zip(objectives, values))
            and any(float(existing[key]) < value for key, value in zip(objectives, values))
            for existing in front
        )
        if not equivalent and not dominated:
            front.append(row)
    return front

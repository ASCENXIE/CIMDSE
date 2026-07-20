"""Single-Tile Macro-composition enumeration and weight-stationary scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import ceil, log2, prod

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
    latency_s: float
    weight_load_latency_s: float


@dataclass(frozen=True)
class ScheduleState:
    energy_j: float
    latency_s: float
    lanes: tuple[tuple[tuple[int, ...], int], ...]


def layer_shapes(mappings, names: set[str] | None = None) -> list[LayerShape]:
    shapes = []
    for einsum_name in mappings.einsum_names:
        if names is not None and einsum_name not in names:
            continue
        spec = mappings.evaluated_specs[einsum_name]
        einsum = spec.workload.einsums[einsum_name]
        sizes = rank_variable_sizes(einsum, spec.workload)
        input_tensor = tensor_for_role(einsum, "input")
        weight_tensor = tensor_for_role(einsum, "weight")
        output_tensor = tensor_for_role(einsum, "output")
        input_vars = set(einsum.tensor2rank_variables[input_tensor.name])
        weight_vars = set(einsum.tensor2rank_variables[weight_tensor.name])
        output_vars = set(einsum.tensor2rank_variables[output_tensor.name])
        shapes.append(
            LayerShape(
                name=einsum_name,
                reduction_length=prod(
                    sizes[var] for var in input_vars & weight_vars - output_vars
                ),
                output_channels=prod(
                    sizes[var] for var in output_vars & weight_vars - input_vars
                ),
                output_positions=prod(
                    sizes[var]
                    for var in output_vars - (output_vars & weight_vars - input_vars)
                ),
            )
        )
    return shapes


def enumerate_tile_designs(
    library: MacroLibrary,
    capacity_tolerance: float,
    area_tolerance: float,
) -> tuple[TileDesign, list[TileDesign]]:
    macro_types = tuple(sorted(library.macro_types.values(), key=lambda macro: macro.rows))
    m64 = library.macro_types["m64x8"]
    m64_register_area_mm2 = _macro_register_area_mm2(
        m64, library.tech_node, library.output_register_entry_bits
    )
    reference = TileDesign(
        counts=(("m64x8", 9),),
        capacity_values=9 * m64.rows * m64.columns,
        macro_area_mm2=9 * m64.area_mm2,
        register_area_mm2=9 * m64_register_area_mm2,
    )
    min_capacity = reference.capacity_values * (1 - capacity_tolerance)
    max_capacity = reference.capacity_values * (1 + capacity_tolerance)
    min_area = reference.area_mm2 * (1 - area_tolerance)
    max_area = reference.area_mm2 * (1 + area_tolerance)
    candidates: list[TileDesign] = []

    def visit(
        index: int,
        counts: list[int],
        capacity: int,
        macro_area: float,
        register_area: float,
    ) -> None:
        if capacity > max_capacity or macro_area + register_area > max_area:
            return
        if index == len(macro_types):
            if (
                capacity >= min_capacity
                and macro_area + register_area >= min_area
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
                        register_area_mm2=register_area,
                    )
                )
            return
        macro = macro_types[index]
        macro_register_area = _macro_register_area_mm2(
            macro, library.tech_node, library.output_register_entry_bits
        )
        max_count = int((max_capacity - capacity) // (macro.rows * macro.columns))
        for count in range(max_count + 1):
            visit(
                index + 1,
                counts + [count],
                capacity + count * macro.rows * macro.columns,
                macro_area + count * macro.area_mm2,
                register_area + count * macro_register_area,
            )

    visit(0, [], 0, 0.0, 0.0)
    candidates.sort(key=lambda design: (design.area_mm2, design.capacity_values, design.name))
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


def _macro_register_area_mm2(
    macro: MacroType,
    tech_node: float,
    output_register_entry_bits: int,
) -> float:
    return (
        2 * macro.rows * _register_area_mm2(tech_node, macro.input_bits)
        + macro.columns
        * _register_area_mm2(tech_node, output_register_entry_bits)
    )


def _macro_register_energy_per_invocation_j(
    macro: MacroType,
    tech_node: float,
    output_register_entry_bits: int,
) -> tuple[float, float]:
    """Return input/output register dynamic energy for one physical VMM.

    One active ping-pong bank is written and read per invocation. The full physical
    input depth is charged, including unused rows. Each Macro output is written to and
    read from its single local partial-sum register bank once.
    """

    input_read, input_write = _register_access_energy_j(tech_node, macro.input_bits)
    output_read, output_write = _register_access_energy_j(
        tech_node, output_register_entry_bits
    )
    return (
        macro.rows * (input_read + input_write),
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
) -> LaneProfile:
    """Return one output block's active Macro invocations and VMM makespan."""

    tasks = []
    task_rows = []
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
        latency_s += max(_vmm_latency_s(macro) for macro in active)
        weight_load_latency_s += max(
            rows / weight_load_rows_per_cycle * macro.cycle_period_s
            for macro, rows in zip(active, active_rows)
        )
    return LaneProfile(
        signature=signature,
        tasks=tuple(tasks),
        task_rows=tuple(task_rows),
        latency_s=latency_s,
        weight_load_latency_s=weight_load_latency_s,
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
            strict_subsets = product(*(range(value + 1) for value in used))
            dominating_states = [
                state
                for subset in strict_subsets
                if subset != used
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


def _nondominated_groups(
    groups: tuple[tuple[int, ...], ...],
    macro_types: tuple[MacroType, ...],
    reduction_length: int,
    weight_load_rows_per_cycle: int,
) -> tuple[tuple[int, ...], ...]:
    profiles = {
        group: _group_profile(
            group, macro_types, reduction_length, weight_load_rows_per_cycle
        )
        for group in groups
    }
    energies = {
        group: sum(macro.energy_per_full_vmm_j for macro in profile.tasks)
        for group, profile in profiles.items()
    }
    return tuple(
        group
        for group in groups
        if not any(
            other != group
            and all(left <= right for left, right in zip(other, group))
            and energies[other] <= energies[group]
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
    tech_node: float,
    output_register_entry_bits: int,
    max_output_lanes: int,
    weight_load_rows_per_cycle: int,
    latency_cap_ratio: float,
    local_epsilon: float,
) -> tuple[list[ScheduleState], float]:
    """Solve Macro grouping and N-block assignment with exact Pareto DP."""

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
        )
        for signature in groups
    }
    feasible_latency_s = min(
        ceil(
            layer.output_blocks
            / min(
                max_output_lanes,
                min(
                    (
                        available // required
                        for available, required in zip(inventory, signature)
                        if required
                    ),
                    default=1,
                ),
            )
        )
        * (
            layer.output_positions * profiles[signature].latency_s
            + profiles[signature].weight_load_latency_s
        )
        for signature in groups
    )
    latency_cap_s = feasible_latency_s * latency_cap_ratio
    for signature in groups:
        profile = profiles[signature]
        macro_energy_per_job = sum(task.energy_per_full_vmm_j for task in profile.tasks)
        register_energy_per_job = sum(
            sum(
                _macro_register_energy_per_invocation_j(
                    task, tech_node, output_register_entry_bits
                )
            )
            for task in profile.tasks
        )
        psum_vectors_per_job = len(profile.tasks) - 1
        energy_per_job = layer.output_positions * (
            macro_energy_per_job
            + register_energy_per_job
            + psum_vectors_per_job * psum_vector_energy_j
        )
        latency_per_job = (
            layer.output_positions * profile.latency_s
            + profile.weight_load_latency_s
        )
        updated = {key: list(value) for key, value in states.items()}
        for (used, completed_jobs), current_states in states.items():
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
                jobs_per_lane_cap = int(latency_cap_s // latency_per_job)
                max_assigned_jobs = min(
                    layer.output_blocks - completed_jobs,
                    copies * jobs_per_lane_cap,
                )
                for assigned_jobs in range(copies, max_assigned_jobs + 1):
                    jobs_per_lane, extra = divmod(assigned_jobs, copies)
                    lane_jobs = [jobs_per_lane + 1] * extra + [jobs_per_lane] * (
                        copies - extra
                    )
                    key = (next_used, completed_jobs + assigned_jobs)
                    candidates = updated.setdefault(key, [])
                    for state in current_states:
                        if len(state.lanes) + copies > max_output_lanes:
                            continue
                        candidates.append(
                            ScheduleState(
                                energy_j=state.energy_j + assigned_jobs * energy_per_job,
                                latency_s=max(
                                    state.latency_s, max(lane_jobs) * latency_per_job
                                ),
                                lanes=state.lanes
                                + tuple((signature, jobs) for jobs in lane_jobs),
                            )
                        )
        states = _resource_front(updated, local_epsilon)
    completed = [
        state
        for (_, completed_jobs), candidates in states.items()
        if completed_jobs == layer.output_blocks
        for state in candidates
    ]
    return _state_front(completed, local_epsilon), latency_cap_s


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
    tech_node: float,
    output_register_entry_bits: int,
    max_macro_types_per_tile: int,
    tile_output_lanes: int,
    weight_load_rows_per_cycle: int,
    latency_cap_s: float,
    latency_cap_ratio: float,
    local_epsilon: float,
    reference: TileDesign,
) -> dict[str, float | int | str]:
    profiles = tuple(
        _group_profile(
            group,
            macro_types,
            layer.reduction_length,
            weight_load_rows_per_cycle,
        )
        for group, _ in state.lanes
    )
    jobs_per_lane = [jobs for _, jobs in state.lanes]
    invocations_by_type = {macro.name: 0 for macro in macro_types if macro in macros}
    psum_vectors = 0
    for jobs, profile in zip(jobs_per_lane, profiles):
        psum_vectors += jobs * layer.output_positions * (len(profile.tasks) - 1)
        for macro in macro_types:
            if macro.name in invocations_by_type:
                invocations_by_type[macro.name] += (
                    jobs * layer.output_positions * profile.tasks.count(macro)
                )
    macro_energy_j = sum(
        invocations_by_type[macro.name] * macro.energy_per_full_vmm_j
        for macro in macro_types
        if macro.name in invocations_by_type
    )
    input_register_read_entries = sum(
        invocations_by_type[macro.name] * macro.rows
        for macro in macro_types
        if macro.name in invocations_by_type
    )
    output_register_read_entries = sum(
        invocations_by_type[macro.name] * macro.columns
        for macro in macro_types
        if macro.name in invocations_by_type
    )
    input_register_energy_j = sum(
        invocations_by_type[macro.name]
        * _macro_register_energy_per_invocation_j(
            macro, tech_node, output_register_entry_bits
        )[0]
        for macro in macro_types
        if macro.name in invocations_by_type
    )
    output_register_energy_j = sum(
        invocations_by_type[macro.name]
        * _macro_register_energy_per_invocation_j(
            macro, tech_node, output_register_entry_bits
        )[1]
        for macro in macro_types
        if macro.name in invocations_by_type
    )
    register_energy_j = input_register_energy_j + output_register_energy_j
    psum_energy_j = psum_vectors * psum_vector_energy_j
    fabric_energy_j = macro_energy_j + register_energy_j + psum_energy_j
    lane_compute_latencies_s = [
        jobs * layer.output_positions * profile.latency_s
        for jobs, profile in zip(jobs_per_lane, profiles)
    ]
    lane_weight_load_latencies_s = [
        jobs * profile.weight_load_latency_s
        for jobs, profile in zip(jobs_per_lane, profiles)
    ]
    compute_only_latency_s = max(lane_compute_latencies_s)
    weight_load_makespan_s = max(lane_weight_load_latencies_s)
    weight_load_added_latency_s = state.latency_s - compute_only_latency_s
    weight_load_rows = sum(
        jobs * sum(profile.task_rows)
        for jobs, profile in zip(jobs_per_lane, profiles)
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
        "max_macro_types_per_tile": max_macro_types_per_tile,
        "tile_output_lanes": tile_output_lanes,
        "macro_row_span": max(row_sizes) / min(row_sizes),
        "lane_count": len(state.lanes),
        "active_lane_count": len(state.lanes),
        "k_partitions_by_lane": "|".join(
            "+".join(macro.name for macro in profile.tasks) for profile in profiles
        ),
        "macro_invocations": sum(invocations_by_type.values()),
        "macro_row_utilization": macro_row_utilization,
        "macro_time_utilization": macro_time_utilization,
        "effective_macro_utilization": macro_row_utilization * macro_time_utilization,
        "psum_vectors": psum_vectors,
        "psum_width_bits": psum_width_bits,
        "psum_vector_energy_j": psum_vector_energy_j,
        "macro_energy_j": macro_energy_j,
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
        "latency_s": state.latency_s,
        "latency_cap_s": latency_cap_s,
        "latency_cap_ratio": latency_cap_ratio,
        "local_pareto_epsilon": local_epsilon,
        "weight_values": layer.weight_values,
        "weight_load_values": layer.weight_values,
        "weight_load_rows": weight_load_rows,
        "weight_load_events": sum(
            jobs * len(profile.tasks) for jobs, profile in zip(jobs_per_lane, profiles)
        ),
        "weight_load_rows_per_cycle_per_macro": weight_load_rows_per_cycle,
        "weight_load_energy_j": 0.0,
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
    latency_cap_ratio: float = 2.0,
    local_epsilon: float = 0.001,
) -> list[dict[str, float | int | str]]:
    """Enumerate local weight-stationary schedules and retain their E-T front."""

    macro_types = tuple(sorted(library.macro_types.values(), key=lambda macro: macro.rows))
    macros = design.macros(library)
    inventory = _inventory(design, macro_types)
    groups = _nondominated_groups(
        _group_signatures(inventory, macro_types, layer.reduction_length),
        macro_types,
        layer.reduction_length,
        library.weight_load_rows_per_cycle,
    )
    psum_width_bits, psum_vector_energy_j = _psum_vector_energy_j(
        layer, macros, tech_node
    )
    states, latency_cap_s = _schedule_front(
        layer,
        inventory,
        macro_types,
        groups,
        psum_vector_energy_j,
        tech_node,
        library.output_register_entry_bits,
        library.tile_output_lanes,
        library.weight_load_rows_per_cycle,
        latency_cap_ratio,
        local_epsilon,
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
            tech_node,
            library.output_register_entry_bits,
            library.max_macro_types_per_tile,
            library.tile_output_lanes,
            library.weight_load_rows_per_cycle,
            latency_cap_s,
            latency_cap_ratio,
            local_epsilon,
            reference,
        )
        for state in states
    ]
    return pareto_energy_latency(rows)


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

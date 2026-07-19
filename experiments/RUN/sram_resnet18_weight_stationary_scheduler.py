"""Weight-stationary, four-Tile scheduler for the ResNet-18 SRAM-CIM study."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, prod

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
class MacroTask:
    output_block: int
    macro: MacroType


@dataclass
class MacroSlot:
    tile: int
    macro: MacroType
    load_s: float = 0.0


def _layer_shape(spec, einsum_name: str) -> LayerShape:
    einsum = spec.workload.einsums[einsum_name]
    sizes = rank_variable_sizes(einsum, spec.workload)
    input_tensor = tensor_for_role(einsum, "input")
    weight_tensor = tensor_for_role(einsum, "weight")
    output_tensor = tensor_for_role(einsum, "output")
    input_vars = set(einsum.tensor2rank_variables[input_tensor.name])
    weight_vars = set(einsum.tensor2rank_variables[weight_tensor.name])
    output_vars = set(einsum.tensor2rank_variables[output_tensor.name])
    return LayerShape(
        name=einsum_name,
        reduction_length=prod(sizes[var] for var in input_vars & weight_vars - output_vars),
        output_channels=prod(sizes[var] for var in output_vars & weight_vars - input_vars),
        output_positions=prod(
            sizes[var] for var in output_vars - (output_vars & weight_vars - input_vars)
        ),
    )


def _vmm_latency_s(macro: MacroType) -> float:
    return macro.bit_cycles_per_full_vmm * macro.cycle_period_s


def _nine_m64_tasks(layer: LayerShape, m64: MacroType) -> list[MacroTask]:
    chunks = ceil(layer.reduction_length / m64.rows)
    return [
        MacroTask(output_block=output_block, macro=m64)
        for output_block in range(layer.output_blocks)
        for _ in range(chunks)
    ]


def _mixed_k_groups(reduction_length: int, m64: MacroType, m512: MacroType) -> list[tuple[MacroType, ...]]:
    if reduction_length <= m64.rows:
        return [(m64,)]
    if reduction_length <= m512.rows:
        return [(m512,)]

    groups: list[tuple[MacroType, ...]] = []
    remaining = reduction_length
    while remaining > 0:
        group_rows = min(remaining, m64.rows + m512.rows)
        if group_rows <= m64.rows:
            groups.append((m64,))
        elif group_rows <= m512.rows:
            groups.append((m512,))
        else:
            groups.append((m512, m64))
        remaining -= group_rows
    return groups


def _mixed_tasks(layer: LayerShape, m64: MacroType, m512: MacroType) -> list[MacroTask]:
    groups = _mixed_k_groups(layer.reduction_length, m64, m512)
    return [
        MacroTask(output_block=output_block, macro=macro)
        for output_block in range(layer.output_blocks)
        for group in groups
        for macro in group
    ]


def _slots(n_tiles: int, macros: tuple[MacroType, ...]) -> list[MacroSlot]:
    return [MacroSlot(tile=tile, macro=macro) for tile in range(n_tiles) for macro in macros]


def _assign_tasks(
    tasks: list[MacroTask], slots: list[MacroSlot]
) -> tuple[float, set[int], dict[int, set[int]]]:
    """Assign resident weight blocks while balancing Macro time and keeping K blocks local."""

    slots_by_tile: dict[int, list[MacroSlot]] = {}
    for slot in slots:
        slots_by_tile.setdefault(slot.tile, []).append(slot)
    tasks_by_output: dict[int, list[MacroTask]] = {}
    for task in tasks:
        tasks_by_output.setdefault(task.output_block, []).append(task)
    output_block_tiles: dict[int, set[int]] = {}
    for output_block, output_tasks in tasks_by_output.items():
        pending = list(output_tasks)
        tiles_for_block = output_block_tiles.setdefault(output_block, set())
        while pending:
            tile, tile_slots = min(
                slots_by_tile.items(),
                key=lambda item: (sum(slot.load_s for slot in item[1]), item[0]),
            )
            capacity_by_type: dict[str, int] = {}
            for slot in tile_slots:
                capacity_by_type[slot.macro.name] = capacity_by_type.get(slot.macro.name, 0) + 1
            selected: list[MacroTask] = []
            for macro_name, capacity in capacity_by_type.items():
                matching = [task for task in pending if task.macro.name == macro_name][:capacity]
                selected.extend(matching)
            if not selected:
                raise ValueError("No physical slot for a required macro type.")
            for task in selected:
                pending.remove(task)
                slot = min(
                    (slot for slot in tile_slots if slot.macro.name == task.macro.name),
                    key=lambda candidate: candidate.load_s,
                )
                slot.load_s += _vmm_latency_s(task.macro)
            tiles_for_block.add(tile)
    return (
        max((slot.load_s for slot in slots), default=0.0),
        {slot.tile for slot in slots if slot.load_s > 0},
        output_block_tiles,
    )


def _policy(design: str, tasks: list[MacroTask]) -> str:
    if design == "nine_m64x8":
        return "m64_k_partitioned"
    names = {task.macro.name for task in tasks}
    if names == {"m64x8"}:
        return "m64_only" if len(tasks) else "empty"
    if names == {"m512x8"}:
        return "m512_only"
    return "m512_m64_k_partitioned"


def _row(
    layer: LayerShape,
    design: str,
    tasks: list[MacroTask],
    tile_macros: tuple[MacroType, ...],
    n_tiles: int,
    psum_vector_energy_j: float,
) -> dict[str, float | int | str]:
    tile_weight_capacity_values = sum(macro.rows * macro.columns for macro in tile_macros)
    replicates_weights = layer.weight_values <= tile_weight_capacity_values
    assignment_tiles = 1 if replicates_weights else n_tiles
    max_slot_time_s, active_tiles, output_block_tiles = _assign_tasks(
        tasks, _slots(assignment_tiles, tile_macros)
    )
    if replicates_weights:
        latency_s = ceil(layer.output_positions / n_tiles) * max_slot_time_s
        active_tiles = set(range(n_tiles))
        inter_tile_psum_vectors = 0
        weight_mapping = "replicated_spatial"
        weight_replication_factor = n_tiles
    else:
        latency_s = layer.output_positions * max_slot_time_s
        inter_tile_psum_vectors = layer.output_positions * sum(
            max(len(tiles) - 1, 0) for tiles in output_block_tiles.values()
        )
        weight_mapping = "partitioned_weight_stationary"
        weight_replication_factor = 1

    tasks_per_output = {output_block: 0 for output_block in range(layer.output_blocks)}
    for task in tasks:
        tasks_per_output[task.output_block] += 1
    psum_vectors = layer.output_positions * sum(
        max(task_count - 1, 0) for task_count in tasks_per_output.values()
    )
    local_psum_vectors = psum_vectors - inter_tile_psum_vectors
    macro_invocations = len(tasks) * layer.output_positions
    m64_invocations = sum(task.macro.name == "m64x8" for task in tasks) * layer.output_positions
    m512_invocations = sum(task.macro.name == "m512x8" for task in tasks) * layer.output_positions
    macro_energy_j = sum(
        layer.output_positions * task.macro.energy_per_full_vmm_j for task in tasks
    )
    macro_capacity_macs = sum(
        layer.output_positions * task.macro.rows * task.macro.columns for task in tasks
    )
    psum_energy_j = psum_vectors * psum_vector_energy_j
    fabric_energy_j = macro_energy_j + psum_energy_j
    return {
        "layer": layer.name,
        "design": design,
        "policy": _policy(design, tasks),
        "weight_mapping": weight_mapping,
        "weight_values": layer.weight_values,
        "tile_weight_capacity_values": tile_weight_capacity_values,
        "weight_replication_factor": weight_replication_factor,
        "active_tiles": len(active_tiles),
        "K": layer.reduction_length,
        "N": layer.output_channels,
        "output_positions": layer.output_positions,
        "MACs": layer.macs,
        "n_tiles": n_tiles,
        "tile_macro_area_mm2": sum(macro.area_mm2 for macro in tile_macros),
        "macro_tasks": len(tasks),
        "latency_model": "macro_vmm_only_psum_latency_ignored",
        "macro_invocations": macro_invocations,
        "m64_invocations": m64_invocations,
        "m512_invocations": m512_invocations,
        "macro_row_utilization": layer.macs / macro_capacity_macs,
        "psum_vectors": psum_vectors,
        "local_psum_vectors": local_psum_vectors,
        "inter_tile_psum_vectors": inter_tile_psum_vectors,
        "psum_vector_energy_j": psum_vector_energy_j,
        "macro_energy_j": macro_energy_j,
        "psum_energy_j": psum_energy_j,
        "fabric_energy_j": fabric_energy_j,
        "latency_s": latency_s,
        "TOPS_per_W": 2 * layer.macs / fabric_energy_j / 1e12,
    }


def evaluate_resnet18_weight_stationary_scheduler(
    mappings, library: MacroLibrary
) -> list[dict[str, float | int | str]]:
    """Evaluate all layers with four Tiles holding unique weights whenever required."""

    first_spec = mappings.evaluated_specs[mappings.einsum_names[0]]
    n_tiles = int(first_spec.arch.variables.n_tiles)
    tech_node = float(first_spec.variables.tech_node)
    accumulator = DigitalSramShiftAccumulator(tech_node=tech_node, width=25).read()
    psum_vector_energy_j = 8 * float(accumulator.energy)
    m64 = library.macro_types["m64x8"]
    m512 = library.macro_types["m512x8"]
    rows = []
    for einsum_name in mappings.einsum_names:
        layer = _layer_shape(mappings.evaluated_specs[einsum_name], einsum_name)
        rows.append(
            _row(
                layer,
                "nine_m64x8",
                _nine_m64_tasks(layer, m64),
                (m64,) * 9,
                n_tiles,
                psum_vector_energy_j,
            )
        )
        rows.append(
            _row(
                layer,
                "m64x8_m512x8",
                _mixed_tasks(layer, m64, m512),
                (m64, m512),
                n_tiles,
                psum_vector_energy_j,
            )
        )
    return rows

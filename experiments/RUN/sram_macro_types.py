"""YAML-backed SRAM-CIM macro library for system-level design exploration."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import accelforge as af
from hwcomponents_library.library.misc import SmartBufferSRAM

from sram_helper_functions import lookup_cim_macro_efficiency


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "experiments" / "SRAM_config"
ENERGY_MODEL_VERSION = "partial_row_measured_slope_all_macros_macro_static_tile_psum_register_weight_bus_v5"


@dataclass(frozen=True)
class MacroType:
    """One characterized SRAM-CIM macro shape at a fixed precision and technology."""

    name: str
    rows: int
    columns: int
    input_bits: int
    weight_bits: int
    input_bits_per_cycle: int
    cycle_period_s: float
    energy_efficiency_tops_per_w: float
    area_efficiency_tops_per_mm2: float
    weight_row_write_energy_j: float
    static_power_fraction: float
    partial_vmm_fixed_energy_j: float
    partial_vmm_energy_per_active_row_j: float

    @property
    def bit_cycles_per_full_vmm(self) -> int:
        return ceil(self.input_bits / self.input_bits_per_cycle)

    @property
    def ops_per_full_vmm(self) -> int:
        return 2 * self.rows * self.columns

    @property
    def peak_tops(self) -> float:
        return (
            self.ops_per_full_vmm
            / self.bit_cycles_per_full_vmm
            / self.cycle_period_s
            / 1e12
        )

    @property
    def energy_per_full_vmm_j(self) -> float:
        return self.ops_per_full_vmm / self.energy_efficiency_tops_per_w / 1e12

    @property
    def dynamic_energy_per_full_vmm_j(self) -> float:
        return self.energy_per_full_vmm_j * (1 - self.static_power_fraction)

    @property
    def static_power_w(self) -> float:
        return (
            self.energy_per_full_vmm_j
            * self.static_power_fraction
            / (self.bit_cycles_per_full_vmm * self.cycle_period_s)
        )

    def dynamic_energy_for_active_rows_j(self, active_rows: int) -> float:
        """Return digital VMM dynamic energy for the rows active this invocation."""

        if not 0 <= active_rows <= self.rows:
            raise ValueError(
                f"active_rows must be in [0, {self.rows}] for {self.name}, "
                f"got {active_rows}."
            )
        if active_rows == 0:
            return 0.0
        if active_rows == self.rows:
            return self.dynamic_energy_per_full_vmm_j
        partial_energy = (
            self.partial_vmm_fixed_energy_j
            + self.partial_vmm_energy_per_active_row_j * active_rows
        )
        return min(partial_energy, self.dynamic_energy_per_full_vmm_j)

    @property
    def energy_per_bit_cycle_j(self) -> float:
        return self.energy_per_full_vmm_j / self.bit_cycles_per_full_vmm

    @property
    def area_mm2(self) -> float:
        return self.peak_tops / self.area_efficiency_tops_per_mm2


@dataclass(frozen=True)
class MacroLibrary:
    """Candidate macro types plus the physical inventory of one Tile."""

    macro_types: dict[str, MacroType]
    tile_inventory: dict[str, int]
    output_register_entry_bits: int
    max_macro_types_per_tile: int
    tile_output_lanes: int
    tech_node: float
    weight_load_rows_per_cycle: int
    macro_static_power_fraction: float
    tile_shared_bus_width_bits: int
    tile_weight_bus_width_bits: int
    tile_bus_energy_per_bit_j: float
    tile_retain_features_across_nblocks: bool
    tile_allow_intra_bank_weight_compute_overlap: bool
    vit_allow_cross_bank_weight_compute_overlap: bool
    tile_latency_batch_granularity: str
    tile_include_pipeline_fill_drain: bool

    def homogeneous_mapping_macro(self, variables) -> MacroType:
        """Return the active type when the AccelForge endpoint is homogeneous.

        The current mapper exposes one logical macro shape. Heterogeneous inventories
        are intentionally rejected here until a macro-assignment scheduler is added.
        """

        active = {
            name: count for name, count in self.tile_inventory.items() if count > 0
        }
        if len(active) != 1:
            raise NotImplementedError(
                "The current AccelForge endpoint has one logical macro shape. "
                "Use exactly one active macro type until heterogeneous macro "
                "assignment is enabled."
            )

        name, count = next(iter(active.items()))
        macro = self.macro_types[name]
        if count != int(variables.macros_per_tile):
            raise ValueError(
                f"Tile inventory has {count} {name} macros, but macros_per_tile is "
                f"{int(variables.macros_per_tile)}."
            )
        if macro.rows != int(variables.logical_array_rows) or macro.columns != int(
            variables.logical_array_columns
        ):
            raise ValueError(
                f"Tile inventory selects {name} ({macro.rows}x{macro.columns}), "
                "but the mapper endpoint is "
                f"{int(variables.logical_array_rows)}x"
                f"{int(variables.logical_array_columns)}."
            )
        return macro


def load_sram_macro_library(config_name: str) -> MacroLibrary:
    """Load and characterize macro_types and tile_macro_inventory from one YAML."""

    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"SRAM-CIM configuration not found: {config_path}")

    variables = af.Variables.from_yaml(config_path, top_key="variables").model_dump()
    arch_variables = af.Arch.from_yaml(config_path, top_key="arch").variables.model_dump()
    raw_types = af.Variables.from_yaml(config_path, top_key="macro_types").model_dump()
    raw_inventory = af.Variables.from_yaml(
        config_path, top_key="tile_macro_inventory"
    ).model_dump()

    input_bits = int(variables["supported_input_bits"])
    weight_bits = int(variables["supported_weight_bits"])
    input_bits_per_cycle = int(arch_variables["input_bits_per_cycle"])
    cycle_period_s = float(arch_variables["macro_cycle_period"])
    static_power_fraction = float(arch_variables["macro_static_power_fraction"])
    if not 0 <= static_power_fraction < 1:
        raise ValueError("macro_static_power_fraction must be in [0, 1).")
    macro_types = {}
    for name, raw_type in raw_types.items():
        rows = int(raw_type["rows"])
        columns = int(raw_type["columns"])
        partial_vmm_fixed_energy_j = float(
            raw_type["macro_partial_vmm_fixed_energy"]
        )
        partial_vmm_energy_per_active_row_j = float(
            raw_type["macro_partial_vmm_energy_per_active_row"]
        )
        if partial_vmm_fixed_energy_j < 0 or partial_vmm_energy_per_active_row_j < 0:
            raise ValueError(
                f"Partial-row VMM energy coefficients cannot be negative for {name}."
            )
        weight_row_width_bits = columns * weight_bits
        weight_sram = SmartBufferSRAM(
            tech_node=float(variables["tech_node"]),
            width=weight_row_width_bits,
            size=rows * weight_row_width_bits,
            n_rw_ports=1,
            n_banks=1,
        )
        macro_types[name] = MacroType(
            name=name,
            rows=rows,
            columns=columns,
            input_bits=input_bits,
            weight_bits=weight_bits,
            input_bits_per_cycle=input_bits_per_cycle,
            cycle_period_s=cycle_period_s,
            energy_efficiency_tops_per_w=lookup_cim_macro_efficiency(
                rows, columns, weight_bits, input_bits, "energy_efficiency"
            ),
            area_efficiency_tops_per_mm2=lookup_cim_macro_efficiency(
                rows, columns, weight_bits, input_bits, "area_efficiency"
            ),
            weight_row_write_energy_j=float(weight_sram.write().energy),
            static_power_fraction=static_power_fraction,
            partial_vmm_fixed_energy_j=partial_vmm_fixed_energy_j,
            partial_vmm_energy_per_active_row_j=(
                partial_vmm_energy_per_active_row_j
            ),
        )

    tile_inventory = {name: int(count) for name, count in raw_inventory.items()}
    unknown = set(tile_inventory) - set(macro_types)
    if unknown:
        raise ValueError(f"Tile inventory references unknown macro types: {sorted(unknown)}")
    if not any(count > 0 for count in tile_inventory.values()):
        raise ValueError("Tile macro inventory must contain at least one active macro.")
    output_register_entry_bits = int(arch_variables["macro_output_register_entry_bits"])
    max_macro_types_per_tile = int(arch_variables["max_macro_types_per_tile"])
    tile_output_lanes = int(arch_variables["tile_output_lanes"])
    tech_node = float(variables["tech_node"])
    weight_load_rows_per_cycle = int(
        arch_variables["macro_weight_load_rows_per_cycle"]
    )
    tile_shared_bus_width_bits = int(
        arch_variables["tile_shared_bus_width_bits"]
    )
    tile_weight_bus_width_bits = int(
        arch_variables["tile_weight_bus_width_bits"]
    )
    tile_bus_energy_per_bit_j = float(
        arch_variables["tile_bus_energy_per_bit"]
    )
    tile_retain_features_across_nblocks = bool(
        arch_variables["tile_retain_features_across_nblocks"]
    )
    tile_allow_intra_bank_weight_compute_overlap = bool(
        arch_variables["tile_allow_intra_bank_weight_compute_overlap"]
    )
    vit_allow_cross_bank_weight_compute_overlap = bool(
        arch_variables["vit_allow_cross_bank_weight_compute_overlap"]
    )
    tile_latency_batch_granularity = str(
        arch_variables["tile_latency_batch_granularity"]
    )
    tile_include_pipeline_fill_drain = bool(
        arch_variables["tile_include_pipeline_fill_drain"]
    )
    if tile_shared_bus_width_bits < 1:
        raise ValueError("tile_shared_bus_width_bits must be positive.")
    if tile_weight_bus_width_bits < 1:
        raise ValueError("tile_weight_bus_width_bits must be positive.")
    if tile_bus_energy_per_bit_j < 0:
        raise ValueError("tile_bus_energy_per_bit cannot be negative.")
    if tile_retain_features_across_nblocks:
        raise ValueError(
            "Tile DSE does not support retaining complete features across N-blocks."
        )
    if tile_allow_intra_bank_weight_compute_overlap:
        raise ValueError("Tile DSE requires exclusive weight load and compute per bank.")
    if not vit_allow_cross_bank_weight_compute_overlap:
        raise ValueError("ViT Tile DSE requires cross-bank weight/compute overlap.")
    if tile_latency_batch_granularity != "nblock_kwave":
        raise ValueError(
            "tile_latency_batch_granularity must be 'nblock_kwave'."
        )
    if tile_include_pipeline_fill_drain:
        raise ValueError("Tile DSE currently excludes pipeline fill/drain latency.")
    return MacroLibrary(
        macro_types=macro_types,
        tile_inventory=tile_inventory,
        output_register_entry_bits=output_register_entry_bits,
        max_macro_types_per_tile=max_macro_types_per_tile,
        tile_output_lanes=tile_output_lanes,
        tech_node=tech_node,
        weight_load_rows_per_cycle=weight_load_rows_per_cycle,
        macro_static_power_fraction=static_power_fraction,
        tile_shared_bus_width_bits=tile_shared_bus_width_bits,
        tile_weight_bus_width_bits=tile_weight_bus_width_bits,
        tile_bus_energy_per_bit_j=tile_bus_energy_per_bit_j,
        tile_retain_features_across_nblocks=tile_retain_features_across_nblocks,
        tile_allow_intra_bank_weight_compute_overlap=(
            tile_allow_intra_bank_weight_compute_overlap
        ),
        vit_allow_cross_bank_weight_compute_overlap=(
            vit_allow_cross_bank_weight_compute_overlap
        ),
        tile_latency_batch_granularity=tile_latency_batch_granularity,
        tile_include_pipeline_fill_drain=tile_include_pipeline_fill_drain,
    )

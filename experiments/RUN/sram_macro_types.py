"""YAML-backed SRAM-CIM macro library for system-level design exploration."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import accelforge as af

from sram_helper_functions import lookup_cim_macro_efficiency


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "experiments" / "SRAM_config"


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
    macro_types = {}
    for name, raw_type in raw_types.items():
        rows = int(raw_type["rows"])
        columns = int(raw_type["columns"])
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
        )

    tile_inventory = {name: int(count) for name, count in raw_inventory.items()}
    unknown = set(tile_inventory) - set(macro_types)
    if unknown:
        raise ValueError(f"Tile inventory references unknown macro types: {sorted(unknown)}")
    if not any(count > 0 for count in tile_inventory.values()):
        raise ValueError("Tile macro inventory must contain at least one active macro.")
    return MacroLibrary(macro_types=macro_types, tile_inventory=tile_inventory)

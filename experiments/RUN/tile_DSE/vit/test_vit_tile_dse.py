"""Tests for the isolated ViT projection/attention Tile DSE."""

from __future__ import annotations

from pathlib import Path
import sys
import time
import unittest


RUN_DIR = Path(__file__).resolve().parents[2]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from sram_macro_types import load_sram_macro_library
from tile_DSE.sram_tile_dse import enumerate_tile_designs
from tile_DSE.vit.vision_transformer_workload import (
    vit_b16_macs,
    vit_b16_operator_groups,
)
from tile_DSE.vit.vit_tile_dse import (
    _aggregate_groups,
    enumerate_bank_assignments,
)
from tile_DSE.vit.run_vit_tile_bayesian_dse import (
    DesignEvaluationTimeout,
    _design_timeout,
)


class VitTileDseTest(unittest.TestCase):
    def test_design_wall_clock_timeout_interrupts_evaluation(self) -> None:
        started_at = time.perf_counter()

        with self.assertRaises(DesignEvaluationTimeout):
            with _design_timeout(0.05):
                time.sleep(1.0)

        self.assertLess(time.perf_counter() - started_at, 0.5)

    def test_attention_shapes_keep_head_and_query_axes(self) -> None:
        groups = {group.name: group for group in vit_b16_operator_groups()}

        self.assertEqual(
            groups["patch_embedding"].shape.output_positions,
            192,
        )
        self.assertEqual(
            (
                groups["attention_qk"].shape.reduction_length,
                groups["attention_qk"].shape.output_channels,
                groups["attention_qk"].shape.output_positions,
            ),
            (64, 192, 12 * 192),
        )
        self.assertEqual(
            (
                groups["attention_av"].shape.reduction_length,
                groups["attention_av"].shape.output_channels,
                groups["attention_av"].shape.output_positions,
            ),
            (192, 64, 12 * 192),
        )
        self.assertEqual(groups["attention_qk"].macs, 12 * 192 * 192 * 64 * 12)
        self.assertEqual(vit_b16_macs(), 17_100_945_408)

    def test_64_512_has_both_role_assignments_and_legal_lanes(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        _, designs = enumerate_tile_designs(library, 0.4608, 0.20)
        design = next(item for item in designs if item.name == "1xm64x8+1xm512x8")

        assignments = enumerate_bank_assignments(design, library)

        roles = {(item.projection.name, item.attention.name) for item in assignments}
        self.assertIn(("1xm512x8", "1xm64x8"), roles)
        self.assertIn(("1xm64x8", "1xm512x8"), roles)
        self.assertTrue(
            all(
                item.projection_lanes + item.attention_lanes
                <= library.tile_output_lanes
                for item in assignments
            )
        )
        self.assertEqual(len(assignments), 12)

    def test_single_macro_uses_sequential_fallback(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        _, designs = enumerate_tile_designs(library, 0.4608, 0.20)
        design = next(item for item in designs if item.name == "1xm512x8")

        [assignment] = enumerate_bank_assignments(design, library)

        self.assertFalse(assignment.concurrent)
        self.assertEqual(assignment.projection, design)
        self.assertEqual(assignment.attention, design)

    def test_partition_budget_does_not_drop_type_specialization(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        _, designs = enumerate_tile_designs(library, 0.4608, 0.20)
        design = next(
            item for item in designs if item.name == "2xm64x8+1xm512x8"
        )

        assignments = enumerate_bank_assignments(
            design, library, partition_budget=1
        )
        roles = {(item.projection.name, item.attention.name) for item in assignments}

        self.assertIn(("1xm512x8", "2xm64x8"), roles)
        self.assertIn(("2xm64x8", "1xm512x8"), roles)


    def test_repeated_operator_scales_energy_compute_and_bus(self) -> None:
        group = vit_b16_operator_groups()[1]
        row = {
            "layer": group.name,
            "schedule_signature": "one",
            "fabric_energy_j": 11.0,
            "macro_static_energy_j": 1.0,
            "latency_s": 7.0,
            "bus_added_latency_s": 2.0,
            "total_bus_cycles": 3,
            "MACs": group.shape.macs,
            "active_lane_count": 2,
        }
        [state] = _aggregate_groups((group,), {group.name: [row]}, 16)

        self.assertEqual(state["dynamic_energy_j"], 10.0 * group.repeat)
        self.assertEqual(state["macro_critical_latency_s"], 5.0 * group.repeat)
        self.assertEqual(state["total_bus_cycles"], 3 * group.repeat)


if __name__ == "__main__":
    unittest.main()

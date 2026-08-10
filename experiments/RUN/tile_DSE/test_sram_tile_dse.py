"""Tests for the peak-compute-window Tile candidate space."""

from __future__ import annotations

import unittest
import csv
from pathlib import Path
import sys
import tempfile

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1]
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

import accelforge as af
from sram_macro_types import load_sram_macro_library
from tile_DSE.run_sram_tile_bayesian_dse import (
    evaluate_design,
    fixed_design_names,
    panel_sha256,
    partial_checkpoint_key,
    prediction_rows,
    result_directory,
    select_validation_batch,
    workload_path,
)
from tile_DSE.run_sram_tile_dse import (
    _compress_energy_latency_front,
    aggregate_energy_latency_area_pareto,
)
from tile_DSE.sram_tile_dse import (
    LaneProfile,
    LayerShape,
    ScheduleSearchMetrics,
    ScheduleState,
    _canonical_lane_key,
    _cap_resource_states,
    _extend_exact_state_front,
    _state_front,
    _strict_resource_subsets,
    _schedule_timing,
    enumerate_tile_designs,
    evaluate_tile_schedules,
    layer_shapes_from_workload,
)


class TileCandidateSpaceTest(unittest.TestCase):
    def test_design_shape_checkpoint_resume(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        reference, designs = enumerate_tile_designs(library, 0.4608, 0.20)
        design = next(item for item in designs if item.name == "1xm512x8")
        layers = [
            LayerShape("T0", 64, 16, 1),
            LayerShape("T1", 64, 16, 1),
        ]
        with tempfile.TemporaryDirectory() as directory:
            partial_path = Path(directory) / "design.pkl"
            first = evaluate_design(
                design,
                layers,
                library,
                library.tech_node,
                reference,
                4.0,
                0.001,
                4096,
                partial_path,
            )
            second = evaluate_design(
                design,
                layers,
                library,
                library.tech_node,
                reference,
                4.0,
                0.001,
                4096,
                partial_path,
            )

            self.assertTrue(partial_path.is_file())

        self.assertEqual(first, second)
        self.assertEqual({row["layer"] for row in first[0]}, {"T0", "T1"})
        self.assertNotEqual(
            partial_checkpoint_key("vgg16", 1, 4.0, 0.001, 4096),
            partial_checkpoint_key("resnet50", 1, 4.0, 0.001, 4096),
        )

    def test_vgg16_shapes_without_system_mapping(self) -> None:
        workload = af.Workload.from_yaml(
            workload_path("vgg16"),
            top_key="workload",
            jinja_parse_data={"BATCH_SIZE": 1},
        )

        shapes = layer_shapes_from_workload(workload)

        self.assertEqual(len(shapes), 16)
        self.assertEqual(shapes[0], LayerShape("T0", 27, 64, 224 * 224))
        self.assertEqual(shapes[-1], LayerShape("T15", 4096, 1000, 1))

    def test_cross_workload_fixed_panel_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file, fieldnames=("design", "selection_order")
                )
                writer.writeheader()
                writer.writerows(
                    (
                        {"design": "9xm64x8", "selection_order": 2},
                        {"design": "1xm64x8+1xm512x8", "selection_order": 1},
                    )
                )

            names = fixed_design_names(path)

        self.assertEqual(names, ["1xm64x8+1xm512x8", "9xm64x8"])
        self.assertEqual(panel_sha256(names), panel_sha256(names))
        self.assertNotEqual(panel_sha256(names), panel_sha256(list(reversed(names))))
        self.assertEqual(workload_path("resnet50").name, "resnet50.yaml")
        self.assertEqual(workload_path("vgg16").name, "vgg16.yaml")
        self.assertEqual(result_directory("resnet50").name, "resnet50")
        with self.assertRaises(ValueError):
            result_directory("../outside")

    def test_segmented_weight_bus_timing_with_unbalanced_lanes(self) -> None:
        first_signature = (1, 0)
        second_signature = (0, 2)
        profiles = {
            first_signature: LaneProfile(
                signature=first_signature,
                tasks=(),
                task_rows=(),
                wave_macro_counts=(1,),
                latency_s=5.0,
                weight_load_latency_s=0.0,
                wave_task_rows=((32,),),
            ),
            second_signature: LaneProfile(
                signature=second_signature,
                tasks=(),
                task_rows=(),
                wave_macro_counts=(2, 1),
                latency_s=10.0,
                weight_load_latency_s=0.0,
                wave_task_rows=((64, 32), (16,)),
            ),
        }

        timing = _schedule_timing(
            LayerShape("timing", 64, 16, 1),
            ((first_signature, 2), (second_signature, 5)),
            profiles,
            cycle_period_s=1.0,
            input_bits=8,
            weight_line_bits=64,
            feature_bus_width_bits=512,
            weight_bus_width_bits=64,
            weight_load_rows_per_cycle=1,
        )

        self.assertEqual(timing.weight_bus_cycles, 624)
        self.assertEqual(timing.feature_bus_cycles, 15)
        self.assertEqual(timing.total_bus_cycles, 639)
        self.assertEqual(timing.compute_only_cycles, 50)
        self.assertEqual(timing.macro_critical_cycles, 674)
        self.assertEqual(timing.latency_cycles, 674)

    def test_nblock_wave_timing_reloads_features_and_sums_batch_maxima(self) -> None:
        signature = (1,)
        profile = LaneProfile(
            signature=signature,
            tasks=(),
            task_rows=(),
            wave_macro_counts=(1, 1),
            latency_s=20.0,
            weight_load_latency_s=0.0,
            wave_task_rows=((64,), (512,)),
            wave_latencies_s=(10.0, 10.0),
        )

        timing = _schedule_timing(
            LayerShape("batch_timing", 576, 16, 1),
            ((signature, 2),),
            {signature: profile},
            cycle_period_s=1.0,
            input_bits=8,
            weight_line_bits=64,
            feature_bus_width_bits=64,
            weight_bus_width_bits=64,
            weight_load_rows_per_cycle=1,
        )

        self.assertEqual(timing.weight_bus_cycles, 2 * 576)
        self.assertEqual(timing.feature_bus_bits, 2 * 576 * 8)
        self.assertEqual(timing.feature_bus_cycles, 2 * (8 + 64))
        self.assertEqual(timing.compute_only_cycles, 2 * (10 + 10))
        self.assertEqual(timing.latency_cycles, 2 * (64 + 10 + 512 + 64))
        self.assertGreater(
            timing.latency_cycles,
            timing.weight_bus_cycles
            + max(timing.feature_bus_cycles, timing.compute_only_cycles),
        )

    def test_search_metrics_and_cached_canonical_helpers(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        reference, designs = enumerate_tile_designs(library, 0.4608, 0.20)
        design = next(item for item in designs if item.name == "1xm512x8")
        metrics = ScheduleSearchMetrics()

        rows = evaluate_tile_schedules(
            LayerShape("profile", 256, 16, 1),
            design,
            library,
            library.tech_node,
            reference,
            metrics=metrics,
        )

        self.assertTrue(rows)
        self.assertGreater(metrics.raw_group_count, 0)
        self.assertGreater(metrics.nondominated_group_count, 0)
        self.assertGreater(metrics.candidate_transitions, 0)
        self.assertGreater(metrics.accepted_transitions, 0)
        self.assertGreater(metrics.peak_transient_states, 0)
        self.assertGreater(metrics.timing_cache_misses, 0)
        self.assertGreater(metrics.elapsed_s, 0.0)
        self.assertEqual(
            _canonical_lane_key((((1, 0), 2), ((0, 1), 1))),
            (((0, 1), 1), ((1, 0), 2)),
        )
        self.assertEqual(
            set(_strict_resource_subsets((1, 2))),
            {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)},
        )

    def test_exact_aggregate_front_rejects_insufficient_cap(self) -> None:
        rows = [
            {"fabric_energy_j": float(index), "latency_s": float(4 - index)}
            for index in range(1, 4)
        ]

        with self.assertRaises(RuntimeError):
            _compress_energy_latency_front(rows, state_cap=2, initial_epsilon=0.0)

    def test_predicted_front_validation_and_prediction_output(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        reference, designs = enumerate_tile_designs(library, 0.4608, 0.20)
        subset = designs[:8]
        x = np.asarray(
            [[index / 7, (7 - index) / 7] for index in range(8)], dtype=float
        )
        evaluated = [0, 3, 7]
        summaries = {
            0: (1.0, 8.0, 8.0),
            3: (3.0, 3.0, 9.0),
            7: (8.0, 1.0, 8.0),
        }

        selected = select_validation_batch(x, evaluated, summaries, 2)
        predictions = prediction_rows(subset, x, evaluated, summaries)

        self.assertEqual(len(selected), 2)
        self.assertTrue(set(selected).isdisjoint(evaluated))
        self.assertEqual(len(predictions), len(subset) - len(evaluated))
        self.assertTrue(
            all("energy_log_uncertainty" in row for row in predictions)
        )
        self.assertTrue(
            any(row["predicted_energy_latency_pareto"] for row in predictions)
        )

    def test_energy_latency_area_pareto(self) -> None:
        rows = [
            {
                "design": "small",
                "fabric_energy_mJ": 2.0,
                "latency_ms": 2.0,
                "tile_area_mm2": 1.0,
            },
            {
                "design": "fast",
                "fabric_energy_mJ": 3.0,
                "latency_ms": 1.0,
                "tile_area_mm2": 2.0,
            },
            {
                "design": "dominated",
                "fabric_energy_mJ": 3.0,
                "latency_ms": 3.0,
                "tile_area_mm2": 2.0,
            },
        ]

        front = aggregate_energy_latency_area_pareto(rows)

        self.assertEqual({row["design"] for row in front}, {"small", "fast"})

    def test_output_register_area_is_fixed_per_tile(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        _, designs = enumerate_tile_designs(library, 0.4608, 0.20)
        one_macro = next(design for design in designs if design.name == "1xm512x8")
        many_macros = next(design for design in designs if design.name == "18xm32x8")
        register_area_per_input = (
            one_macro.register_area_mm2
            - 2
            * library.macro_types["m512x8"].rows
            * self._register_area_mm2(library, 8)
        )
        many_macro_output_area = (
            many_macros.register_area_mm2
            - 18
            * 2
            * library.macro_types["m32x8"].rows
            * self._register_area_mm2(library, 8)
        )

        self.assertAlmostEqual(register_area_per_input, many_macro_output_area)
        self.assertAlmostEqual(
            register_area_per_input,
            32 * self._register_area_mm2(library, 25),
        )

    @staticmethod
    def _register_area_mm2(library, width: int) -> float:
        from tile_DSE.sram_tile_dse import _register_area_mm2

        return _register_area_mm2(library.tech_node, width)

    def test_shared_bus_and_fixed_output_register_events(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        reference, designs = enumerate_tile_designs(library, 0.4608, 0.20)
        design = next(item for item in designs if item.name == "1xm512x8")

        [row] = evaluate_tile_schedules(
            LayerShape("partial", 256, 8, 1),
            design,
            library,
            library.tech_node,
            reference,
        )

        self.assertEqual(row["feature_bus_bits"], 256 * 8)
        self.assertEqual(row["weight_bus_bits"], 256 * 64)
        self.assertEqual(row["feature_bus_cycles"], 4)
        self.assertEqual(row["weight_bus_cycles"], 256)
        self.assertEqual(row["total_bus_cycles"], 260)
        self.assertAlmostEqual(row["latency_s"], 264 * 2.5e-9)
        self.assertEqual(row["output_register_read_entries"], 8)
        self.assertEqual(row["output_register_write_entries"], 8)
        self.assertEqual(row["bus_energy_j"], 0.0)

    def test_cross_wave_energy_stays_in_tile(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        reference, designs = enumerate_tile_designs(library, 0.4608, 0.20)
        design = next(item for item in designs if item.name == "1xm512x8")

        [row] = evaluate_tile_schedules(
            LayerShape("two_waves", 768, 8, 1),
            design,
            library,
            library.tech_node,
            reference,
        )

        self.assertEqual(row["max_k_wave_count"], 2)
        self.assertEqual(row["output_register_read_entries"], 16)
        self.assertEqual(row["output_register_write_entries"], 16)
        self.assertEqual(row["cross_wave_psum_vectors"], 1)
        self.assertGreater(row["cross_wave_psum_energy_j"], 0.0)
        self.assertEqual(row["psum_energy_j"], row["cross_wave_psum_energy_j"])
        self.assertEqual(row["core_psum_buffer_vectors"], 0)
        self.assertEqual(row["cross_wave_psum_resolved_in_tile"], 1)

    def test_partial_row_dynamic_energy_model(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        macro = library.macro_types["m512x8"]

        self.assertEqual(macro.dynamic_energy_for_active_rows_j(0), 0.0)
        self.assertAlmostEqual(
            macro.dynamic_energy_for_active_rows_j(256) * 1e12,
            377.5932641192593,
            places=9,
        )
        self.assertAlmostEqual(
            macro.dynamic_energy_for_active_rows_j(512) * 1e12,
            macro.energy_per_full_vmm_j * 0.95 * 1e12,
            places=9,
        )
        self.assertLess(
            macro.dynamic_energy_for_active_rows_j(256),
            macro.dynamic_energy_for_active_rows_j(512),
        )

    def test_partial_row_coefficients_are_macro_specific(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        expected_pj = {
            "m32x8": (56.77126143438238, 0.7258980801755507),
            "m64x8": (96.52411517183711, 0.47614490409539745),
            "m128x8": (116.67678947338607, 0.43598031273562526),
            "m256x8": (202.61576175034622, 0.3661864076238677),
            "m512x8": (298.2047314093936, 0.31011145589791284),
        }

        for name, (fixed_pj, row_pj) in expected_pj.items():
            macro = library.macro_types[name]
            self.assertAlmostEqual(
                macro.partial_vmm_fixed_energy_j * 1e12, fixed_pj, places=9
            )
            self.assertAlmostEqual(
                macro.partial_vmm_energy_per_active_row_j * 1e12,
                row_pj,
                places=9,
            )
            reconstructed_pj = fixed_pj + row_pj * macro.rows
            self.assertAlmostEqual(
                reconstructed_pj,
                macro.dynamic_energy_per_full_vmm_j * 1e12,
                places=9,
            )

    def test_m512_partial_row_coefficients_preserve_measured_power_ratio(self) -> None:
        macro = load_sram_macro_library("system_digital_sram_4x4").macro_types[
            "m512x8"
        ]
        measured_fixed_mw = 4.793
        measured_row_mw = 0.004984375

        self.assertAlmostEqual(
            macro.partial_vmm_energy_per_active_row_j
            / macro.partial_vmm_fixed_energy_j,
            measured_row_mw / measured_fixed_mw,
            places=15,
        )
        self.assertAlmostEqual(
            macro.partial_vmm_fixed_energy_j
            + macro.partial_vmm_energy_per_active_row_j * macro.rows,
            macro.dynamic_energy_per_full_vmm_j,
            places=21,
        )

    def test_all_macro_row_slopes_use_m512_measured_correction(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        original_row_pj = {
            "m32x8": 1.4686684073107048,
            "m64x8": 0.9633569739952718,
            "m128x8": 0.8820942347297592,
            "m256x8": 0.7408841857436951,
            "m512x8": 0.627430916902612,
        }
        correction = 0.31011145589791284 / original_row_pj["m512x8"]

        for name, original_slope in original_row_pj.items():
            macro = library.macro_types[name]
            self.assertAlmostEqual(
                macro.partial_vmm_energy_per_active_row_j * 1e12,
                original_slope * correction,
                places=12,
            )

    def test_partial_row_dynamic_energy_validates_row_count(self) -> None:
        macro = load_sram_macro_library("system_digital_sram_4x4").macro_types[
            "m512x8"
        ]
        with self.assertRaises(ValueError):
            macro.dynamic_energy_for_active_rows_j(-1)
        with self.assertRaises(ValueError):
            macro.dynamic_energy_for_active_rows_j(513)

    def test_peak_compute_window_keeps_the_comparison_baselines(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        reference_tops = 0.4608
        tolerance_ratio = 0.20
        _, designs = enumerate_tile_designs(
            library, reference_tops, tolerance_ratio
        )

        names = {design.name for design in designs}
        self.assertIn("9xm64x8", names)
        self.assertIn("1xm64x8+1xm512x8", names)
        self.assertTrue(designs)
        lower_tops = reference_tops * (1 - tolerance_ratio)
        upper_tops = reference_tops * (1 + tolerance_ratio)
        self.assertTrue(
            all(
                lower_tops - 1e-12
                <= design.peak_compute_tops
                <= upper_tops + 1e-12
                for design in designs
            )
        )
        self.assertTrue(all(len(design.counts) <= 2 for design in designs))
        self.assertEqual(len(designs), 138)
        self.assertEqual(sum(len(design.counts) == 1 for design in designs), 14)
        self.assertAlmostEqual(min(design.peak_compute_tops for design in designs), 0.384)
        self.assertAlmostEqual(
            max(design.peak_compute_tops for design in designs), 0.5376
        )
        self.assertEqual(library.tile_output_lanes, 4)
        self.assertEqual(library.tile_shared_bus_width_bits, 512)
        self.assertEqual(library.tile_weight_bus_width_bits, 64)
        self.assertFalse(library.tile_retain_features_across_nblocks)
        self.assertFalse(library.tile_allow_intra_bank_weight_compute_overlap)
        self.assertTrue(library.vit_allow_cross_bank_weight_compute_overlap)
        self.assertEqual(library.tile_latency_batch_granularity, "nblock_kwave")
        self.assertFalse(library.tile_include_pipeline_fill_drain)

    def test_peak_compute_window_arguments_are_validated(self) -> None:
        library = load_sram_macro_library("system_digital_sram_4x4")
        with self.assertRaises(ValueError):
            enumerate_tile_designs(library, 0.4608, -0.01)
        with self.assertRaises(ValueError):
            enumerate_tile_designs(library, 0.4608, 1.0)
        with self.assertRaises(ValueError):
            enumerate_tile_designs(library, 0.4608, 0.20, -1e-12)

    def test_incremental_state_pruning_preserves_exact_front(self) -> None:
        candidates = [
            ScheduleState(
                energy_j=float(index % 31 + 1),
                latency_s=float(300 - index % 47),
                lanes=(((index % 3,), index % 5 + 1),) * (index % 8 + 1),
            )
            for index in range(600)
        ]
        incremental = []
        for candidate in candidates:
            _extend_exact_state_front(incremental, candidate)

        self.assertEqual(
            _state_front(incremental, preserve_lane_count=True),
            _state_front(candidates, preserve_lane_count=True),
        )

    def test_resource_state_cap_is_deterministic_and_preserves_progress(self) -> None:
        states = {
            ((used,), completed): [
                ScheduleState(
                    energy_j=float(used + lane_count),
                    latency_s=float(20 - used + lane_count),
                    lanes=(((used,), 1),) * lane_count,
                )
                for lane_count in range(1, 5)
            ]
            for completed in range(4)
            for used in range(6)
        }
        capped, applied = _cap_resource_states(states, 16)
        repeated, repeated_applied = _cap_resource_states(states, 16)

        self.assertTrue(applied)
        self.assertTrue(repeated_applied)
        self.assertLessEqual(sum(map(len, capped.values())), 16)
        self.assertEqual(capped, repeated)
        self.assertEqual({key[1] for key in capped}, {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()

# Innovation 1 Artifact Map

This document records the maintained files for the compiler-driven heterogeneous
SRAM-CIM Tile study. The recommended paper figure is the normalized 2x2 result
figure; older single-workload figures were removed because they used inconsistent
layouts and were not referenced by the current draft.

## Directory Structure

```text
paper/
|-- figures/innovation1/       # Paper-ready figures and raster previews
|   |-- tile_dse_combined_result_normalized.{pdf,png}  # Recommended 2x2 figure
|   |-- tile_dse_combined_result.{pdf,png}             # Absolute-EDAP 2x2 alternate
|   |-- tile_dse_core_result.{pdf,png}                 # Pareto-only alternate
|   `-- tile_dse_edap_ranking.{pdf,png}                # Standalone ranking alternate
|-- tables/                     # CSV/Markdown numeric data behind the figures
`-- innovation1_artifacts.md    # This map and reproduction guide

experiments/
|-- RUN/tile_DSE/               # Figure and DSE generators
|   |-- generate_innovation1_core_result.py
|   `-- generate_innovation1_experiments.py
|-- results/tile_DSE/            # Generated aggregate DSE inputs (git-ignored)
|   |-- resnet18_v6/
|   `-- vision_transformer_l16_v2/
`-- docs/                        # Frozen panels and experiment documentation
```

## Recommended Figure

`paper/figures/innovation1/tile_dse_combined_result_normalized.pdf`

The main 2x2 figure for the paper. Columns are ResNet-18 and ViT-B/16. The top
row shows EDP-area Pareto points; the bottom row shows EDAP normalized to the
best evaluated architecture for each workload.

`paper/figures/innovation1/tile_dse_combined_result_normalized.png`

600 dpi raster preview of the recommended figure.

## Alternate Figures

`paper/figures/innovation1/tile_dse_combined_result.pdf`

Unnormalized 2x2 version using absolute EDAP in the bottom row.

`paper/figures/innovation1/tile_dse_core_result.pdf`

Compact two-panel EDP-area Pareto figure.

`paper/figures/innovation1/tile_dse_edap_ranking.pdf`

Standalone two-panel EDAP ranking figure. Each panel shows the Top 10 plus
explicit comparison baselines.

PNG files with the same basename are raster previews of these PDFs.

## Data Tables

`paper/tables/innovation1_core_result_pareto.csv`

Architecture-level EDP-area Pareto rows for both workloads.

`paper/tables/innovation1_core_result_selection.csv`

Comparison of `64+512`, `2x256`, and `512`, including EDP/EDAP reductions and
cross-workload EDAP regret.

`paper/tables/innovation1_core_result_edap_ranking.csv`

Rows displayed in the standalone EDAP ranking figure, including true ranks and
absolute EDAP.

`paper/tables/innovation1_tile_comparison.csv` and
`paper/tables/innovation1_tile_comparison.md`

Earlier comparison tables retained for provenance and cross-checking.

## Workload and Macro Inputs

`paper/tables/innovation1_macro_characterization.csv`

Compiler-characterized Macro efficiency and area data.

`paper/tables/innovation1_macro_row_utilization.csv`

Workload-weighted Macro row utilization data.

`paper/tables/innovation1_workload_reduction_dimensions.csv`

Reduction-dimension statistics used to explain granularity mismatch.

`paper/tables/innovation1_tile_edp_area_pareto_points.csv`

Legacy EDP-area points retained because the experiment-generation script still
produces and documents this intermediate representation.

`paper/tables/innovation1_tile_energy_adp_pareto_points.csv` and
`paper/tables/innovation1_tile_latency_energy_area_pareto_points.csv`

Legacy multi-objective points retained for experiment reproducibility.

## Generators

`experiments/RUN/tile_DSE/generate_innovation1_core_result.py`

Generates the core Pareto figure, standalone EDAP ranking, normalized and
unnormalized 2x2 figures, and their result tables from the aggregate CSVs.

`experiments/RUN/tile_DSE/generate_innovation1_experiments.py`

Generates the earlier Macro characterization, utilization, and legacy Pareto
artifacts. Its legacy table outputs are retained for provenance.

## Source Results

Source aggregate CSVs are generated under:

`experiments/results/tile_DSE/resnet18_v6/`

`experiments/results/tile_DSE/vision_transformer_l16_v2/`

These generated result directories are not committed to Git, but are required to
reproduce the figures locally.

## Reproduction

From the repository root:

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/tile_DSE/generate_innovation1_core_result.py
```

The command regenerates all maintained Innovation 1 figures and core result
tables without rerunning the expensive Tile DSE searches.

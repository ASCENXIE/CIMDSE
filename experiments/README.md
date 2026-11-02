# SRAM-CIM Design Space Exploration

This directory contains the SRAM-CIM Tile DSE code, configuration,
documentation, fixed inputs, and generated results. The active result sets are
ResNet-18 v6 and ViT-L/16 v2. ResNet-18 completed all 48 fixed-panel Tiles;
ViT-L/16 completed 36 of the 71 ordered candidates and recorded 35 timeouts.

## Directory Map

```text
experiments/
├── SRAM_config/          AccelForge architecture and Macro configuration
├── docs/
│   ├── SRAM_DATA/        characterized Macro efficiency input
│   └── tile_DSE/         current Tile model, constraints, ViT pipeline, fixed panel
├── RUN/
│   ├── tile_DSE/         active CNN Tile DSE implementation
│   │   └── vit/          isolated ViT pipeline DSE and visualization
│   └── *.py              shared models used by the active Tile DSE
└── results/
    └── tile_DSE/
        ├── resnet18_v6/            current complete ResNet-18 results
        ├── vision_transformer_l16_v2/ current ViT-L/16 results (36/71 completed)
        └── vision_transformer_v5/  legacy ViT-B/16 proxy data retained for reproducibility
```

Generated Python caches, completed-run checkpoints, pre-v5 Tile results, and
obsolete Core/System code and results are intentionally removed. Reproducible
search inputs are retained as the frozen panel, configuration, source code, and
formal CSVs.

## Current Model

- Technology: 22 nm, 0.72 V, 400 MHz.
- Macro types: `32x8`, `64x8`, `128x8`, `256x8`, and `512x8`.
- Tile output lanes: 4.
- Feature ingress bandwidth: 512 bits/cycle.
- Weight ingress bandwidth: 64 bits/cycle; FEATURE/WEIGHT traffic is exclusive.
- Maximum Macro types per Tile: 2.
- Candidate compute window: `0.4608 TOPS +/-20%`.
- Partial-row energy: Macro-specific `E_fixed + E_rows * active_rows`.
- Energy-model version: `partial_row_measured_slope_all_macros_..._v5`.

The v5 model preserves each Macro's full-row TOPS/W energy and applies the
measured `m512x8` row-slope correction to all Macro sizes. See
`docs/tile_DSE/TILE_DSE_SEARCH_CONDITIONS.md` for the exact coefficients and
assumptions.

## Active ViT-L DSE

The current ViT-L/16 workload uses 192 tokens, hidden dimension 1024, MLP
dimension 4096, 16 heads, and 24 Transformer blocks.

The formal search evaluates the frozen 48-design panel, caps physical bank
partitions at 8 per design, and skips a design after 600 seconds:

```bash
third_party/accelforge/.venv/bin/python -u \
  experiments/RUN/tile_DSE/vit/run_vit_tile_bayesian_dse.py \
  --workload-model vit_l16 \
  --partition-budget 8 \
  --design-timeout-s 600 \
  --fixed-designs-csv experiments/docs/tile_DSE/frozen_panels/vit_l16_bayesian_panel_96_filtered.csv \
  --completed-design-target 48 \
  --output-dir experiments/results/tile_DSE/vision_transformer_l16_v2
```

Regenerate all ViT figures without rerunning the search:

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/tile_DSE/vit/generate_vit_tile_dse_visualizations.py \
  --result-dir experiments/results/tile_DSE/vision_transformer_l16_v2 \
  --prefix vit_l16_pipeline_tile_dse \
  --workload-label ViT-L/16
```

The ordered input panel is stored at
`docs/tile_DSE/frozen_panels/vit_l16_bayesian_panel_96_filtered.csv`.

## Verification

```bash
third_party/accelforge/.venv/bin/python -m unittest \
  experiments.RUN.tile_DSE.test_sram_tile_dse \
  experiments.RUN.tile_DSE.vit.test_vit_tile_dse
```

Future Core/System exploration must be rebuilt against the current Tile model
before system-level results are used.

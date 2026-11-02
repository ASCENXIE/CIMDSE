# Experiment Runners

`RUN/` contains shared SRAM-CIM models and the active Tile DSE entry points.

## Active Tile DSE

- `tile_DSE/`: CNN Tile scheduler, Bayesian runner, tests, and Pareto plotting.
- `tile_DSE/vit/`: ViT projection/attention pipeline model, runner, tests, and visualizations.
- `sram_macro_types.py`: shared Macro characterization and partial-row energy model.
- `sram_workload.py`: shared CNN workload loading.

Run the active ViT search from the repository root:

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

The obsolete Core/System runners were removed because they targeted an earlier
model state. A future system-level study should be implemented against the
current Tile model and formal result schema.

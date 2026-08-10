# Experiment Runners

`RUN/` contains shared SRAM-CIM models and the active Tile DSE entry points.

## Active Tile DSE

- `tile_DSE/`: CNN Tile scheduler, Bayesian runner, tests, and Pareto plotting.
- `tile_DSE/vit/`: ViT-B/16 projection/attention pipeline model, runner, tests, and visualizations.
- `sram_macro_types.py`: shared Macro characterization and partial-row energy model.
- `sram_workload.py`: shared CNN workload loading.

Run the active ViT search from the repository root:

```bash
third_party/accelforge/.venv/bin/python -u \
  experiments/RUN/tile_DSE/vit/run_vit_tile_bayesian_dse.py \
  --partition-budget 8 \
  --design-timeout-s 600 \
  --output-dir experiments/results/tile_DSE/vision_transformer_v5
```

The obsolete Core/System runners were removed because they targeted an earlier
model state. A future system-level study should be implemented against the
current Tile model and formal result schema.

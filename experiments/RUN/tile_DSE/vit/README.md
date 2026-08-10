# ViT Tile DSE

This directory is an isolated ViT-B/16 Tile DSE implementation. It imports the
existing single-bank Tile scheduler as a read-only dependency but does not alter
the CNN workload loader, sequential CNN aggregation, Bayesian entry point, or
CNN result files.

Run the formal 48-design search with exhaustive physical bank partitions:

```bash
third_party/accelforge/.venv/bin/python -u \
  experiments/RUN/tile_DSE/vit/run_vit_tile_bayesian_dse.py \
  --partition-budget 8 \
  --design-timeout-s 600 \
  --output-dir experiments/results/tile_DSE/vision_transformer_v5
```

The default input panel is `experiments/docs/tile_DSE/frozen_panels/
resnet18_tile_panel_48.csv`, a frozen 48-design ResNet-18 panel. ResNet-50 and
VGG16 contain the same design set, so this preserves the required cross-network
comparison without depending on stale CNN result directories. `--select-new-panel`
is available only for a separately labelled independent ViT Bayesian experiment.

Resume the same configuration with `--resume`. Checkpoint compatibility covers
the energy, workload, and pipeline model versions, batch size, partition budget,
and all local and aggregate state limits.

The formal ViT search uses `partition_budget=8`. A Tile that exceeds 600 seconds
is recorded as `timeout_skipped` and the fixed panel continues. Timeout records
are written to `vit_b16_pipeline_tile_dse_timeouts.csv`.

Generate the ViT result visualizations without rerunning the search:

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/tile_DSE/vit/generate_vit_tile_dse_visualizations.py
```

The command writes PNG and PDF versions of the energy-latency, energy-area, and
energy-latency-area Pareto plots plus the per-architecture comparison into the
ViT result directory. The energy-area plot uses one minimum-energy mapping per
completed architecture and colors each point by its latency. It also writes
`vit_b16_pipeline_tile_dse_architecture_summary.csv`, which contains one row per
completed Tile design, objective regrets, minimum-EDP rank, and Pareto membership.
The seven-row `vit_b16_pipeline_tile_dse_energy_area_pareto.csv` records the
plotted front. Timeout-skipped designs remain explicitly excluded from ranking
and Pareto claims.

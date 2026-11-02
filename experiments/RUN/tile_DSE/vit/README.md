# ViT Tile DSE

This directory is an isolated ViT Tile DSE implementation. It imports the
existing single-bank Tile scheduler as a read-only dependency but does not alter
the CNN workload loader, sequential CNN aggregation, Bayesian entry point, or
CNN result files.

Run the current ViT-L/16 ordered-panel search:

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

Future runs pre-skip the empirically explosive dense-small-Macro region when a
Tile contains at least nine `m32x8` Macros, or when a heterogeneous Tile has at
least twelve total Macros, no `m512x8`, and at most one `m256x8`. Such points are
recorded as `complexity_skipped`, not `timeout_skipped`. Pass
`--evaluate-dense-small-macro-designs` to disable this rule and reproduce the
earlier exhaustive-per-design behavior.

Use `--completed-design-target 48` with a larger ordered candidate panel to keep
selecting replacements until 48 designs have valid metrics; skipped and timed-out
designs do not count toward this target.

Generate the ViT result visualizations without rerunning the search:

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/tile_DSE/vit/generate_vit_tile_dse_visualizations.py \
  --result-dir experiments/results/tile_DSE/vision_transformer_l16_v2 \
  --prefix vit_l16_pipeline_tile_dse \
  --workload-label ViT-L/16
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

## ViT-L/16

The ViT-L/16 workload uses 192 tokens, hidden dimension 1024, 16 attention
heads, head dimension 64, MLP dimension 4096, and 24 Transformer blocks. It
uses the same frozen 48-Tile panel and mapping constraints as ViT-B/16, while
writing to an independent result directory:

```bash
third_party/accelforge/.venv/bin/python -u \
  experiments/RUN/tile_DSE/vit/run_vit_tile_bayesian_dse.py \
  --workload-model vit_l16 \
  --partition-budget 8 \
  --design-timeout-s 600
```

The default output directory is
`experiments/results/tile_DSE/vision_transformer_l16_v2/`, with prefix
`vit_l16_pipeline_tile_dse_*`.

# Tile DSE

- `sram_tile_dse.py`: Tile composition, mapping, energy, latency, and area model.
- `run_sram_tile_dse.py`: exhaustive CNN Tile runner.
- `run_sram_tile_bayesian_dse.py`: fixed-panel/Bayesian CNN Tile runner.
- `generate_sram_tile_dse_pareto.py`: CNN result visualization.
- `profile_sram_tile_dse.py`: scheduler profiling entry point.
- `test_sram_tile_dse.py`: shared Tile model regression tests.
- `vit/`: isolated ViT-B/16 projection/attention pipeline DSE.

Shared Macro definitions and architecture loading remain one level above in
`experiments/RUN/`.

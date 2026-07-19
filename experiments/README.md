# SRAM-CIM Design Space Exploration

This directory contains the source configuration and scripts for Macro-Tile-Core-System
resource exploration. The current scope models Macro VMM energy, abstract vector psum
energy, and weight-stationary scheduling across four Tiles in one Core.

## Source Layout

- `SRAM_config/`: AccelForge architecture and Macro type table.
- `docs/SRAM_DATA/`: characterized Macro efficiency table.
- `RUN/`: Macro library loader, ResNet-18 scheduler, and visualization script.
- `results/`: generated output, ignored by Git.

## Run

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/run_sram_resnet18_tile_scheduler.py

third_party/accelforge/.venv/bin/python \
  experiments/RUN/generate_sram_resnet18_tile_scheduler_visualization.py
```

The scheduler excludes line FIFOs, GLB, NoC, DRAM, leakage, and data-transfer latency.

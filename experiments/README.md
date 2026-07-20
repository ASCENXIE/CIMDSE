# SRAM-CIM Design Space Exploration

This directory contains the source configuration and scripts for Macro-Tile-Core-System
resource exploration. The current Tile-DSE scope models Macro VMM energy, Aladdin
input/output-register dynamic energy, abstract vector psum energy, and single-Tile
weight-stationary scheduling.

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

third_party/accelforge/.venv/bin/python \
  experiments/RUN/run_sram_tile_dse.py \
  --all-layers --capacity-tolerance 0.20 --area-tolerance 0.20

third_party/accelforge/.venv/bin/python \
  experiments/RUN/run_sram_tile_bayesian_dse.py \
  --design-budget 40 --initial-designs 16 --batch-size 4 --workers 2 \
  --latency-cap-ratio 2.0 --local-pareto-epsilon 0.001 --resume

third_party/accelforge/.venv/bin/python \
  experiments/RUN/generate_sram_tile_dse_pareto.py
```

Each Macro has two `rows x 8-bit` ping-pong input-register banks and one non-ping-pong
`8 x 25-bit` output-register bank. Per VMM invocation, the energy model charges one
full input-bank read and write plus one output-bank read and write. Aladdin register
area is included in the Tile area and area-tolerance filter. Each Macro independently
loads one valid eight-weight row per cycle; loads do not overlap computation on the
same Macro, but different Macros may load in parallel. Weight-load energy is zero.
The model excludes
ping-pong control, fixed bank overhead, input broadcast, output reduction, multi-Macro
synchronization, register latency, line FIFOs, GLB, NoC, DRAM, leakage, and data-transfer
latency.
Each Tile may contain at most two distinct Macro types, with any instance count that
also satisfies the capacity/area window. A layer activates one to four output lanes;
Macro results in the same lane are accumulated, while different lanes calculate
independent output blocks.
The exhaustive Tile DSE defaults to representative ResNet-18 layers
`T0,T1,T7,T12,T17,T20`. The Bayesian runner searches the outer Tile-composition
space while retaining exact per-layer Macro grouping and N-block assignment. Its
checkpoints can be resumed without reevaluating completed Tile designs. Cross-layer
energy-latency states use a recorded epsilon compression and a default cap of 512
states per Tile. The default scheduler also limits each layer to twice the latency of
a simple feasible schedule and applies a 0.1% local Pareto epsilon. CSV output reports
row utilization, temporal Macro utilization, and their product separately.

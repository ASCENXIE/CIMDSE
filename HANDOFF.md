# SRAM-CIM System Evaluation Handoff

## Current Objective

Continue building and validating a system-level AccelForge model composed of SRAM-CIM
black-box macros. The user's latest technical question is:

> Why is the T0 GlobalBuffer energy so high? T0 should be blockable, and because its
> reduction fits in one macro invocation, partial sums should not need to be written
> back to the GlobalBuffer.

This question is not resolved yet. The evidence below strongly suggests that the high
T0 output traffic is a mapping/modeling artifact rather than an algorithmic necessity.

## Intended Architecture

```text
LPDDR-style DRAM
  -> 384 KiB / 16-bank shared GlobalBuffer
    -> 512 bit/cycle 1x4 inter-Tile NoC
      -> four Tiles
        -> one TileInputBuffer per Tile
        -> four 512x8 SRAM-CIM black-box macros per Tile
```

- Total macros: 16.
- Four macros in a Tile share one input and produce `4 x 8 = 32` outputs in parallel.
- Four Tiles parallelize independent output positions.
- Reduction (`K`) tiles execute sequentially; no cross-macro reduction is modeled.
- Macro compiler characterization: 17.03 TOPS/W, 320252.5 um2/macro,
  1.4278 ns cycle, 64 Kib/macro.
- NoC assumptions: 0.25 pJ/bit/hop, average 1.5 hops, 512 bit/cycle.
- Native AccelForge `!Network` is not used for ResNet convolution because mesh and
  all-to-all topology models raise `NotImplementedError` for partially-relevant
  sliding-window transfers. NoC traffic is calculated explicitly in Python.

## Important Files

- `experiments/SRAM_config/basic_digital_sram_512x8.yaml`
  - Validated single-macro black-box configuration.
- `experiments/SRAM_config/system_digital_sram_4x4.yaml`
  - System architecture. The filename is historical; it now means four Tiles with
    four macros each, not a flat 4x4 macro mesh.
- `experiments/RUN/load_sram_spec.py`
  - Loads experiment-local YAML and helper/component models.
- `experiments/RUN/sram_macro_model.py`
  - Counts physical macro invocations, schedules Tiles/macros, and explicitly estimates
    NoC energy/latency.
- `experiments/RUN/run_sram_system_resnet18.py`
  - Runs ResNet-18 Conv/FC einsums and prints system energy, latency, and area breakdown.
- `experiments/docs/learning/sram_system_resnet18_results.csv`
  - Last generated per-layer system results. Check timestamps before trusting it.
- `experiments/docs/SRAM_DATA/cim_macro_efficiency.csv`
  - Macro efficiency source table.
- `experiments/SRAM_config/README.md`
  - Current architecture assumptions and commands.

## Critical Disk-State Mismatch

At handoff time, `system_digital_sram_4x4.yaml` was modified on 2026-07-18 and does
not match the last validated configuration or README:

```yaml
# Current YAML on disk
tile_input_buffer_size_bits: {{TILE_INPUT_BUFFER_KIB | default(1)}} * 1024 * 8
...
extra_attributes_for_component_model:
  width: supported_input_bits  # 8-bit interface
```

However, README and the last successful run describe:

```yaml
tile_input_buffer_size_bits: {{TILE_INPUT_BUFFER_KIB | default(12)}} * 1024 * 8
tile_input_buffer_width_bits: 512
actions:
- {name: read, bits_per_action: tile_input_buffer_width_bits}
- {name: write, bits_per_action: tile_input_buffer_width_bits}
extra_attributes_for_component_model:
  width: tile_input_buffer_width_bits
```

Do not silently overwrite the current YAML. Determine whether the 2026-07-18 change
was intentional, then restore or reconcile it with the user. The current 1 KiB/8-bit
version is expected to fail full ResNet-18 mapping.

The CSV timestamp was older than the current YAML, so it represents the previously
validated 12 KiB/512-bit configuration, not the YAML currently on disk.

## TileInputBuffer Capacity Results

A capacity sweep was run with the full ResNet-18 Conv/FC workload:

- 1 KiB: failed mapping at T0.
- 2 KiB: failed mapping at T0.
- 4 KiB: failed mapping at T0.
- 8 KiB: failed mapping at T0.
- 12 KiB: passed.
- 16 KiB: passed.

Therefore 12 KiB/Tile was selected as the tested baseline. It is an input-tile buffer,
not a convolution line buffer; it does not yet explicitly model overlap reuse between
adjacent windows.

With an 8-bit interface, TileInputBuffer latency was incorrectly dominant
(`~14.3 ms`). Setting the memory model and actions to 512 bits/access reduced the
TileInputBuffer path to approximately 0.25 ms.

## Last Validated System Result

For 384 KiB GLB + four 12 KiB/512-bit TileInputBuffers:

```text
Energy: 4.5607 mJ
  Macro:            0.3244 mJ
  DRAM:             0.9309 mJ
  GlobalBuffer:     3.1092 mJ
  TileInputBuffers: 0.0091 mJ
  NoC:              0.1871 mJ

Latency: 1.6412 ms
  Macro path:            0.4862 ms
  DRAM path:             0.3829 ms
  GlobalBuffer path:     0.3574 ms
  TileInputBuffer path:  0.2498 ms
  NoC path:              1.3911 ms

Effective throughput: 2.2107 TOPS
Effective efficiency: 0.7955 TOPS/W

Modeled on-chip area: 13.4603 mm2
  Macro array:       5.1240 mm2
  GlobalBuffer:      8.2784 mm2
  TileInputBuffers:  0.0578 mm2
  NoC area:          uncharacterized, currently zero
```

Single-macro regression remained:

```text
Energy: 0.3244 mJ
Latency: 7.7029 ms
Effective efficiency: 11.1844 TOPS/W
```

## Evidence About the T0 GLB Problem

T0 is the first convolution:

```text
K = C x R x S = 3 x 7 x 7 = 147
macro reduction capacity = 512
reduction_tiles = ceil(147 / 512) = 1
```

The generated CSV confirms `reduction_tiles=1`. Therefore T0 does not require
cross-macro or cross-K-tile partial-sum accumulation. A correct output-stationary
blocked schedule can complete an output locally and write it once.

Nevertheless, the last CSV reports for T0:

```text
global_buffer_energy_j = 0.002632034450458558  # 2.632 mJ
macro_energy_j         = 0.0000482726708162
```

Before shrinking the GLB and adding TileInputBuffer, the 1 MiB system had total GLB
energy of 1.6563 mJ over all layers. Its role breakdown was approximately:

```text
input:              1.0476 mJ
weight:             0.4151 mJ
intermediate/output 0.1918 mJ
leak:               0.0018 mJ
```

After moving to 384 KiB GLB + TileInputBuffer, diagnostic mapping showed approximately:

```text
GlobalBuffer input:  0.0985 mJ  # TileInputBuffer successfully reduced this
GlobalBuffer weight: 0.2557 mJ
GlobalBuffer output: 2.7542 mJ  # new dominant term
GlobalBuffer leak:   0.0250 mJ
```

Thus TileInputBuffer did reduce input traffic. The regression is output traffic caused
by mapping/storage behavior after reducing GLB capacity, not input traffic.

The first output tensor has about `112 x 112 x 64 x 8 bit = 784 KiB`, larger than the
384 KiB GLB. That alone requires blocking, but it does not require partial sums to be
written repeatedly because T0 has only one reduction tile. The mapper appears to be
choosing an undesirable output storage/loop placement, or the current memory hierarchy
is charging output reads/writes that should stay local/finalize once.

## Next Diagnostic Steps

1. Reconcile the current YAML mismatch with the user. Use the validated 12 KiB/512-bit
   TileInputBuffer when reproducing results.
2. Print the selected T0 mapping and inspect the relative order of:
   - `GlobalBuffer` output storage/reservation
   - `TileInputBuffer`
   - `TileArray` and `MacrosInTile` spatial loops
   - K loops (`c0`, `r0`, `s0`)
   - output loops (`p0`, `q0`, `m0`, `n`)
3. Extract T0 GLB energy by tensor/action and divide by action energy to get exact bit
   access counts. Focus on `T0 read` and `T0 write`.
4. Verify whether an output storage node is below any K temporal loop despite
   `reduction_tiles=1`, or whether GLB output is being read once per compute wave.
5. Constrain output tiling so a completed output tile is written once. Likely options:
   - add a real per-Tile `TileOutputBuffer`/accumulation buffer below the Tile spatial
     level and retain output there until all K work completes;
   - add output `tile_shape` constraints or storage-order constraints;
   - explicitly calculate black-box system output traffic, as is already done for NoC,
     if FFM cannot express the intended block schedule reliably.
6. Do not assume a TileOutputBuffer is mandatory for T0 mathematically. First establish
   whether the extra GLB reads are an FFM mapping artifact. The user's objection is
   technically correct.

Useful run command:

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/run_sram_system_resnet18.py \
  --output experiments/docs/learning/sram_system_resnet18_results.csv
```

Useful single-macro regression:

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/run_sram_resnet18.py \
  --config basic_digital_sram_512x8
```

## Known Modeling Boundaries

- ResNet-18 workload includes 21 Conv/FC einsums only. ReLU, BN, pooling, residual add,
  control, and complete tensor dependencies are not modeled.
- The provided ResNet workload names layer inputs independently (`I0`, `I1`, ...), so
  end-to-end feature-map producer/consumer relationships are abstract rather than a
  full executable graph.
- NoC area and Tile-local broadcast energy are zero because no characterization exists.
- NoC traffic is an explicit average-hop model, not native AccelForge routing.
- Macro energy is applied after mapping from physical invocation count. Mapper energy
  optimization therefore does not directly include the characterized macro energy.
- `GlobalBuffer: {keep: All}` gives FFM broad freedom and may encourage storage choices
  inconsistent with the intended hand-designed blocked dataflow.

## Worktree Safety

The worktree is dirty:

```text
 m third_party/accelforge
?? experiments/
?? tutorial_basic_digital_sram.ipynb
```

Do not reset or revert these paths. The root notebook and third-party changes may be
user work. Experiment files are currently untracked but contain the ongoing model.

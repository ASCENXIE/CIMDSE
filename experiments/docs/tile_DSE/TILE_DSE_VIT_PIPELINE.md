# ViT-B/16 Tile DSE Pipeline Specification

> Version: 1
> Date: 2026-07-27

## 1. Isolation and reproducibility

The ViT implementation is located under `experiments/RUN/tile_DSE/vit/`. It does
not modify the existing CNN Tile DSE files:

- `tile_DSE/sram_tile_dse.py`
- `tile_DSE/run_sram_tile_dse.py`
- `tile_DSE/run_sram_tile_bayesian_dse.py`
- `tile_DSE/test_sram_tile_dse.py`

The CNN entry points, workload parsing, sequential network aggregation, result
directories, model-version strings, and checkpoints retain their previous
behavior. ViT results use the independent directory
`experiments/results/tile_DSE/vision_transformer_l16_v2/` for the current
ViT-L/16 study. The older ViT-B/16 result is retained only for proxy-analysis
reproducibility.

The upstream AccelForge `vision_transformer.yaml` is also left unchanged. The
ViT DSE uses an explicit compact workload definition so the corrected attention
shape cannot change a third-party workload used by other experiments.

## 2. Workload

The first workload is ViT-B/16 with batch size 1:

| Parameter | Value |
|---|---:|
| Transformer blocks | 12 |
| Tokens | 192 |
| Hidden dimension | 768 |
| MLP dimension | 3072 |
| Heads | 12 |
| Head dimension | 64 |
| Class count | 1000 |

The cross-workload comparison also evaluates ViT-L/16 under the same Tile
search panel and constraints:

| Parameter | ViT-L/16 value |
|---|---:|
| Transformer blocks | 24 |
| Tokens | 192 |
| Hidden dimension | 1024 |
| MLP dimension | 4096 |
| Heads | 16 |
| Head dimension | 64 |
| Class count | 1000 |

The ViT-L/16 GEMM-only workload contains `59,946,016,768` MACs.

The GEMM-only workload has seven repeated operator groups and
`17,100,945,408` MACs. Softmax, LayerNorm, GELU, residual addition, and pipeline
fill/drain overhead are outside the current CIM model.

The attention shapes preserve both the head and query-token axes:

| Operator | K | N | P | Repetitions |
|---|---:|---:|---:|---:|
| `QK^T` | 64 | 192 | `12 * 192 = 2304` | 12 |
| `Attention * V` | 192 | 64 | `12 * 192 = 2304` | 12 |

This replaces the upstream generic representation whose `P=12` omits the 192
query positions.

## 3. Physical Macro assignment

For every multi-Macro Tile, all physical Macro instances are partitioned into
two non-empty, disjoint banks:

- projection bank: Q, K, V, O, FFN-up, and FFN-down projections;
- attention bank: `QK^T` and `Attention * V`.

The partition is fixed for all 12 Transformer blocks. The mapper searches both
role directions and does not give `m64x8+m512x8` a preferred assignment. For
example, it evaluates both `m512x8 -> projection, m64x8 -> attention` and the
reverse assignment.

A single-Macro Tile remains feasible as a sequential baseline. Its projection
and attention streams cannot overlap.

## 4. Lane and register constraints

The two concurrent banks share the physical four-lane Tile interface:

```text
projection_lanes >= 1
attention_lanes >= 1
projection_lanes + attention_lanes <= 4
```

All six legal ordered lane allocations are evaluated for every retained Macro
partition. The four lanes still correspond to the fixed 32 output registers and
32 output ports. A sub-bank never receives an independent four-lane interface.

Patch embedding and classification are outside the concurrent Transformer
pipeline and use the full Tile with all four lanes.

## 5. Pipeline latency

Each operator schedule first uses the Tile-level N-block/K-wave equation:

```text
T_operator = sum_over_batches(
    T_weight[b] + max(T_feature[b], T_compute[b])
)
```

The ViT pipeline then separates each schedule's Macro critical component
(`T_weight + T_compute`) from its feature-limited extension. Across all repeated
Transformer operators:

```text
T_projection = sum(projection Macro-critical latency)
T_attention  = sum(attention Macro-critical latency)
T_bus        = sum(all projection and attention shared-bus cycles) * cycle

T_transformer = max(T_projection, T_attention, T_bus)
T_network = T_patch_embedding + T_transformer + T_classifier
```

This is the agreed steady-state pipeline abstraction. Operator dependencies do
not add a separate critical path, and fill/drain bubbles are ignored. Weight
loading and compute remain mutually exclusive within one bank, but projection-bank
compute may overlap attention-bank weight loading and vice versa.

For the single-Macro sequential baseline:

```text
T_transformer = max(T_projection + T_attention, T_bus)
```

## 6. Shared bus

Projection and attention share mutually exclusive Tile ingress timing. Feature
transfers use 512 bit/cycle and weight transfers use 64 bit/cycle. Feature and
weight transfers cannot occur in the same cycle. Bus arbitration is not modeled,
but all bus cycles from both streams are summed before applying the pipeline
maximum. This prevents the two banks from implicitly receiving independent
configured feature and weight bandwidths.

The bus energy parameter remains present and is currently zero, consistent with
the CNN Tile DSE. Weight-row write energy is still counted, including dynamic
attention operands written to the attention bank.

## 7. Energy and area

Dynamic energy is additive across all operators and both streams:

```text
E_dynamic = E_projection + E_attention + E_patch + E_classifier
```

The single-bank scheduler's partial static energy is removed during ViT
aggregation. Static energy is then recomputed exactly once using every physical
Macro in the Tile:

```text
E_static = P_static_full_tile * T_network
E_total = E_dynamic + E_static
```

Idle Macros therefore have zero dynamic energy but retain static power. Tile area
is always the original full physical Tile area, not the sum of temporary bank
diagnostics.

## 8. Search and outputs

The outer search retains the existing peak-compute window centered at
0.4608 TOPS with +/-20% tolerance, at most two Macro types per Tile, and the same
energy, latency, and area objectives. The formal run evaluates the frozen set of
48 Tile designs originally selected by Bayesian optimization for ResNet-18. The
ResNet-50 and VGG16 result panels contain the same design set, so this keeps every
network comparison paired by physical Tile composition. A new ViT-specific panel
may only be generated as a separately labelled experiment with
`--select-new-panel`; it is not used for the final cross-network threshold list.

The formal ViT run uses `partition_budget=8`: it keeps the analytically best
eight partitions and all two-type pure-specialization assignments. The six legal
lane allocations remain exhaustive. Validation against the first 18 exact Tile
results retained all 54 energy, latency, and EDP endpoints.

Each physical Tile has a 600-second wall-clock evaluation limit. A timeout is
recorded as `timeout_skipped` in the sample and timeout CSV files, and the runner
continues with the next fixed-panel Tile. Timeout-skipped Tiles do not enter the
aggregate or Pareto fronts.

Later ViT runs pre-skip the empirically explosive dense-small-Macro region:
`m32x8_count >= 9`, or a heterogeneous Tile with at least twelve total Macros,
no `m512x8`, and at most one `m256x8`. These points remain in the frozen panel
with status `complexity_skipped`; they are not counted as evaluated designs or
silently removed. Pass `--evaluate-dense-small-macro-designs` to disable the
rule and reproduce the earlier exhaustive-per-design behavior.

The ViT aggregate CSV reports at least:

- projection and attention bank compositions;
- lane budgets and concurrent/sequential mode;
- projection, attention, shared-bus, Transformer, endpoint, and network latency;
- dynamic energy, full-Tile static energy, total energy, area, and TOPS/W;
- pipeline/workload model versions and schedule trace.

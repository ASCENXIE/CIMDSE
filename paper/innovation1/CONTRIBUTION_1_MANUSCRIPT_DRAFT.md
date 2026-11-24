# Contribution 1 manuscript draft

## Introduction contribution statement

We propose an all-digital SRAM-CIM tile that combines macros with different numbers of compute rows. The tile can combine these macros along the reduction dimension of a convolution or allocate them to different Transformer operators. A workload-driven design-space exploration jointly selects the macro composition and its mappings under common peak-throughput and interface constraints. Across four vision workloads and 48 evaluated tile configurations, a tile with one 64-row macro and one 512-row macro achieves the lowest joint normalized energy–delay–area product (EDAP).

## II. Heterogeneous SRAM-CIM Tile and Design-Space Exploration

### A. Compute-row heterogeneity

The reduction dimension $K$ is the number of input terms accumulated for an output. Large macros amortize peripheral costs over more operations when most rows are active. Small macros can handle short reductions with fewer idle rows. In the full-row W8A8 characterization, the 64-row and 512-row macros achieve 7.36 and 17.24 TOPS/W, respectively. A short reduction occupies only part of a large macro and still incurs a fixed cost per vector–matrix multiplication. Macro sizing therefore depends on the reduction lengths encountered in the workload and on the cost of combining or moving partial sums.

The proposed tile groups macros into output lanes. Macros in one lane may receive disjoint $K$ slices for the same output channels and combine their partial sums locally. Different lanes can instead process distinct output channels. For a $3\times3$ convolution with 64 input channels, $K=576$. A 512-row macro alone executes this reduction in two waves and stores an intermediate partial sum between them. Pairing it with a 64-row macro in the same lane covers all 576 terms in one wave. The local combination remains, and the cross-wave partial-sum write and read are eliminated for this layer.

The same physical macros can also serve different operator roles. In the selected ViT mapping, the 512-row macro handles projections and the feed-forward network, while the 64-row macro handles the attention products $QK^{\mathsf T}$ and $AV$. The $QK^{\mathsf T}$ reduction spans the 64 elements in a head. The $AV$ reduction spans the key/value block and can require multiple waves on the same macro. The mapper therefore evaluates the attention block sizes together with the role assignments.

### B. Workload-driven composition and mapping

A candidate specifies the counts of 32-, 64-, 128-, 256-, and 512-row macros. Each configuration contains at most two row counts and falls within a peak-throughput window of 0.288 TOPS ±20%. The configuration stays fixed across workloads, while the mapper can regroup or idle its macros. The resulting design space contains 134 eligible configurations. An initial set of configurations and Bayesian-guided selection on ResNet-18 and ViT-B/16 yield a 48-configuration evaluation set. The same configurations are then evaluated on VGG-16 and ViT-L/16.

For CNNs, the mapper explores macro grouping, output-channel assignment, and reduction waves. It screens mappings by estimated energy and latency, evaluates shortlisted mappings on weighted spatial positions, and combines layer options into a network energy–latency frontier. It then replays the selected layers over all spatial positions, updates their costs, and reselects the network schedule until every selected layer has been fully replayed. For ViTs, the mapper explores query and key/value block sizes and serial and role-partitioned mappings. A block-level estimate guides detailed scheduling, and finalists are evaluated on full-network event graphs under data dependencies and shared-resource constraints. Only verified schedules supply the reported costs.

For tile $x$ and workload $w$, let $Q(x,w)$ be the minimum $E\times T\times A$ among the retained, verified schedules, with energy $E$, latency $T$, and local tile area $A$ taken from the same schedule and physical configuration. The joint score is the geometric mean of the per-workload normalized EDAP values over the four workloads $\mathcal W$ and the common evaluation set $\mathcal P$.

$$
G(x)=\left[\prod_{w\in\mathcal W}
\frac{Q(x,w)}{\min_{x'\in\mathcal P}Q(x',w)}\right]^{1/4}.
$$

## IV. Experimental Results — Tile selection and mapping ablation

We evaluated all 48 configurations on the same four workloads under a single-tile model at 22 nm, 0.72 V, and 250 MHz with W8A8 operands and eight output columns per macro. Energy accounts for macro computation and static power, weight loading, partial-sum handling, and dynamic transfers through the on-tile buses and reference global buffer. Area covers the macros and local registers. The evaluation assumes sufficient external buffer service and ideal service for non-CIM operators. The resulting EDAP characterizes modeled CIM execution and its interface costs using local tile area.

The `64+512` tile achieves the lowest joint score, 1.054288, which is 5.38% below the second-ranked configuration with a single 512-row macro. It ranks first for ResNet-18 and VGG-16, fifth for ViT-B/16, and second for ViT-L/16. The EDAP of the selected tile is 18.34% above the workload-specific minimum on ViT-B/16 and 4.40% above it on ViT-L/16, quantifying the trade-off associated with selecting a single configuration for all four networks. Its peak throughput is 0.288 TOPS, and its modeled local area is 0.373928 mm².

To measure the mapping benefit at fixed hardware, we kept both macros, both output lanes, and the cost model unchanged, and searched separately under each mapping condition. The CNN control permits only one macro type per lane, while the enabled condition allows mixed-type reduction within a lane. The ViT control uses a common serial mapping, while the enabled condition allows role-specific macro allocation. Both ViT conditions use query and key/value blocks of 192 tokens.

| Workload | Enabled mapping capability | Energy reduction | Latency reduction | EDAP reduction |
|---|---|---:|---:|---:|
| ResNet-18 | Mixed macro types within a lane | 9.38% | 20.96% | 28.37% |
| VGG-16 | Mixed macro types within a lane | 4.51% | 18.67% | 22.34% |
| ViT-B/16 | Role-partitioned mapping | 2.53% | 11.61% | 13.85% |
| ViT-L/16 | Role-partitioned mapping | 2.15% | 10.78% | 12.69% |

In both CNNs, the selected cooperative mapping reduces the representative $K=576$ layer from two waves to one. Its cross-wave partial-sum reads and writes fall to zero, and whole-network partial-sum traffic falls by 42.98% and 38.62% per direction for ResNet-18 and VGG-16, respectively. In the ViTs, the role-partitioned schedule changes the macro assigned to attention and lets projection and attention for different heads overlap. The reported ViT reductions measure the combined effect of role assignment and legal scheduling overlap.

## Editorial notes and data sources

- The contribution statement belongs in the introduction, and the architecture and search text can replace the archived Section II. The result paragraphs and table fit in the evaluation section. Reserve the implementation and separate evaluation of dual-lane multi-core stage fusion for Contribution 2. [Archived manuscript](../manuscript/1_2026_TCAS_II_Aisen.pdf) · [Contribution positioning](CONTRIBUTION_1_POSITIONING.md)
- The eight-column macro library, throughput window, and local-area definition apply to the current Tile DSE. The archived system illustration uses a different column count. The manuscript should identify which configuration each figure and result describes. [Macro characterization](../../experiments/docs/SRAM_DATA/MACRO_CHARACTERIZATION_2026-11-16.md) · [Current experimental protocol](../../experiments/docs/tile_DSE/TILE_DSE_EXPERIMENT_SPECIFICATION_V5.md)
- The ranking is over the 48 configurations evaluated on all four workloads. The model area excludes the global buffer, line buffer, interconnect, control, and vector processing unit. Partial-row energy is extrapolated from the macro characterization. [Search method](../../experiments/docs/tile_DSE/SEARCH_METHOD.md) · [Joint results](../../experiments/docs/tile_DSE/EXPERIMENT_CONCLUSION.md) · [Ranking CSV](../../experiments/results/tile_DSE/v5/pareto_review_20261120/joint_ranking.csv)
- The same-tile ablation keeps the physical resources and area fixed, including the static cost of idle macros. The ViT ablation quantifies the full role-partitioned schedule. [Ablation protocol and traces](../../experiments/docs/tile_DSE/MAPPING_ABLATION_64_512.md) · [Paired results JSON](../../experiments/results/tile_DSE/v5/mapping_ablation_64_512_20261124/comparison.json)

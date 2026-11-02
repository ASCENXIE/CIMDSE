# 创新点一实验说明

> 生成日期：2026-08-09

## 实验产物

### 1. Macro 能效与面效双轴图

- PNG：`paper/figures/innovation1/macro_efficiency_vs_size.png`
- PDF：`paper/figures/innovation1/macro_efficiency_vs_size.pdf`
- 数据：`paper/tables/innovation1_macro_characterization.csv`

该图固定使用 Tile DSE 实际采用的 `32x8`、`64x8`、`128x8`、`256x8` 和
`512x8` Macro。当前表征显示 TOPS/W 随行数明显提高，但 TOPS/mm2 在 `128x8`
达到最高值后略有下降。因此，这组数据支持“大 Macro 能效更高”，但不支持将
`512x8` 描述为面效最高的 Macro。

### 2. 归约维度与行利用率图

- PNG：`paper/figures/innovation1/workload_reduction_dimension_and_utilization.png`
- PDF：`paper/figures/innovation1/workload_reduction_dimension_and_utilization.pdf`
- 算子数据：`paper/tables/innovation1_workload_reduction_dimensions.csv`
- 利用率数据：`paper/tables/innovation1_macro_row_utilization.csv`

左图按照算子实际重复次数统计归约维度 K。右图使用下式计算理想行利用率：

```text
utilization(R) = sum(K * N * P * repeat)
               / sum(ceil(K/R) * R * N * P * repeat)
```

因此，利用率按实际 VMM 工作量加权，而不是对不同网络层做简单算术平均。当前结果中，
512-row Macro 在 ResNet-18 和 ViT-B/16 上的加权行利用率分别为 `65.67%` 和
`71.99%`，明显低于 64-row Macro 的 `98.05%` 和 `99.39%`。

### 3. 核心结果图

- 推荐主图 PNG：`paper/figures/innovation1/tile_dse_combined_result_normalized.png`
- 推荐主图 PDF：`paper/figures/innovation1/tile_dse_combined_result_normalized.pdf`
- 未归一化备选图：`paper/figures/innovation1/tile_dse_combined_result.pdf`
- 架构级 Pareto 数据：`paper/tables/innovation1_core_result_pareto.csv`

该 2x2 图按列对齐 ResNet-18 与 ViT-B/16。上排为 EDP-area Pareto，下排为归一化
EDAP 排行；每个网络的最佳已评价架构归一化为 1.0，避免把不同网络的绝对计算规模误读
为架构优劣。归一化图的柱尾只显示归一化 EDAP，绝对值和选择依据见对应数据表。

未归一化版本保留绝对 EDAP，适合审计单个网络的工作负载成本，不建议用于跨网络视觉比较。

### 4. 两个工作负载的同构/异构结果表

- Markdown：`paper/tables/innovation1_tile_comparison.md`
- CSV：`paper/tables/innovation1_tile_comparison.csv`

表中每个架构使用一条最低 EDP 映射，同时列出：

- 最低 EDP 同构 Tile；
- 最低 EDP 异构 Tile；
- 本文提出的 `64+512`；
- 严格等算力的 `9x64`；
- 单个 `512` Macro 基线。

能量、延迟和 EDP 来自同一条调度，避免把最低能量端点和最低延迟端点拼接为一个不存在
的工作点。表中的 E-ADP Pareto 标记按照相同最低 EDP 映射计算。

### 5. `64+512` 跨工作负载选择依据

- 图：`paper/figures/innovation1/tile_dse_combined_result_normalized.pdf`
- EDAP 排行图：`paper/figures/innovation1/tile_dse_edap_ranking.pdf`
- Pareto 数据：`paper/tables/innovation1_core_result_pareto.csv`
- 选择数据：`paper/tables/innovation1_core_result_selection.csv`
- 排行数据：`paper/tables/innovation1_core_result_edap_ranking.csv`

核心结果图显式比较 `64+512`、`2x256` 和单个 `512` Macro。选择表记录三种设计在
ResNet-18 与 ViT-B/16 上的 EDP、EDAP、相对最低 EDP 同构设计的改善，以及在两个网络
共同完成候选集合上的 EDAP regret。跨工作负载选择采用最小化最坏归一化 EDAP regret，
而不是根据单个网络的最低 EDP 事后选择设计。

EDAP 排行图分别显示两个工作负载的 Top 10，并额外补入未进入 Top 10 的 `2x256` 或
单个 `512` 比较基线。排行仍使用每个架构的最低 EDP 映射计算 EDAP，与 Pareto 图口径
一致。该图用于直观呈现单网络名次；最终选择仍依据跨工作负载最坏 EDAP regret。

## 复现实验

从仓库根目录运行完整 ResNet-18 v5 固定面板：

```bash
PYTHONPATH=experiments/RUN:third_party/accelforge/src \
third_party/accelforge/.venv/bin/python -u \
  experiments/RUN/tile_DSE/run_sram_tile_bayesian_dse.py \
  --workload resnet18 \
  --fixed-designs-csv \
    experiments/docs/tile_DSE/frozen_panels/resnet18_tile_panel_48.csv \
  --output-subdir resnet18_v5 \
  --workers 1
```

生成全部图表和数据表：

```bash
PYTHONPATH=experiments/RUN:third_party/accelforge/src \
third_party/accelforge/.venv/bin/python \
  experiments/RUN/tile_DSE/generate_innovation1_experiments.py \
  --require-dse-results
```

生成器会检查 ResNet-18 与 ViT-B/16 是否使用相同的 Macro 能量模型版本和相同的固定
候选面板。如果任一结果缺失或版本不一致，带 `--require-dse-results` 的命令会失败，
而不会混合生成论文结果。

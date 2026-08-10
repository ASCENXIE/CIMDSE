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

### 3. Tile 能量-面积延迟积 Pareto 图

- PNG：`paper/figures/innovation1/resnet18_vitb_tile_energy_adp_pareto.png`
- PDF：`paper/figures/innovation1/resnet18_vitb_tile_energy_adp_pareto.pdf`
- 绘图数据：`paper/tables/innovation1_tile_energy_adp_pareto_points.csv`

每个 Tile 使用其最低 EDP 映射形成一个架构点：横轴为该映射的能量，纵轴为同一映射的
`面积 x 延迟`。蓝色圆点表示同构 Tile，橙色方块表示异构 Tile，黑色边框表示
能量-面积延迟积 Pareto，红色叉号表示 `64+512`。

ResNet-18 使用完整的 48/48 固定面板。ViT-B/16 当前只有 38/48 个完成点，另外 10 个
小 Macro 密集候选因 600 秒限制被跳过。因此 ViT 子图是当前已完成候选上的前沿，不能
在论文中表述为完整 48 点 Pareto。

### 3.1 Tile 延迟-能量面积积 Pareto 图

- PNG：`paper/figures/innovation1/resnet18_vitb_tile_latency_energy_area_pareto.png`
- PDF：`paper/figures/innovation1/resnet18_vitb_tile_latency_energy_area_pareto.pdf`
- 绘图数据：`paper/tables/innovation1_tile_latency_energy_area_pareto_points.csv`

该图使用相同的最低 EDP 代表映射，横轴为延迟，纵轴为 `能量 x 面积`。蓝色圆点表示
同构 Tile，橙色方块表示异构 Tile，黑色边框表示延迟-能量面积积 Pareto，红色叉号
表示 `64+512`。该图与能量-面积延迟积图使用相同的架构点，仅改变两个复合目标的组织
方式，便于观察不同指标定义对前沿成员的影响。

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

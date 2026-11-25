# `64+512` Tile 的映射消融

固定一个 64 行和一个 512 行 Macro 组成的 Tile，在四个负载上分别运行两种映射条件。四组对照共完成 8 项整网模型评价。开放异构协作后，两款 CNN 的 EDAP 分别下降 **28.37%** 和 **22.34%**。允许按角色分区后，两款 ViT 的 EDAP 分别下降 **13.85%** 和 **12.69%**。四组启用侧的能量、延迟和 EDAP 均复现了原共同面板选中路径的成本。 [配对数据与机制核验](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/comparison.json) · [原共同面板逐调度成本](../../results/tile_DSE/v5/pareto_review_20261120/points.csv)

## 对照如何设置

两侧共用每个负载的同一份定义、W8A8 精度、250 MHz 频率、参考 GB 和本地通路。同一 Tile 始终安装两个 Macro 和两条输出 lane，本地面积均为 **0.3739277984 mm²**。闲置 Macro 仍计静态能量。 [实验规范](TILE_DSE_EXPERIMENT_SPECIFICATION_V5.md) · [消融入口](../../RUN/tile_DSE/v5/run_mapping_ablation.py)

| 负载 | 对照组 | 启用组 | 两侧怎样选出调度 |
|---|---|---|---|
| ResNet-18、VGG-16 | 每条 lane 只能使用一种 Macro 类型。两种 Macro 可以分在不同 lane 各自计算，也可以按层闲置。 | 允许 64 行与 512 行 Macro 在同一 lane 沿 K 协作并归约。 | 两侧分别重做候选生成、配额检查、最多 4 个映射的带权评价和整网路径重选。最终路径逐层完整回放，并校验工作量、访问账本与映射约束。 |
| ViT-B/16、ViT-L/16 | `serial`。所有角色使用同一份 `full` 映射，候选中包括两块 Macro 合为一条 lane 或分在两条 lane 的布局。 | `partitioned`。Q/K/V 投影与注意力可用不同 Macro，首尾算子、O 和 FFN 仍使用 `full`。 | 两侧均固定 `Bq=Bk=192`，分别详评所属模式的全部生成策略（串行 7 条、分区 10 条）。关闭经验早停，各取单 block 推算 EDAP 最低的 2 条做整网评价与校验。 |

CNN 的限制施加在粗筛前，两侧都可以针对各层重新选择映射。ViT 的对照衡量允许按角色分区后整套映射的收益，其中包含注意力所用 Macro 的变化和跨 head 重叠。 [CNN 候选限制](../../RUN/tile_DSE/v5/search.py#L76) · [ViT 策略筛选](../../RUN/tile_DSE/v5/vit_hierarchical.py#L185) · [ViT 算子角色分配](../../RUN/tile_DSE/v5/mapping.py#L69)

## 整网模型结果

每格按「对照组 → 启用组」书写。EDAP 降幅以对应对照组为分母。本地面积相同，各对中的 EDAP 变化来自能量和延迟。 [原始数值与计算结果](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/comparison.json)

| 负载 | 能量（mJ） | 延迟（ms） | EDAP（J·s·mm²） | EDAP 降幅 |
|---|---:|---:|---:|---:|
| ResNet-18 | 0.423911 → 0.384158 | 31.191952 → 24.653904 | `4.944307e-6 → 3.541470e-6` | **28.37%** |
| VGG-16 | 3.265463 → 3.118064 | 252.595328 → 205.432000 | `3.084309e-4 → 2.395195e-4` | **22.34%** |
| ViT-B/16 | 4.242232 → 4.134905 | 254.084992 → 224.578576 | `4.030521e-4 → 3.472335e-4` | **13.85%** |
| ViT-L/16 | 13.458181 → 13.169330 | 755.683920 → 674.241552 | `3.802895e-3 → 3.320221e-3` | **12.69%** |

### CNN 减少了跨 K 波次的部分和搬运

两款 CNN 的代表层 `T1` 都有 `K=576` 和 `N=64`。对照组的获胜路径在这层用 512 行 Macro 独自执行，第一波覆盖 512 行，第二波覆盖其余 64 行。启用协作后，两侧的 `jobs` 仍为 `[8]`，但 64 行 Macro 在同一 lane 补齐剩余的 64 行，一波覆盖 `K=576`。这一层的跨波次部分和读写因此降为 0。 [ResNet-18 配对数据](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/resnet18) · [VGG-16 配对数据](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/vgg16)

| 负载 | `T1` 延迟（ms） | 整网部分和传输量（bit，每方向） | 每方向降幅 |
|---|---:|---:|---:|
| ResNet-18 | 1.869184 → 0.958784 | 106,480,640 → 60,720,128 | 42.98% |
| VGG-16 | 29.098624 → 14.619200 | 856,504,064 → 525,743,872 | 38.62% |

启用组的最佳路径分别在 ResNet-18 的 13/21 层和 VGG-16 的 12/16 层使用同 lane 异构协作。两侧每个网络的权重传输位数相同，分别为 93,431,296 和 1,106,753,024 bit。 [ResNet-18 启用组账本](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/resnet18/mixed_lanes/result.json) · [VGG-16 启用组账本](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/vgg16/mixed_lanes/result.json)

### ViT 的两种任务在不同 Macro 上重叠

两款 ViT 的串行对照获胜调度都用 512 行 Macro 执行全部 CIM 算子。分区获胜调度把 QKᵀ 和 AV 移到 64 行 Macro，512 行 Macro 继续负责投影及 FFN。整网轨迹显示，一个 head 的注意力和下一 head 的 Q/K/V 投影出现重叠。按带 VMM 的批事件占用区间统计，串行对照均为 0，分区后 ViT-B/16 和 ViT-L/16 分别重叠 **8,941,020** 和 **22,934,160** 周期。批事件包含供数、计算和写回的流水阶段，单列的权重写入事件未计入这项重叠统计。 [ViT-B/16 获胜轨迹](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/vit_b16/partitioned/best_trace.json) · [ViT-L/16 获胜轨迹](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/vit_l16/partitioned/best_trace.json)

分区后整网延迟分别下降 **11.61%** 和 **10.78%**。两组最优策略对注意力 Macro 的分配也不同，因此上述延迟降幅是分区映射的总效果。轨迹里的重叠周期只说明哪些活动同时发生，整网时间以实际调度为准。 [ViT-B/16 配对数据](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/vit_b16) · [ViT-L/16 配对数据](../../results/tile_DSE/v5/mapping_ablation_64_512_20261124/vit_l16)

## 复现

从仓库根目录在 `cim-dse` 环境中运行。每个案例写出含负载和源码 SHA-256 的 `manifest.json`、验证结果及产物 SHA-256。CNN 保留每层回放账本，ViT 保存获胜整网的完整事件和轨迹。以下命令用新的输出目录。 [环境配置](../../ENVIRONMENT.md) · [独立消融入口](../../RUN/tile_DSE/v5/run_mapping_ablation.py)

```bash
V5=experiments/RUN/tile_DSE/v5
OUTPUT=experiments/results/tile_DSE/v5/mapping_ablation_64_512_new

for workload in resnet18 vgg16; do
  for variant in single_type_lanes mixed_lanes; do
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
      python "$V5/local_runtime.py" "$V5/run_mapping_ablation.py" \
      --workload "$workload" --variant "$variant" \
      --output-dir "$OUTPUT/$workload/$variant"
  done
done

for workload in vit_b16 vit_l16; do
  for variant in serial partitioned; do
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
      python "$V5/local_runtime.py" "$V5/run_mapping_ablation.py" \
      --workload "$workload" --variant "$variant" \
      --output-dir "$OUTPUT/$workload/$variant"
  done
done
```

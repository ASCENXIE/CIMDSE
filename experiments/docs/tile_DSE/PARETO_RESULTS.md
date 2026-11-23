# 四网 Tile Pareto 复核

`64+512` 在共同实测的 48 个 Tile 中取得最低四网联合 EDAP 分数 **1.054288**。该 Tile 在四网均命中 E–T–A 前沿，四网中最大的 E–T 前沿差距为 0.2306%。 [实验规范](TILE_DSE_EXPERIMENT_SPECIFICATION_V5.md#6-指标与通过标准) · [联合排名](JOINT_SEARCH_RESULTS.md)

## 逐网判定

| 网络 | 纳入的整网调度数 | E–T 前沿调度数 / Tile 数 | E–T–A 前沿调度数 / Tile 数 | `64+512` 命中的前沿 | `r_ET` | `r_ETA` | EDAP 超出该网最低值 |
|---|---:|---:|---:|---|---:|---:|---:|
| ResNet-18 | 541 | 53 / 14 | 72 / 19 | E–T、E–T–A | 0.0000% | 0.0000% | 0.0000% |
| ViT-B/16 | 48 | 5 / 5 | 8 / 8 | E–T–A | 0.2306% | 0.0000% | 18.3441% |
| VGG-16 | 227 | 33 / 11 | 49 / 16 | E–T、E–T–A | 0.0000% | 0.0000% | 0.0000% |
| ViT-L/16 | 48 | 3 / 3 | 7 / 7 | E–T–A | 0.1986% | 0.0000% | 4.3976% |

- 四网均有 `r = max(r_ET, r_ETA) ≤ 5%`，近优率为 100%，中位 `r` 为 0.0993%，最坏 `r` 为 0.2306%，严格命中至少一个前沿的比例为 100%。判定使用未舍入值。 [逐网判据与最近前沿点](../../results/tile_DSE/v5/pareto_review_20261120/summary.json)
- 两个 ViT 的最近 E–T 前沿点均为 `128+512`。`64+512` 相对它在 ViT-B/16 多用 0.1219% 能量和 0.2306% 时间，在 ViT-L/16 多用 0.1050% 能量和 0.1986% 时间，本地面积少 9.7106%。 [逐调度成本和前沿标记](../../results/tile_DSE/v5/pareto_review_20261120/points.csv)
- ViT-B/16 的最低 EDAP 来自 `2×256`。`64+512` 相对它节能 1.3396%，延迟增加 9.1944%，本地面积增加 9.8508%，三项相乘使 EDAP 增加 18.3441%。 [开发网排名 CSV](../../results/tile_DSE/v5/e04_development48_joint_ranking_20261118/joint_ranking.csv)

## 调度取证与适用范围

复核对象是已结案的 ResNet-18、ViT-B/16、VGG-16、ViT-L/16 共同面板。审计匹配了两张排名表中的 48 个设计、192 条通过 600 秒准入的记录，并核对结果行哈希、搜索摘要和回放摘要哈希。CNN 的 79 份归档搜索文件从 7 个归档流式读取，逐份核对完整文件的 SHA-256。 [审计采用的规范快照](../../results/tile_DSE/v5/pareto_review_20261120/specification_snapshot.txt) · [审计脚本](../../results/tile_DSE/v5/pareto_review_20261120/audit_pareto.py)

CNN 从已保留的整网聚合前沿取调度。每条路径必须为所有层找到同映射的完整空间回放，整网能量和延迟分别按同一路径的逐层实测复算，面积取该 Tile 的本地面积。ResNet-18 有 8,265 条、VGG-16 有 5,013 条聚合路径至少一层缺少完整回放，均排除。ViT 使用各 Tile 的完整事件图入围摘要，两网各 48 条，均与获胜回放匹配。`64+512` 各网各有一条符合上述证据的保留调度。判定按相对 `1e-9` 支配容差保留并列调度身份，覆盖已保留且有回放证据的调度。 [逐调度成本和前沿标记](../../results/tile_DSE/v5/pareto_review_20261120/points.csv)

已有审计的规范哈希对应上方快照。从项目根目录运行下方命令会重写审计脚本所在目录的四网联合排名、逐调度成本和 Pareto 摘要，并在新摘要中记录工作区规范的哈希。

```bash
PYTHONPATH=experiments/RUN /home/ascen/anaconda3/envs/cim-dse/bin/python -B \
  experiments/results/tile_DSE/v5/pareto_review_20261120/audit_pareto.py
```

## CNN 行缓冲核验

`64+512` 最低 EDAP 路径中的行缓冲峰值需求为 ResNet-18 的 7,168 B（T1–T5）和 VGG-16 的 28,672 B（T1）。当前回放计供数时序和 GB、总线搬运。行缓冲本体成本标记为 `not_characterized_not_in_objective`。 [规范的 CNN 数据流](TILE_DSE_EXPERIMENT_SPECIFICATION_V5.md#4-cnn映射)

现有批量评价器让各输入行的源事件依次占用 `line_buffer_source`，投影消费等待本组源事件。源事件之间的依赖只约束链路顺序，缺少对同一缓冲字最后一次读取和后续覆盖的检查。用 8×8×8 输入、3×3 卷积、padding 1、单个 128 行 Macro 复现时，调度给行缓冲计 128 B，整张输入的行源传输在周期 0–8 完成，首个投影事件到周期 72 才开始，现有 `validate_trace` 仍通过。此轨迹允许两行容量之外的数据提前进入，证明资源校验尚缺缓冲内容生命周期约束。 [批量评价器](../../RUN/tile_DSE/v5/batch_evaluator.py#L200) · [空间扫描统计](../../RUN/tile_DSE/v5/geometry.py#L46)

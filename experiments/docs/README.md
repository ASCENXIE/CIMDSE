# Tile DSE 实验文档

本目录汇总 SRAM-CIM Tile 的实验结论、评价规范和成本模型。已结案面板在 ResNet-18、ViT-B/16、VGG-16 和 ViT-L/16 上完成了同一组 48 个 Tile 的 192 项整网模型评价。按四网等权的归一化 EDAP 排名，`64+512` 以 **1.054288** 居首。 [实验结论](tile_DSE/EXPERIMENT_CONCLUSION.md)

## 实验结果

| 文档 | 内容 |
|---|---|
| [四网实验结论](tile_DSE/EXPERIMENT_CONCLUSION.md) | `64+512` 的联合排名、逐网成本、收益与代价，以及结论的适用范围。 |
| [联合搜索结果](tile_DSE/JOINT_SEARCH_RESULTS.md) | 四网完整排名 CSV、两组分项排名、归一化口径和终态审计。 |
| [Pareto 复核](tile_DSE/PARETO_RESULTS.md) | E–T 和 E–T–A 前沿、逐网差距、回放证据和模型边界。 |

## 规范与方法

| 文档 | 内容 |
|---|---|
| [现行实验规范](tile_DSE/TILE_DSE_EXPERIMENT_SPECIFICATION_V5.md) | 硬件约束、合法映射、能量与面积口径，以及四网联合 EDAP 选型判据。 |
| [搜索方法](tile_DSE/SEARCH_METHOD.md) | 候选选点、CNN 和 ViT 的整网搜索、回放验证和预算。 |
| [Macro 表征](SRAM_DATA/MACRO_CHARACTERIZATION_2026-11-16.md) | Macro 面积、能量、部分行模型和访存计费。原始表征数据见 [Macro 参数 CSV](SRAM_DATA/cim_macro_efficiency.csv)。 |

## 复现与项目记录

- [环境配置](../ENVIRONMENT.md)与[运行命令](../RUN/tile_DSE/v5/README.md)用于配置依赖、执行搜索和验证结果。
- [结果目录说明](../results/README.md)提供整网评价、排名、审计和归档恢复入口。
- 后续实验及性能回归要求见[交接记录](../../HANDOFF.md)。研究与写作资料见[论文目录](../../paper/README.md)。

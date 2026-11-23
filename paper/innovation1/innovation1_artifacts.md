# 创新点一图表索引

## 四网结果与当前 Macro 表征

| 用途 | 图表或数据 |
|---|---|
| 四网联合排名与选型解释 | [四网实验结论](../../experiments/docs/tile_DSE/EXPERIMENT_CONCLUSION.md) · [联合排名 CSV](../../experiments/results/tile_DSE/v5/pareto_review_20261120/joint_ranking.csv) |
| 四网 E–T 与 E–T–A 前沿 | [Pareto 复核](../../experiments/docs/tile_DSE/PARETO_RESULTS.md) · [逐调度成本及前沿标记](../../experiments/results/tile_DSE/v5/pareto_review_20261120/points.csv) |
| 250 MHz Macro 能效与面效 | [PDF](../figures/innovation1/macro_efficiency_vs_size.pdf) · [PNG](../figures/innovation1/macro_efficiency_vs_size.png) · [论文表格 CSV](../tables/innovation1_macro_characterization.csv) · [原始表征数据](../../experiments/docs/SRAM_DATA/cim_macro_efficiency.csv) |

在项目根目录的 `cim-dse` 环境中运行以下命令，可生成 Macro 表格和两种图像格式。`--figure-dir` 和 `--table-dir` 可指定输出目录。 [生成脚本](../../experiments/RUN/tile_DSE/generate_innovation1_experiments.py) · [环境配置](../../experiments/ENVIRONMENT.md)

```bash
python experiments/RUN/tile_DSE/generate_innovation1_experiments.py
```

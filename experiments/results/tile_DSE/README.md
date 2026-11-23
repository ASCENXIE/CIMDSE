# Tile DSE 结果索引

当前共同面板覆盖 ResNet-18、ViT-B/16、VGG-16 和 ViT-L/16，四网使用同一组 48 个 Tile。 [实验结论](../../docs/tile_DSE/EXPERIMENT_CONCLUSION.md)

| 结果 | 文件 |
|---|---|
| 四网联合排名与 Pareto 数据 | [48 点排名](v5/pareto_review_20261120/joint_ranking.csv) · [调度与前沿](v5/pareto_review_20261120/points.csv) · [审计摘要](v5/pareto_review_20261120/summary.json) · [审计采用的规范快照](v5/pareto_review_20261120/specification_snapshot.txt) |
| ResNet-18 / ViT-B/16 | [两网排名](v5/e04_development48_joint_ranking_20261118/joint_ranking.csv) · [首轮终态](v5/e04_development48_250mhz_bus4_20261117/terminal_audit.json) · [补测审计](v5/timeout_recovery_20261118/recovery_audit.json) |
| VGG-16 / ViT-L/16 | [两网排名](v5/timeout_retest_vgg16_round6_20261119_203209/joint_ranking.csv) · [终态与补测审计](v5/timeout_retest_vgg16_round6_20261119_203209/retest-audit.json) |

排名和审计文件属于本地实验产物。运行来源、准入记录与复现命令可从[联合搜索结果](../../docs/tile_DSE/JOINT_SEARCH_RESULTS.md)和 [Pareto 复核](../../docs/tile_DSE/PARETO_RESULTS.md)进入。归档恢复见[结果目录说明](../README.md)。

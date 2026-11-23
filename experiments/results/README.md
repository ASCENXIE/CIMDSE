# DSE 结果目录

- [Tile DSE 结果索引](tile_DSE/README.md)列出共同 48 点面板、两组原始排名及对应审计文件。
- [开发组明细恢复](maintenance/cleanup_20261119_cold_data/README.md)提供 ResNet-18 / ViT-B/16 原批次和补测的归档索引与恢复命令。
- [固定候选组明细恢复](maintenance/cleanup_20261119_closed_panels/README.md)提供 VGG-16 / ViT-L/16 批次与补测的归档索引与恢复命令。[ViT-L/16 事件与轨迹](maintenance/archive_vit_l16_traces_20261119/README.md)另有按 Tile 保存的归档。

查看排名、终态或回放摘要可直接使用索引中的文件。需按原路径读取归档明细时，按对应 README 恢复所需成员。归档索引和运行状态记录原路径与哈希，供恢复和审计校验。

请为新运行使用独立目录，保存来源 `manifest.json`、状态、评价与验证证据。运行输出和本地归档默认由 Git 忽略。 [运行命令](../RUN/tile_DSE/v5/README.md)

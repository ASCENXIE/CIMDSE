# Tile DSE 待办

更新于北京时间 2026-11-20。

## 待完成

后续实验由用户指定。

## 后续优化的回归验收

- **每次搜索性能优化后复测 `20×32 × ResNet-18`。** 使用当前源码冻结和新输出目录，采用共同控制器运行一个候选任务、四个层进程。600 秒端到端时限包含搜索、21 层所选路径完整回放、独立校验和产物哈希。 [运行命令](experiments/RUN/tile_DSE/v5/README.md#整网优化验收) · [优化技能](.agents/skills/dse-search-optimization/SKILL.md)

# 创新点一：归约粒度异构的 SRAM-CIM Tile 及工作负载驱动探索

本文概括归约粒度异构的硬件组织，并结合四网 Tile DSE 解释 `64+512` 的选型。 [现行实验规范](../../experiments/docs/tile_DSE/TILE_DSE_EXPERIMENT_SPECIFICATION_V5.md) · [四网实验结论](../../experiments/docs/tile_DSE/EXPERIMENT_CONCLUSION.md)

## 核心主张

> 面向 CNN 与 Transformer 算子不同的归约维度，在同一 Tile 内组合不同计算行数的全数字 SRAM-CIM 宏，并通过工作负载驱动的组合—映射联合探索，在宏本征能效、有效行利用率及部分和归约代价之间取得平衡。

该贡献包括两个相互支撑的部分：**提出怎样的硬件组织，以及根据工作负载如何选择这种组织。**

## 1. 归约粒度异构的数字 SRAM-CIM Tile

### 问题

对于点积或矩阵乘法，K 表示参与一次输出归约的元素数量。不同网络、不同算子的 K 差异明显：

- 大行数宏能够摊薄外围开销，但处理短归约时可能留下较多无效行。
- 小行数宏更容易匹配短归约，但处理长归约时需要更多宏或归约波次，并增加外围及部分和处理成本。

因此，单独追求宏的峰值能效或行利用率，都不足以确定系统所需的宏组织。

### 机制

在同一计算 Tile 内集成不同计算行数的 SRAM-CIM Macro。`64+512` 各安装一个 64 行和一个 512 行 Macro。

- 小、大宏具有不同的实际点积归约粒度。
- 宏输出可以独立使用，也可以局部组合部分和；例如共同覆盖最多 576 行的归约。
- 根据算子维度和执行模式，配置宏的分组及归约方式。

**关键区别是实际计算行数不同。** 宏容量和面积的变化是相关属性，不能仅用“不同大小的 SRAM”概括这一机制。

## 2. 工作负载驱动的宏组合—映射联合探索

### 需要回答的问题

在统一算力窗口、接口和成本模型下，比较 `64+512` 与同构及其他异构候选，解释 Macro 组成和映射如何影响整网能量、延迟与本地面积。

### 现行探索流程

| 层次 | 决策内容 |
|---|---|
| 宏组合 | 从 32、64、128、256、512 行 Macro 中选择实例数和至多两种尺寸，峰值算力限制在 0.288 TOPS ±20% |
| 算子映射 | 搜索 Macro 分组、K 维归约波次、输出通道划分及 ViT 投影与注意力的分工，按本地资源约束评价调度 |
| 工作负载评价 | 用同一条通过验证的调度计算整网能量、延迟与本地面积，按各网最低 EDAP 归一化后取四网等权几何平均 |

外层用 ResNet-18 和 ViT-B/16 的联合评价驱动贝叶斯选点。同一组 48 个候选随后用于 VGG-16 和 ViT-L/16。内层在有界空间内搜索合法调度，由整网回放及校验提供入表证据。 [搜索方法](../../experiments/docs/tile_DSE/SEARCH_METHOD.md) · [联合搜索结果](../../experiments/docs/tile_DSE/JOINT_SEARCH_RESULTS.md)

**这里的贡献重点是针对归约粒度异构建立组合与映射相互关联的决策空间。** 工作负载驱动 DSE、贝叶斯优化和动态规划本身均不应作为首次提出的概念。

`64+512` 的四网联合 EDAP 为 **1.054288**，在共同评价的 48 个候选中排名第一。排名适用于该面板的四个网络、成本模型和已验证调度。 [四网实验结论](../../experiments/docs/tile_DSE/EXPERIMENT_CONCLUSION.md)

固定 `64+512` 进行映射消融时，开放同 lane 异构归约使 ResNet-18 和 VGG-16 的 EDAP 分别下降 **28.37%** 和 **22.34%**。固定 `Bq=Bk=192`，允许按角色分区使 ViT-B/16 和 ViT-L/16 的 EDAP 分别下降 **13.85%** 和 **12.69%**。这两组对照量化了相同物理资源上允许不同映射方式的整网收益。 [消融协议、整网数据和轨迹证据](../../experiments/docs/tile_DSE/MAPPING_ABLATION_64_512.md)

## 3. 与 AIG-CIM 的区别

依据 [AIG-CIM 全文](../reference/AIG-CIM.pdf)，重点为第3页 §3.1–3.2、第4页 §3.3、第5页 §5.1及第6页 Table 2。

### AIG-CIM 已经实现的内容

AIG-CIM 在同一 Tile 中组合 E/B/P 三类数字6T SRAM-CIM宏。Table 2给出的实现参数如下：

| 参数 | E-Macro | B-Macro | P-Macro |
|---|---:|---:|---:|
| 宏容量 | 32 Kb | 128 Kb | 512 Kb |
| 宏面积 | 0.038 mm² | 0.076 mm² | 0.226 mm² |
| 输入通道数 | 16 | 16 | 16 |
| 输出通道数 | 32 | 32 | 32 |
| 吞吐 | 64 GOPS | 64 GOPS | 64 GOPS |
| 存储—计算比 | 8 | 32 | 128 |

它通过调整每个计算子阵列能够存储的权重组数，改变存储容量与计算资源的比例。三类宏具有相同的输入输出通道数和吞吐，但容量、面积与资源密度不同。

AIG-CIM 也已经进行工作负载驱动的异构组合探索：搜索宏档位、存储—计算比、数量比例、Tile规模及映射，最终选择6个P、14个B和44个E宏。其目标主要是匹配 Diffusion U-Net 各层不同的权重容量需求、复用率和流水负载。

### 本工作的区分点

| 维度 | AIG-CIM | 本工作 |
|---|---|---|
| 异构轴 | 存储—计算比 | 实际计算归约粒度 |
| 主要变化参数 | 每组计算资源配套的权重存储量 | 宏参与点积的计算行数 |
| 计算输入宽度 | 三类宏保持相同 | 64-row与512-row不同 |
| 主要匹配因素 | 权重容量、复用率、各层计算强度 | K维度、行利用率、部分和归约成本 |
| 组合的主要作用 | 用不同存储/计算密度承载相应层任务 | 由不同粒度宏独立或协同完成算子归约 |
| 探索侧重点 | 宏档位及资源配比与层映射 | 宏尺寸/数量与分组、归约波次、输出划分的联合选择 |

可概括为：**AIG-CIM 在相同计算宽度下调整权重存储深度；本工作调整点积计算宽度，以匹配不同长度的归约。**

所以，AIG-CIM 是“异构 SRAM-CIM 组合”和“工作负载驱动组合探索”的直接相关工作，不能声称它没有这些机制。本工作的差异应落在计算归约粒度及其组合—映射问题上。

## 4. 与 Harmony 的区别

依据 [Harmony 全文](../reference/ISQED.pdf)，重点为 §III-B Hardware Template、§III-C Hardware Feature及 §IV Algorithm。

Harmony 的硬件配置表示为 V=[A1,A2,D,L]：A1、A2为两类模拟RRAM-CIM配置，D为数字SRAM-CIM配置，L为静态权重层到RRAM宏类型的分配。其明确提出的两类尺寸异构发生在RRAM侧；数字SRAM路径主要承担需要频繁写入的动态矩阵计算。

Harmony 已进行工作负载驱动的硬件—映射联合探索，采用知识引导网格搜索及改进遗传算法，考虑宏参数和空间映射。它是方法层面的相关工作，不能只因介质不同而忽略。

| 维度 | Harmony | 本工作 |
|---|---|---|
| 计算介质 | 模拟RRAM与数字SRAM混合 | 全数字SRAM |
| 明确的尺寸异构 | 两类RRAM配置 | 同一Tile内不同计算行数的SRAM宏 |
| 主要映射问题 | 静态/动态权重分工、宏配置与NoC空间映射 | K维度与异构宏分组、归约方式的匹配 |
| 主要方法区别 | 搜索宏参数与空间布局/层映射 | 搜索宏组合及其算子归约映射 |

本工作相对 Harmony 更明确的架构区分是：**在全数字SRAM体系中，以Tile内部不同计算行数的宏协同处理归约。** 方法层面则需强调所建模的决策变量和代价，而非泛称“首次工作负载感知探索”。

## 5. 建议用于论文的贡献表述

### 中文

> 提出一种归约粒度异构的全数字 SRAM-CIM Tile，通过组合不同计算行数的宏，兼顾多样归约维度下的宏本征能效与有效行利用率。进一步建立工作负载驱动的宏组合—映射联合探索方法，在统一算力窗口和 Tile 接口约束下，联合选择宏尺寸、数量、分组及归约波次，并计入部分和累加与搬运成本。

### 英文

> We propose a reduction-granularity-heterogeneous digital SRAM-CIM tile that combines macros with different compute-row counts to balance intrinsic macro efficiency and useful row utilization across diverse reduction dimensions. A workload-driven composition–mapping exploration jointly selects macro sizes, counts, groupings, and reduction waves within a common peak-throughput range and tile interface constraints, accounting for partial-sum accumulation and data movement costs.

### 结论的适用范围

- 架构贡献聚焦实际计算归约粒度的异构及其组合与映射问题。相关工作中的异构组织和搜索方法见上文对照。
- 当前 EDAP 采用本地 Tile 面积，能量包含权重写入、总线和 GB 动态搬运。行缓冲本体成本、GB 与控制及互联的完整面积，以及非 CIM 算子的成本属于模型边界。 [实验规范第 6 节](../../experiments/docs/tile_DSE/TILE_DSE_EXPERIMENT_SPECIFICATION_V5.md#6-指标与通过标准) · [CNN 行缓冲核验](../../experiments/docs/tile_DSE/PARETO_RESULTS.md#cnn-行缓冲核验)

两个子贡献的关系是：**硬件组织提供不同归约粒度的可组合资源，探索方法决定这些资源怎样配置和使用。** 二者共同构成创新点一，而不是两个互不关联的优化。

## 来源

- [AIG-CIM，DAC 2024](../reference/AIG-CIM.pdf)：§3.1–3.3、§5.1–5.2、Table 2。
- [Harmony](../reference/ISQED.pdf)：§III-B、III-C及§IV。

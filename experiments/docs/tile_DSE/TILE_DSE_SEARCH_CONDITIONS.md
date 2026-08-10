# Tile 级 DSE 搜索条件与冻结规范

> 版本：Tile Search Specification v2  
> 冻结日期：2026-07-25  
> 适用目录：`experiments/RUN/tile_DSE/`  
> 目标结果目录：`experiments/results/tile_DSE/`  
> 目标结果前缀：`tile_window_dse_*`  
> 状态：搜索限制和代码模型已经闭环；2026-07-26 已完成正式 48 点 Bayesian 搜索。

## 1. 研究问题与边界

当前 DSE 只评价一个 SRAM-CIM Tile，回答以下问题：

> 在 22 nm、400 MHz、8-bit 输入和 8-bit 权重条件下，以 0.4608 TOPS 为中心、允许
> 峰值算力偏差 `+-20%` 时，哪些同构或最多双类型异构 Macro 组合能形成完整
> ResNet-18 的能量-延迟 Pareto？

评估对象不乘以物理 Tile 数，不包含 Core/System 并行、GLB、NoC、DRAM 和非 CIM
算子。结果只能解释为给定算力窗口内的 Tile CIM fabric 架构选择，不能单独证明异构
组成在严格等算力下必然更优。

### 1.1 长期研究假设

本项 Tile DSE 的核心目的是验证：

> `1xm64x8+1xm512x8` 是否能在多种网络负载下，稳定成为能量-延迟或
> 能量-延迟-面积 Pareto 最优组合之一。

这是待检验假设，不是通过搜索约束预先保证的结论。后续多负载 DSE 必须：

- 对每个网络使用相同的候选空间、硬约束和能量/时序模型；
- 将 `1xm64x8+1xm512x8` 作为必测锚点，但不给予 Pareto 剪枝或目标函数优待；
- 同时报告其在每个网络上的 Pareto 成员资格、到前沿的归一化 regret 和面积；
- 与等算力 `9xm64x8` 及全局竞争者（当前包括 `1xm128x8+1xm512x8`）比较；
- 在启动多网络搜索前冻结“最优之一”的判定阈值和跨网络聚合方式。

当前 ResNet-18 结果只是该假设的第一个负载证据，不足以单独支持
“面向各种网络负载”的结论。

### 1.2 “最优组合之一”的冻结判定阈值

对网络负载 `w`、目标集 `Q` 和任意一个已实际评价的 Tile 候选 `c`，定义乘法型
Pareto regret：

\[
r_Q(c,w)=
\min_{s\in S_c}
\min_{f\in F_Q(w)}
\max_{q\in Q}
\max\left(0,\frac{s_q}{f_q}-1\right)
\]

其中：

- `S_c` 是候选 `c` 在该网络上的全部合法聚合调度；
- `F_ET(w)` 是同一组 48 个实测 Tile 候选的能量-延迟前沿；
- `F_ETA(w)` 是同一组 48 个实测 Tile 候选的能量-延迟-面积前沿；
- 能量、延迟和面积均为越小越好的正值目标；
- `r_Q(c,w)=0` 表示候选 `c` 严格命中相应 Pareto 前沿。

每个网络的联合 regret 定义为：

\[
r(c,w)=\max\left(r_{ET}(c,w),r_{ETA}(c,w)\right)
\]

判定阈值冻结为：

| 判定项 | 冻结阈值 |
|---|---:|
| 单网络近优通过 | `r(c,w) <= 5%` |
| 跨网络通过率 | `>= 80%` |
| 跨网络中位 `r(c,w)` | `<= 3%` |
| 最坏网络 `r(c,w)` | `<= 10%` |
| 严格命中 E-T 或 E-T-A Pareto 的网络比例 | `>= 50%` |

聚合和报告规则：

- 各网络等权，不按 MAC 数、层数或运行时间加权；
- 同时报告 `r_ET`、`r_ETA`、`r`和严格 Pareto 命中，不只报告通过/失败；
- 138 个理论候选不做完整评价；跨网络统计固定使用 Bayesian 优化选出的同一组
  48 个实测候选，每个候选必须在每个纳入统计的网络上完成实际评价；
- GP 只负责选择这 48 个候选，未实际评价候选的预测值不得进入 Pareto、regret 或
  最终合格名单；
- 最终只输出从完整 `0.36864` 到 `0.55296 TOPS` 算力窗口中采样的 48 个实测候选里，
  满足全部跨网络阈值的 Tile 组合，不单独输出严格 `0.4608 TOPS` 名单；
- 严格 `0.4608 TOPS` 子空间仅保留为 `64+512` 与等算力组合的内部诊断，
  不作为全窗口候选的淘汰条件；
- 最终结论必须表述为“Bayesian 实测 48 候选中的稳健近优组合”，不能外推为
  138 个理论候选的穷举全局最优；
- 阈值从本版规范开始冻结，不得根据后续网络结果事后放宽。若能量、时序或
  候选空间发生版本变更，必须建立新实验版本，不覆盖本阈值的结果。

按该定义，当前 ResNet-18 的参考结果为：

```text
r_ET                          = 1.861618%
r_ETA                         = 0%
r                             = 1.861618%  -> 单网络通过
exact E-T Pareto hit          = no
exact E-T-A Pareto hit        = yes
0.4608-TOPS exact E-T hit     = yes
0.4608-TOPS exact E-T-A hit   = yes
```

重点对照设计为：

```text
9xm64x8                 = 0.4608 TOPS
1xm64x8 + 1xm512x8     = 0.4608 TOPS
```

### 1.3 调度搜索实现版本

正式多网络结果必须记录并匹配 `schedule_search_version`。当前版本为：

```text
reachable_state_bounded_timing_cache_v2
```

该版本不改变 Tile 候选、能量/延迟/面积模型或冻结阈值，只降低局部调度 DP 的无效
复杂度：

- 4 条输出 lane 已占满时，只有一次性完成全部剩余输出块的转移可达；
- 剩余 Macro 资源无法容纳任何后续分组时，未完成状态不可达；
- lane 超限和不可完成转移按类别批量剪枝，不逐输出块生成状态；
- 每层 timing cache 使用 131072 条目的 LRU 上限，淘汰只导致时序重算，不参与目标值
  或 Pareto 判定。

由于旧 4096 state cap 曾被不可完成状态占用，v2 会把配额用于更多有效状态，可能改变
局部 Pareto 结果。不同 `schedule_search_version` 的网络结果不得直接合并计算跨网络
regret；ResNet-18、ResNet-50 和 VGG16 必须统一用 v2 复评同一组 48 个 Tile。

## 2. 固定输入

### 2.1 工作负载

| 项目 | 固定值 |
|---|---:|
| 网络 | AccelForge ResNet-18 |
| Batch size | 1 |
| 计算层 | `T0` 到 `T20`，共 21 个 Conv/FC einsum |
| 网络 MAC 数 | 1,814,073,344 |
| 网络操作数 | 3,628,146,688，按 1 MAC = 2 operations |
| 层间执行 | 21 层顺序执行，不做跨层流水 |

每层统一表示为：

```text
Y[P,N] = X[P,K] * W[K,N]
P = Batch * Hout * Wout
K = Cin * R * S
N = Cout
MACs = P * K * N
N_blocks = ceil(N / 8)
```

当前不评价 ReLU、BN、pooling、residual add，也不建立完整激活生命周期。

### 2.2 工艺与 Macro 表征

| 项目 | 固定值 |
|---|---:|
| 工艺 | 22 nm |
| 电压 | 0.72 V |
| 频率 | 400 MHz |
| bit-cycle | 2.5 ns |
| 输入/权重精度 | 8 bit / 8 bit |
| Macro 输出列 | 8 |
| 输入串行速率 | 1 bit/cycle |
| 完整 VMM | 8 cycles，即 20 ns |

Macro 表来自 `experiments/docs/SRAM_DATA/cim_macro_efficiency.csv`：

| Macro | Rows x columns | 峰值算力 | 表征 VMM 能量 | Macro 面积 | 权重行写能量 |
|---|---:|---:|---:|---:|---:|
| `m32x8` | 32 x 8 | 0.0256 TOPS | 84.211 pJ | 0.011480 mm2 | 0.333 pJ |
| `m64x8` | 64 x 8 | 0.0512 TOPS | 133.681 pJ | 0.023273 mm2 | 0.549 pJ |
| `m128x8` | 128 x 8 | 0.1024 TOPS | 181.560 pJ | 0.044329 mm2 | 0.843 pJ |
| `m256x8` | 256 x 8 | 0.2048 TOPS | 311.957 pJ | 0.089825 mm2 | 1.009 pJ |
| `m512x8` | 512 x 8 | 0.4096 TOPS | 481.033 pJ | 0.182857 mm2 | 1.321 pJ |

表征 TOPS/W 已包含 Macro 工作状态下的静态功耗。面积和权重写能量分别由表征
TOPS/mm2 和 22 nm `SmartBufferSRAM` 模型获得。

## 3. Tile 架构硬约束

### 3.1 Macro 组成与算力窗口

一个 Tile 表示为：

```text
n = (n32, n64, n128, n256, n512)
```

必须满足：

```text
1 <= 非零 Macro 类型数 <= 2

0.36864 TOPS
  <= 0.0256*n32 + 0.0512*n64 + 0.1024*n128
     + 0.2048*n256 + 0.4096*n512
  <= 0.55296 TOPS
```

候选资格不使用独立容量或面积窗口，也不设置 Macro 实例数上限。由于 Macro 算力以
0.0256 TOPS 为粒度，实际候选只落在 0.384 到 0.5376 TOPS：

| 项目 | 数量/范围 |
|---|---:|
| 同构候选 | 14 |
| 双类型异构候选 | 124 |
| 候选池总数 | 138 |
| Macro 实例数 | 1 到 21 |
| 逻辑权重容量 | 3840 到 5376 values |

### 3.2 Lane 与 Macro 使用

- Tile 固定最多 4 条物理输出 lane；
- 每条 lane 对应 8 个输出寄存器和 8 个输出端口；
- 整个 Tile 固定有 32 个 25-bit 输出寄存器和 32 个输出端口；
- 一条 lane 可包含一个或多个 Macro，当前不设置 Macro fan-in 上限；
- 同一物理 Macro 在同一时刻只能属于一条 lane，并且最多执行一个 VMM；
- 允许物理 Macro 在某层闲置，闲置 Macro 动态功耗为 0，但仍支付静态功耗；
- Macro-to-lane 分组在一个网络层内保持固定；
- 不同层之间可以重新配置分组，暂不收取重配置能量和延迟。

### 3.3 输出寄存器与 psum

对一条 lane 执行一个 8-output block，若需要 `w` 个 K waves：

```text
output-register writes = 8 * w entries
output-register reads  = 8 * w entries
cross-wave adds        = 8 * (w - 1) scalar adds
```

其中读取事件包括 `w-1` 次跨 wave 反馈读取和 1 次最终输出读取。

同一 wave 内若有 `m` 个 Macro 共同覆盖 K：

```text
cross-Macro reductions = 8 * (m - 1) scalar adds
```

同 wave 多 Macro 归约和跨 wave 加法均计算动态能量，但其延迟被 VMM 周期完全掩盖，
不增加额外 reduction cycle。跨 wave psum 保存在 Tile 固定输出寄存器中，不转交 Core
缓存处理。

### 3.4 输入寄存器

- 每个 Macro 有两组 `rows x 8-bit` ping-pong 输入寄存器；
- 当前 VMM 使用一组时，另一组可接收后续特征；
- 每次实际 Macro VMM 对活动 bank 计算相应读写动态能量；
- 未活动 Macro 不产生输入寄存器动态访问。

## 4. 共享总线约束

### 4.1 物理接口

输入特征和权重采用互斥的 Tile 入口时序。特征模式带宽为512 bit/cycle，权重模式
带宽统一为64 bit/cycle。每个周期只有以下一种模式：

```text
bus_mode[t] in {FEATURE, WEIGHT, IDLE}
```

- `FEATURE` 周期只能传特征，不能利用剩余 bit 传权重；
- `WEIGHT` 周期只能传权重，不能利用剩余 bit 传特征；
- 特征和权重在单一周期内严格互斥；
- 不建立仲裁器模型，仲裁已经由上层解决；
- 不计算仲裁器的能量、面积和额外周期。

### 4.2 权重传输

- 一行权重为 `8 columns x 8 bit = 64 bit`；
- 一个 WEIGHT 周期只携带一个 Macro 的一行64-bit权重；
- 每个 Macro 本地端口每周期最多接收一行；
- 同一 bank 内禁止权重写入与 Macro VMM 同时发生；
- ViT 投影 bank 计算时允许注意力 bank 加载权重，反向同理；
- 一个 N-block 的 K 维权重装载一次，并在所有 `output_positions` 上复用；
- 每层总有效权重行数为 `K * ceil(N/8)`。

### 4.3 特征传输

- 同一 lane round 中，不同 lane 计算不同 N-block，但共享一次 K 维特征广播；
- 特征通过共享总线传入活动 Macro 的 ping-pong 输入寄存器；
- 若 N-block 数超过当前活动 lane 数，需要多个 lane rounds，特征按 round 重新传输；
- 特征传输可以与当前 VMM 重叠，但必须满足输入可用性和共享总线互斥约束。

### 4.4 总线延迟与能量口径

- 总线 FEATURE/WEIGHT 周期进入层延迟计算；
- 使用满足数据依赖的理想无仲裁开销调度；
- 总线传输允许与 Macro 计算重叠；
- 总线动态能量当前设为 0，不进入正式 Tile 能量目标；
- 总线能量与未显式建模的外围能量联合省略：假设大 Macro 的总线能量较高而外围开销
  较低，小 Macro 则相反，两者在候选排序中一阶抵消；
- 保留参数 `bus_energy_per_bit`，当前值为 `0 J/bit`，供后续优化；
- 必须输出 `feature_bus_bits`、`weight_bus_bits`、`feature_bus_cycles`、
  `weight_bus_cycles` 和 `total_bus_cycles`，以便后续补算总线能量。

“不考虑总线仲裁”只表示不添加仲裁开销，不表示忽略共享总线带宽、周期互斥或数据依赖。

### 4.5 YAML时序契约

| 配置字段 | 当前值 | 含义 |
|---|---:|---|
| `tile_retain_features_across_nblocks` | `False` | 不配置完整输入缓存，特征按 N-block 重新加载 |
| `tile_allow_intra_bank_weight_compute_overlap` | `False` | 同一 bank 禁止权重加载与计算重叠 |
| `vit_allow_cross_bank_weight_compute_overlap` | `True` | ViT 不同 bank 允许一侧计算、另一侧加载权重 |
| `tile_latency_batch_granularity` | `nblock_kwave` | 按 N-block/K-wave 累加批次延迟 |
| `tile_include_pipeline_fill_drain` | `False` | 不加入首次填充和末尾排空 |

加载器只接受上述已实现组合；设置为其他值会直接报错，而不会静默使用不匹配的模型。

## 5. 能量模型

### 5.1 Macro 动态与静态功耗拆分

Macro 表征 TOPS/W 包含工作状态静态功耗。当前统一假设静态功耗占表征活动功耗的 5%。
对 Macro 类型 `q`：

```text
E_char_vmm[q] = 由表征 TOPS/W 反推的完整 VMM 总能量
T_vmm[q]      = 20 ns

E_dynamic_vmm[q] = 0.95 * E_char_vmm[q]
P_static[q]      = 0.05 * E_char_vmm[q] / T_vmm[q]
```

当一次纯数字 VMM 只开启 `r` 行时，固定开销和单行开销都必须与物理 Macro 类型 `q`
绑定，不能让所有 Macro 共用一组全局系数：

```text
E_dynamic_partial(q, r)
  = E_fixed[q] + E_rows[q] * r
```

`E_fixed[q]` 表示类型 `q` 的一次有效 bit-serial VMM 固定数字开销，`E_rows[q]` 表示
该物理规模下每个活动行带来的 SRAM 访问和数字计算开销。Macro 深度会改变字线/位线
负载、列外围和数字归约结构，因此两个系数都属于 Macro 类型参数。

除 `m512x8` 外，当前每种 Macro 只有一个完整行 VMM 表征点，无法为每种类型独立识别
两个自由参数。第一版使用完整动态能量随物理行数变化的局部斜率估算 `E_rows[q]`。
设 Macro 类型按行数
`R_0 < ... < R_n` 排列，`D_i = E_dynamic_vmm[R_i]`：

```text
E_rows[0] = (D_1 - D_0) / (R_1 - R_0)

E_rows[i] = (D_(i+1) - D_(i-1)) / (R_(i+1) - R_(i-1)),  0 < i < n

E_rows[n] = (D_n - D_(n-1)) / (R_n - R_(n-1))

E_fixed[i] = D_i - E_rows[i] * R_i
```

中间规模采用中心差分，最小和最大规模采用单侧差分。反求 `E_fixed` 可以保证：

```text
E_fixed[q] + E_rows[q] * R_q = E_dynamic_vmm[q]
```

即每组参数严格恢复对应 Macro 的完整动态 VMM 表征。该方法仍是由 Macro 家族完整行
表征推导的局部斜率近似。

`m512x8` 已获得同一 Macro 的部分行实测活动功耗：64、128、256、512 行分别为
5.112、5.431、6.069、7.345 mW，严格拟合为：

```text
P_measured(N) = 4.793 mW + 0.004984375 mW/row * N
```

搜索仍以原 TOPS/W 反推的满行动态能量 `456.981797 pJ` 为绝对基准，仅采用实测曲线的
固定项/行相关项比例。用 `456.981797 pJ / 7.345 mW` 归一化后：

```text
E_fixed[m512x8] = 298.204731 pJ
E_rows[m512x8]  = 0.310111456 pJ/row
```

这使满行动态能量保持不变，同时将512行时固定动态能量占比从29.70%提高至实测比例
65.26%。实测标定将原 `m512x8` 行斜率缩放为：

```text
slope_correction = 0.310111456 / 0.627430917
                 = 0.494255937
```

32、64、128和256 Macro尚无同类部分行实测。为了修正原家族模型普遍偏大的行相关能量，
这些类型暂时采用同一斜率修正因子，同时保留原有类型间的相对斜率，并重新计算
`E_fixed[q] = E_dynamic_vmm[q] - E_rows[q] * R_q`。因此所有类型仍严格恢复各自TOPS/W
给出的满行动态能量。该处理属于由512 Macro实测向其它规模的保守外推，不代表其它
Macro已经完成部分行实测；得到对应数据后应分别替换。

完整的分段定义为：

```text
E_dynamic(q, r) = 0,                                      r = 0
E_dynamic(q, r) = min(E_dynamic_vmm[q],
                      E_fixed[q] + E_rows[q] * r),        0 < r < R_q
E_dynamic(q, r) = E_dynamic_vmm[q],                       r = R_q
```

`min` 用于抵御表征噪声，保证部分行能量不高于同一 Macro 的完整动态能量。本设计是
纯数字 SRAM-CIM，不在该模型中加入 ADC、DAC、TIA 或模拟积分能量。部分行不会改变
8 个 bit-cycle 的完整 VMM 延迟。

由当前 8 列、8-bit 输入/权重 Macro 表征得到的目标参数为：

| Macro | `E_dynamic_vmm` | `E_fixed` | `E_rows` |
|---|---:|---:|---:|
| `m32x8` | 80.000000 pJ | 56.771261 pJ | 0.725898080 pJ/row |
| `m64x8` | 126.997389 pJ | 96.524115 pJ | 0.476144904 pJ/row |
| `m128x8` | 172.482270 pJ | 116.676789 pJ | 0.435980313 pJ/row |
| `m256x8` | 296.359482 pJ | 202.615762 pJ | 0.366186408 pJ/row |
| `m512x8` | 456.981797 pJ | 298.204731 pJ | 0.310111456 pJ/row |

配置中的 `macro_partial_vmm_fixed_energy` 和
`macro_partial_vmm_energy_per_active_row` 已放入每个 `macro_types.<name>` 条目，不再
作为所有类型共享的 `arch.variables` 全局值。

以 `m512x8` 为例：

```text
E_dynamic(512 active rows) = 0.95 * 481.033470 pJ
                          = 456.981797 pJ

E_dynamic(256 active rows) = 298.204731 + 0.310111456 * 256
                           = 377.593264 pJ

dynamic energy reduction   = 79.388533 pJ = 17.37%
```

两种情况均执行 20 ns，因此单个 `m512x8` 在这 20 ns 内的静态能量均为 24.051674 pJ。
包含这部分静态能量后，512 行和 256 行分别为 481.033470 pJ 和 401.644938 pJ，差值仍为
79.388533 pJ，总能量降幅为 16.50%。层级统计时，静态能量不是按 VMM 次数收费，而是按
所有物理 Macro 的静态功率乘以完整层延迟收费。

一个层的 Macro 能量为：

```text
E_macro_dynamic_layer
  = sum_over_invocations(E_dynamic(q, active_rows))

E_macro_static_layer
  = sum(physical_macro_count[q] * P_static[q]) * T_layer
```

所有物理 Macro 在完整层延迟 `T_layer` 内支付静态功耗，包括未参与该层计算的 Macro。
该拆分保证 Macro 连续满负载时不会重复计算表征中已有的静态功耗。

5% 只用于拆分 Macro 表征 TOPS/W，不代表输入/输出寄存器、共享总线、归约网络、控制器
和时钟网络的静态功耗已经被覆盖。

### 5.2 Tile 正式能量

```text
E_tile_layer = E_macro_dynamic_layer
             + E_macro_static_layer
             + E_input_register_dynamic
             + E_output_register_dynamic
             + E_cross_macro_reduction
             + E_cross_wave_accumulation
             + E_weight_write
```

其中：

- 输入寄存器按实际活动 Macro VMM 访问收费；
- 部分行 VMM 的输入寄存器读写和权重行写只按实际活动行数收费；
- 输出寄存器按 lane、N-block 和 K-wave 事件收费，而不是按物理 Macro 数收费；
- 同 wave 与跨 wave psum 均属于 Tile 动态能量；
- 每个有效权重行按对应 Macro 的 SRAM 行写能量收费；
- 总线动态能量当前不计，但流量和周期必须输出。

完整网络能量为 21 层 Tile 能量之和。

## 6. 延迟模型

层延迟由 Macro VMM、特征加载和权重写入共同决定：

- lane 内 K-wave 串行；
- 不同 lane 和不同物理 Macro 可并行；
- 同一 lane 顺序执行分配给它的 N-block；
- 权重在一个 N-block 的所有 `output_positions` 上复用；
- 同一 Macro 权重写入与 VMM 不重叠；
- 每个 N-block/K-wave 批次先完成权重加载；
- 同一 bank 禁止权重加载与 Macro 计算重叠；
- 当前批次权重加载完成后，该批次的特征加载和计算可以完全重叠；
- 同 wave 归约、跨 wave反馈加法、寄存器访问和输出端口不增加额外延迟；
- 层延迟为所有 N-block/K-wave 批次延迟之和；
- 不加入首次流水填充和末尾输出排空延迟；
- 完整网络延迟为 21 层延迟之和。

当前代码对一个完整 lane 分配分别计算：

```text
T_weight[b] = weight cycles of N-block/K-wave batch b * 2.5 ns
T_feature[b] = feature cycles of batch b * 2.5 ns
T_compute[b] = max(active-lane compute cycles in batch b) * 2.5 ns
```

当前公平比较模型将特征和权重的有效入口带宽分开计费：特征为
`512 bit/cycle`，权重为 `64 bit/cycle`。权重路径每周期只装载一行64-bit权重，
且同一 bank 在权重装载期间不能执行VMM。ViT 的投影与注意力使用不同 bank 时，允许
一侧计算而另一侧加载权重，但两侧仍共享互斥的 FEATURE/WEIGHT 入口时序。Tile不配置能够保存完整层输入特征的
大容量SRAM；特征由入口按 N-block/K-wave 的实际需求重复流入，并通过Macro输入
双缓冲寄存器与当前批次计算重叠。

```text
T_layer = sum_over_batches(
    T_weight[b] + max(T_feature[b], T_compute[b])
)
```

权重周期在同一 wave 的不同 Macro 和不同 lane 之间不再合并为8行；权重入口每周期只
传输一行64-bit权重。同一Macro每周期最多接收一行。`total_bus_cycles` 仍记录
`feature_bus_cycles + weight_bus_cycles`，用于流量和占用统计，但不直接作为层延迟；
层延迟严格使用上式。不同 lane 在同一 wave 所需 K 区间不一致时，特征入口传输这些
区间的并集；由于没有完整输入缓存，后续 wave 再次需要的区间会重新加载。该定义也保证
FEATURE 与 WEIGHT 不会被建模为同周期传输。

## 7. 层内调度搜索空间

局部 DP 对每个“Tile 设计 x 网络层”搜索：

- 每条 lane 使用哪些 Macro；
- 每条 lane 需要多少 K waves；
- 使用 1 到 4 条活动 lane；
- `N_blocks` 如何分配给这些 lane；
- 哪些物理 Macro 在该层闲置；
- 共享总线与 Macro 计算的理想依赖调度。

一个中间状态至少需要区分：

```text
已使用的各类 Macro 数量
已分配的 N-block 数量
活动 lane 分组及任务数
共享总线/数据依赖进度
累计动态能量
当前 makespan
```

对于相同完成进度，只有能量、延迟、Macro 资源和 lane 资源都不占优的状态才继续扩展。

## 8. 搜索近似参数与收敛检查

以下均为算法预算，不是硬件限制：

| 参数 | 正式值 | 收敛性对照 | 含义 |
|---|---:|---:|---|
| `latency_cap_ratio` | 4.0 | 不设 cap | 层调度不得慢于简单可行调度的 4 倍 |
| `local_state_cap` | 4096 | 8192 | 层内 DP 中间状态预算 |
| `local_pareto_epsilon` | 0.001 | 0 | 层内约 0.1% 的近似 Pareto 合并 |
| aggregate state cap | 512/Tile | 1024/Tile | 21 层组合中每个 Tile 的状态预算 |
| aggregate epsilon | 0.001 | 0 | 跨层状态压缩起始 epsilon |

### 8.1 `latency_cap_ratio`

若简单可行调度延迟为 `T_feasible`：

```text
T_local <= 4.0 * T_feasible
```

正式运行前，对代表 Tile 和复杂层执行不设 latency cap 的对照，以检查低能量慢方案是否
被错误删除。

### 8.2 `local_state_cap`

层内状态超过 4096 时，按已完成 N-block 进度、能量、延迟、lane 数和 Macro 使用量进行
确定性分层采样。输出必须记录 `local_schedule_state_cap_applied`。使用 8192 状态的代表
设计对照最低能量、最低延迟和 Pareto 成员变化。

### 8.3 收敛判据

建议将以下条件作为 4096/512 正式预算可接受的标准：

```text
代表测试中最低能量相对变化 < 1%
代表测试中最低延迟相对变化 < 1%
关键 Pareto 架构成员无实质变化
```

若不满足，应提高相应 state cap。即使通过，结果仍应称为 bounded-DP 近似 Pareto，
不能称为所有合法调度的精确枚举。

### 8.4 2026-07-25 收敛检查结果

代表层使用 ResNet-18 `T16 (K=4608, N=512, P=49)`：

| Tile | 对照 | 最低能量变化 | 最低延迟变化 |
|---|---|---:|---:|
| `9xm64x8` | 4096/0.001/4x 对 8192/0/no-cap | 0.088% | 0.725% |
| `1xm64x8+1xm512x8` | 4096/0.001/4x 对 8192/0/no-cap | 0.000% | 0.000% |

两项均低于 1% 判据。对异构基线执行 21 层聚合时，512 和 1024 state cap 的最低能量与
最低延迟完全一致，分别为 `0.265542268 mJ` 和 `12.925295 ms`；最终状态数分别为 263
和 277。对高压力层 `T15` 到 `T20`，aggregate epsilon 取 0.001 和 0 时均得到 462 个
状态，最低能量和最低延迟完全一致。

完整 21 层在 aggregate epsilon=0 时出现组合状态爆炸，未将其作为正式运行前置条件；
正式结果应继续标记为 bounded-DP 近似 Pareto，而不是精确调度枚举。

## 9. Bayesian 架构搜索

候选池含 138 个 Tile，当前不全量评价。正式预算为：

| 阶段 | 实际评价数 |
|---|---:|
| 初始覆盖 | 16 |
| Bayesian 采集 | 24 |
| 预测前沿验证 | 8 |
| 总实际评价预算 | 48 |
| Batch size | 4 |

初始 16 个候选必须包含：

- 全部 14 个同构 Tile；
- 指定异构基线 `1xm64x8+1xm512x8`；
- 特征空间中与已选集合距离最远的一个异构 Tile。

Bayesian 规则：

- GP 只用于选择下一批实际评价候选；
- 预测结果必须输出预测值和不确定度，并与实际评价数据分开；
- 预测前沿上进入候选推荐集合的设计必须补做真实局部 DP 评价；
- 最终正式 Pareto 只能由实际评价过的设计组成；
- 实际评价预算固定为48个设计，不因预测前沿状态继续扩展。

当前只统计整个 `+-20%` 算力窗口的全局结果，不按 7 个离散算力档位分别统计。报告中
必须同时给出每个设计的实际峰值 TOPS，避免把资源规模效应误写成异构组成效应。

## 10. 优化目标与面积处理

### 10.1 正式目标

主 Pareto：

```text
minimize E_network
minimize T_network
```

辅助架构 Pareto：

```text
minimize E_network
minimize T_network
minimize A_tile
```

面积不进入层内 DP，因为同一 Tile 的面积不随层调度变化；三维 E-T-A Pareto 在实际评价
结果上后处理生成。

### 10.2 面积口径

```text
A_tile = sum(A_macro)
       + sum(2 * rows * A_8bit_input_register)
       + 32 * A_25bit_output_register
```

输出寄存器面积固定为 32 entries，不随 Macro 数量变化。当前不包含共享总线、crossbar、
控制器、时钟树和物理布线面积。

### 10.3 派生指标

- EDP 只用于摘要和 Bayesian 采集，不作为最终独立 Pareto 维度；
- effective TOPS/W 为固定网络操作数除以完整网络 Tile 能量；
- TOPS/mm2、Macro 数、类型数、容量和利用率作为诊断字段。

## 11. 必需输出

### 11.1 逐层输出

至少包含：

- 架构：Macro 组成、容量、面积、峰值 TOPS、相对中心算力比；
- 映射：活动 lane、每 lane Macro 分组、N-block 数和 K-wave 数；
- Macro：实际调用数、活动/闲置时间、动态能量、静态功耗和静态能量；
- 寄存器：输入和固定输出寄存器读写 entries 及动态能量；
- psum：同 wave 跨 Macro 与跨 wave 累加事件及能量；
- 权重：有效行数、写入能量和装载周期；
- 总线：feature/weight bits、各自 cycles、total cycles；
- 延迟：compute、bus、暴露的 weight/input stall 和 layer makespan；
- 审计：latency cap、epsilon、state cap 及是否触发。

### 11.2 网络级输出

- 实际评价设计的二维 E-T Pareto；
- 实际评价设计的三维 E-T-A Pareto；
- Bayesian 预测结果和不确定度，单独存放；
- 48 个候选的实际评价顺序和模型版本；
- 每个架构的最低能量、最低延迟和最低 EDP 摘要；
- 21 层 schedule trace 与所有近似预算。

不生成按 7 个离散峰值算力档位拆分的正式统计。

## 12. 当前未建模项

- 共享总线动态能量，当前 `bus_energy_per_bit = 0`；
- 总线仲裁器、crossbar、静态布线和层间重配置成本；
- Macro fan-in 上限及大 fan-in 的物理时序退化；
- 输入/输出寄存器、总线、归约、控制和时钟网络的静态功耗；
- clock tree、power gating、温度和 PVT 对 leakage 的影响；
- GLB、NoC、DMA、DRAM 和 line FIFO；
- 激活、BN、pooling、residual add 等非 CIM 算子；
- 跨层流水和完整张量生命周期。

当前比较接受这些未建模项：总线能量和外围能量不分别估算，并按二者对大、小 Macro
具有相反趋势而在一阶近似中抵消。该口径是统一的搜索抽象假设，不是对实际抵消量的
电路测量结论。

## 13. 代码实现状态

当前 runner 已与本规范一致，实现闭环审计如下：

| 项目 | 当前代码状态 | 实现说明 |
|---|---|---|
| 算力窗口与 4 lane | 已实现 | 保持 |
| 固定 32 个输出寄存器 | 已实现 | 每 Tile 只计一次面积 |
| 跨 wave psum 属于 Tile | 已实现 | DP 与报告能量口径一致 |
| Macro 5% 静态拆分 | 已实现 | 输出动态/静态分项 |
| 512-bit特征/64-bit权重互斥入口 | 已实现 | 层内时序严格互斥 |
| 总线 traffic 输出 | 已实现 | 包含 bits/cycles 字段 |
| `latency_cap_ratio=4` | 已实现 | runner 和 checkpoint 版本一致 |
| Bayesian 16/24/8 流程 | 已实现 | 三阶段实际数量已校验 |
| 三维 E-T-A Pareto | 已实现 | 仅使用实际评价结果 |
| 收敛性对照 | 已完成 | 见 8.4 节 |

上述差异已全部关闭，`tile_window_dse_*` 正式结果已生成。

## 14. 目标运行参数

```text
peak-compute reference       = 0.4608 TOPS
peak-compute tolerance       = 0.20
Tile output lanes            = 4
feature bus                  = 512 bit/cycle
weight bus                   = 64 bit/cycle, FEATURE/WEIGHT exclusive
Macro static fraction        = 0.05
bus energy per bit           = 0 J/bit
latency cap ratio            = 4.0
local Pareto epsilon         = 0.001
local state cap              = 4096
aggregate epsilon            = 0.001
aggregate state cap          = 512
Bayesian initial designs     = 16
Bayesian acquisition designs = 24
predicted-front validation   = 8
Bayesian batch size          = 4
total evaluated designs      = 48 / 138
```

## 15. 2026-07-26 正式搜索结果

- 候选池：138 个 Tile；实际评价：48 个。
- 阶段数量：初始 16、Bayesian 采集 24、预测前沿复评 8。
- 聚合可行点：10224；E-T Pareto：1098；E-T-A Pareto：2851。
- 最低能量：`1xm128x8+1xm512x8`，0.262338 mJ，13.956590 ms。
- 最低延迟：`17xm32x8+2xm64x8`，7.448548 ms，0.604695 mJ。
- E-T 折中点：`1xm32x8+5xm128x8`，0.358644 mJ，8.331530 ms。
- 最小面积架构：`15xm32x8`，0.190778 mm2。

主结果与可视化位于 `experiments/results/tile_DSE/tile_window_dse_*`。该结果仍应
标记为 bounded-DP 近似 Pareto，不是 138 个架构的全枚举结果。

## 16. 局部 DP 性能优化

2026-07-26 对小 Macro 密集异构设计完成不改变搜索语义的优化：

- 将内层全局状态数统计改为增量维护；
- 缓存 Macro 资源向量的严格子集；
- 时序缓存使用与物理 lane 顺序无关的规范化 key；
- 将共享权重总线的逐 job-round 循环改为按 lane job 数变化点分段求和；
- 缓存单个 profile 和 active-profile 组合的权重装载周期。

代表长尾层为 `5xm32x8+4xm128x8` 上的
`T16 (K=4608, N=512, P=49)`：

| 项目 | 优化前 | 优化后 |
|---|---:|---:|
| 单层时间 | 270.375 s | 7.988 s |
| Pareto 行数 | 17 | 17 |
| 结果 SHA-256 | `f90514...2823` | `f90514...2823` |

加速比约为 33.8x。该层有 29 个原始分组、25 个剪枝后分组、383031 个候选转移、
324463 个 timing-cache miss，峰值临时状态为 8197。完整 21 层复评产生 169 行，
与优化前正式 CSV 的能量、延迟和调度签名哈希完全一致。

可使用 `experiments/RUN/tile_DSE/profile_sram_tile_dse.py` 对任意 Tile/层输出相同计数器。
这些优化不修改能量、延迟、Pareto 剪枝、state cap 或候选架构，因此不需要
改变冻结模型版本。

当前冻结模型的正式运行命令为：

```bash
third_party/accelforge/.venv/bin/python -u \
  experiments/RUN/tile_DSE/run_sram_tile_bayesian_dse.py \
  --tile-peak-compute-tops 0.4608 \
  --peak-compute-tolerance-ratio 0.20 \
  --design-budget 48 --initial-designs 16 --validation-designs 8 \
  --batch-size 4 \
  --latency-cap-ratio 4.0 \
  --local-pareto-epsilon 0.001 --local-state-cap 4096 \
  --resume
```

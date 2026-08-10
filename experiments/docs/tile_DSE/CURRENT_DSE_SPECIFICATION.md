# CIM计算资源DSE整体设计规范

> 版本：Tile-only Review Draft 5
> 更新时间：2026-07-26
> 当前工作范围收敛为独立 Tile DSE。Core/System 搜索和既有结果均暂停，不作为本阶段
> 结论。本文其余 Core/System 内容保留为后续路线图，不是当前运行配置。

## 当前Tile阶段

本阶段只回答：在以 0.4608 TOPS 为中心的 ±20% 峰值算力窗口内，哪些同构或最多双类型异构 Macro 组合能形成 Tile 级
能量-延迟 Pareto，重点检验`1xm64x8 + 1xm512x8`相对于等算力同构基准的价值。

长期核心研究假设是：`1xm64x8+1xm512x8`能在多种网络负载下稳定成为
E-T 或 E-T-A Pareto 最优组合之一。该组合只作为必测锚点，不得在候选生成、
调度剪枝或目标函数中获得优待。当前 ResNet-18 结果只是首个负载证据；
跨网络结论必须等多负载搜索完成后才能形成。当前已完成 ResNet-18 和 ResNet-50，
后续仍需继续扩展冻结网络集合。
“最优组合之一”的 regret 公式和 `5%/80%/3%/10%/50%` 冻结阈值以
`TILE_DSE_SEARCH_CONDITIONS.md` 第 1.2 节为唯一准据。
跨网络统计不穷举138个理论候选，而是固定使用Bayesian优化选出的同一组48个实测
Tile；每个Tile必须在所有纳入统计的网络上实际评价。最终只列出这48个候选中满足
全部冻结阈值的全算力窗口组合，GP预测值和单独的0.4608 TOPS名单均不进入最终名单。

| 项目 | 当前固定值 |
|---|---|
| 工作负载 | ResNet-18（21层）与 ResNet-50（54层），batch size = 1 |
| 运行频率 | 400 MHz，bit-cycle = 2.5 ns |
| Tile峰值算力参考值 | 0.4608 TOPS |
| Tile候选 | 至多两种Macro类型；峰值算力允许参考值 ±20% |
| Tile输出lane | 1..4条活动lane，硬件上限4 |
| 优化目标 | 主前沿为能量-延迟，同时输出能量-延迟-面积前沿；EE仅作派生输出 |
| 非目标 | 面积硬上限、Core配置、System配置、外设联合搜索 |

`0.4608 TOPS`仍保留为两个对照组合 `9xm64x8` 与 `1xm64x8 + 1xm512x8` 的中心值。
当前只对峰值算力使用 `+-20%` 窗口，不对容量或面积单独设窗。

新候选池为138个，2026-07-26已生成48点/4-lane正式Bayesian结果。层内调度采用
`local_pareto_epsilon=0.001`、`local_state_cap=4096`、`latency_cap_ratio=4.0`的
bounded Pareto DP，跨层聚合上限为512。
状态上限按已完成输出块进度分层采样，并由`local_schedule_state_cap_applied`记录是否触发。
旧24候选/8-lane结果仅作为历史回归基线，不能作为当前配置结论。

## 1. 研究目标

本DSE回答以下问题：

> 给定Macro参数库、神经网络工作负载、最大System峰值算力和一套固定参考外设，Macro应
> 如何组成Tile、Tile应如何组成Core、Core应如何组成System，才能形成有价值的计算资源
> 能效比-延迟Pareto候选，供后续架构设计参考？

本阶段重点搜索计算资源在Tile/Core/System之间的分布。缓存、总线、NoC和控制模块使用
固定参考配置参与成本评价，但不进入第一阶段联合搜索。

## 2. 已确认的整体输入

### 2.1 Macro参数库

DSE接受一种或多种Macro。每种Macro至少包含：

| 字段 | 含义 |
|---|---|
| `macro_type` | Macro类型标识 |
| `rows`, `columns` | 阵列行列数 |
| `input_bits`, `weight_bits`, `output_bits` | 数据精度 |
| `bit_cycle_ns`或`vmm_latency_ns` | 计算周期或完整VMM延迟 |
| `compute_energy_j` | 单次完整VMM动态能量 |
| `macro_area_mm2` | Macro物理表征面积 |
| `weight_row_write_energy_j` | 一行权重写入能量 |
| `interface_class` | 时钟、位宽和接口兼容类别 |

当前Macro表由AccelForge表征，候选为：

```text
m32x8, m64x8, m128x8, m256x8, m512x8
```

当前统一输入/权重精度为8 bit，Macro列数为8，bit-cycle为`2.5 ns`（400 MHz）。权重写入能量
按照同等规模、64-bit写宽、单bank SRAM的写入能量建模。

### 2.2 工作负载

当前工作负载为`batch_size = 1`的ResNet-18，共21个Conv/FC计算层。每层统一表示为：

\[
Y_l[P_l,N_l]=X_l[P_l,K_l]W_l[K_l,N_l]
\]

其中：

```text
P = Batch * Hout * Wout
K = Cin * R * S
N = Cout
OP_l = 2 * P_l * K_l * N_l
```

每层输入还包括输入、权重和输出数据量，以及与前后层的依赖关系。

### 2.3 最大System峰值算力

主资源约束输入为：

```text
peak_compute_budget_TOPS
```

当前第一版取：

```text
peak_compute_budget_TOPS = 4.0
```

它表示固定精度和频率下允许配置的最大理论System峰值算力，不是ResNet-18映射后的有效
吞吐率。

对Macro类型`q`：

\[
\phi_q=\frac{2R_qC_q}{t_{VMM,q}}
\]

若输入按1 bit/cycle串行，则：

\[
t_{VMM,q}=input\_bits\times t_{bit,q}
\]

### 2.4 固定参考外设

外设不参与第一阶段配置搜索，但参与能量和延迟计算。当前参考平台为：

| 层级 | 固定配置 |
|---|---|
| Tile | 每Macro两组ping-pong输入寄存器、一组输出寄存器、最多4条逻辑输出lane |
| Core总线 | 特征入口512 bit/cycle、权重入口64 bit/cycle，两类传输时序互斥 |
| Core缓存 | 1 MiB，256-bit访问宽度，4 banks |
| Core向量模块 | 8 lane，每元素最大32 bit |
| System SRAM | 4 MiB，512-bit访问宽度，16 banks |
| System NoC | 512-bit/cycle二维Mesh，确定性XY路由 |
| 成本来源 | AccelForge、Aladdin和SmartBufferSRAM组件模型 |

外设变化时，重新评价候选的事件计数和调度，不重新定义计算资源组合。

## 3. 已确认的整体输出

DSE不输出一个唯一最优架构，而是输出一组候选Tile/Core/System组合。

### 3.1 Tile输出

- 最多两种Macro类型及各自数量；
- Tile峰值算力、计算面积和权重容量；
- 各层Macro映射、活动lane数和K-wave数；
- 各层有效利用率、计算能量、权重写入能量和延迟；
- Tile内部分和事件数量。

### 3.2 Core输出

- Tile类型和每Core的物理Tile数量；
- `core_compute_area_mm2`；
- Core峰值算力和各层有效算力；
- 各层活动Tile数量及N维任务划分；
- 共享总线流量、带宽利用率和瓶颈类型；
- Core缓存、向量处理、广播、控制和同步能量。

### 3.3 System输出

- Core类型和物理Core数量；
- System配置峰值算力及其相对预算的占用率；
- `system_compute_area_mm2`，即物理Core数乘以`core_compute_area_mm2`；
- 每层活动Core数和`C_P x C_N`映射；
- 完整网络单次推理能量、延迟、TOPS/W和TOPS/mm2；
- 权重复制、输入广播、System SRAM访问和NoC bit-hop；
- 每层计算、Core总线、System通信和输出回写延迟分项。

### 3.4 最终候选集合

最终至少输出：

- 完整能效比-延迟Pareto；
- 最大能效比端点；
- 最小延迟端点；
- Pareto knee候选；
- 计算资源层级划分明显不同的代表候选；
- 每个候选的面积、利用率、带宽压力和缓存需求，供后续架构筛选。

## 4. 计算资源搜索空间

一个完整硬件配置表示为：

\[
x=(\mathbf n_{tile},T,C)
\]

其中`n_tile`是Tile内各Macro类型的数量向量，`T`是每Core的Tile数量，`C`是System物理
Core数量。

### 4.1 Macro层

Macro是输入候选库，不在本阶段重新搜索SRAM cell、ADC、加法树、数据精度或工艺参数。
Macro参数发生变化时，作为新的Macro表重新运行上层DSE。

### 4.2 Tile层

Tile允许同构或双类型异构Macro：

\[
\left|\{q\mid n_q>0\}\right|\leq2
\]

```text
同构Tile：n_a * Macro_a
异构Tile：n_a * Macro_a + n_b * Macro_b
```

其中`n_a >= 1`，异构时`n_b >= 1`且`a != b`。

Tile峰值算力和计算面积为：

\[
\phi_{tile}=\sum_qn_q\phi_q
\]

\[
A_{tile}=\sum_qn_qA_q+A_{mandatory\_registers}
\]

当前Tile-only搜索使用相对算力窗口：

\[
0.36864\ \mathrm{TOPS}\leq\phi_{tile}=\sum_qn_q\phi_q\leq0.55296\ \mathrm{TOPS}
\]

候选枚举不再使用容量、面积、Macro总数量或旧的48个Tile作为筛选条件。面积和容量仅保留
为结果诊断，不参与合法性判断或能量-延迟Pareto。每个Tile可激活`1..4`条
输出lane；同一lane内的Macro沿K维协作并在Tile内部完成跨Macro部分和，不同lane沿N维并行。

`9xm64x8`和`1xm64x8+1xm512x8`均位于窗口中心，必须作为对照候选。当前结论应限定为
“在0.4608 TOPS参考值、±20%算力窗口和最多4 lane下的能量-延迟Pareto”。若要判断异构
组成本身的价值，应在相同离散峰值算力档位内再做分层比较。

### 4.2.1 历史48个Tile的来源

当前48个Tile来自一次旧版Tile候选生成和贝叶斯采样流程，具体过程如下。

#### 第一步：定义参考Tile

以`9xm64x8`作为参考Tile：

```text
reference capacity = 9 * 64 * 8 = 4608 个逻辑权重值
reference area     = 0.390881021 mm2
```

其中Tile面积包括Macro面积和每个Macro对应的输入/输出寄存器面积。

#### 第二步：枚举理论候选

对5种Macro类型的数量进行整数枚举：

```text
m32x8, m64x8, m128x8, m256x8, m512x8
```

枚举过程中只保留：

- 至少包含一种Macro；
- 最多包含两种Macro类型；
- Tile容量位于参考容量的`+-20%`范围；
- Tile总面积位于参考面积的`+-20%`范围。

因此理论候选池为137个：

| 类别 | 数量 |
|---|---:|
| 同构Tile | 14 |
| 双类型异构Tile | 123 |
| 总计 | 137 |

Macro总数量范围为1到21。这里的`+-20%`只解释当前48个候选是如何历史产生的，不再作为
新规范下的Tile、Core或System正式硬约束。

#### 第三步：选择16个贝叶斯初始点

在137个理论候选中，初始点固定包含：

- `9xm64x8`参考Tile；
- `1xm64x8+1xm512x8`异构基线；
- 全部14个同构Tile。

因此初始设计数为16。这样可以确保不同Macro尺寸的同构基线和两个原始参考点都被评价。

#### 第四步：贝叶斯采样到48个设计

其余32个设计通过自定义NumPy Gaussian Process贝叶斯搜索选择。每个候选的输入特征为：

```text
5种Macro的数量
Tile容量相对参考值
Tile面积相对参考值
Macro总数量
```

特征归一化后，使用已经评价设计的完整ResNet-18结果训练GP模型。GP同时预测以下三个
网络级指标：

```text
minimum network energy
minimum network latency
minimum network EDP
```

采集函数综合考虑：

- 预测目标值；
- GP预测不确定性；
- 与已采样点的距离，以保持候选结构多样性。

每轮最多选择4个候选，直到完成48个Tile的完整逐层调度评价。因此：

```text
137 个理论候选
  -> 16 个固定初始点
  -> 32 个贝叶斯采样点
  -> 48 个实际评价Tile
```

48个已评价Tile中包含14个同构Tile和34个异构Tile。每个已评价Tile都覆盖ResNet-18的
21个计算层，并保存层级调度、能量、延迟、EDP、面积和利用率结果。

#### 第五步：生成Core使用的25个代表Tile

48个Tile并非全部直接作为Core搜索输入。当前Core DSE使用一个25个Tile的代表性coreset，
选择规则为：

- 保留全部14个同构Tile，覆盖不同Macro尺寸和Tile规模；
- 保留参考Tile和原始异构基线；
- 保留最小能量、最小延迟和最小EDP候选；
- 保留能效-延迟knee点；
- 覆盖已经采样的异构Macro类型组合；
- 保留具有面积和资源粒度差异的候选。

因此当前候选关系为：

```text
137 个理论Tile候选
  -> 48 个已完整评价Tile
  -> 25 个Core DSE代表Tile
```

25个代表Tile是48个已评价Tile的压缩集合，不是新的Tile搜索结果。第一版System结果必须
记录使用的是48个已评价池，还是25个Core代表coreset，避免混淆。

### 4.3 Core层

第一阶段Core由同一种Tile复制组成：

```text
tile_design = 一个Tile候选
tile_count_per_core = T
```

\[
\phi_{core}=T\phi_{tile}
\]

`T`不再固定限制为`1..8`，其自然上界为：

\[
1\leq T\leq\left\lfloor\frac{\phi_{max}}{\phi_{tile}}\right\rfloor
\]

Core内部不同Tile只做输出维`N`并行。一个输出通道块由一个Tile完成，禁止跨Tile切分
`K`，因此不产生跨Tile部分和规约。

### 4.4 System层

第一阶段System由同一种Core复制组成：

```text
core_design = 一个Core候选
physical_core_count = C
```

\[
\phi_{system}=C\phi_{core}
\]

Core数量满足：

\[
1\leq C\leq
\min\left(
\left\lfloor\frac{\phi_{max}}{\phi_{core}}\right\rfloor,
\max_l(P_l\lceil N_l/8\rceil)
\right)
\]

第二项是当前`P x N`映射下由工作负载决定的最大有用并行度，避免生成任何层都不可能
使用的物理Core。

每层独立搜索：

\[
C_{P,l}\geq1,\qquad C_{N,l}\geq1
\]

\[
C_{P,l}C_{N,l}\leq C
\]

```text
K_splits = 1
weight_replication_factor = C_P
input_broadcast_factor = C_N
```

一个Core任务为`[P_tile,N_tile] x full-K`。不同层可以启用不同数量的Core，未使用Core
动态能量记为0。

## 5. 限制条件

### 5.1 全局计算资源约束

所有物理配置的峰值算力都计入预算，包括某些层无法使用的空闲资源：

\[
C T\sum_qn_q\phi_q\leq\phi_{max}
\]

一个System候选必须使用同一个固定Tile/Core/System配置完成全部21层，不能逐层更换物理
资源配置。

### 5.2 Tile约束

- 一个Tile最多包含两种Macro类型；
- 同一Tile内两种Macro必须具有兼容的数据精度、输出格式和接口类别；
- 所需输入和权重就绪后才启动一组Macro；
- 同一lane内允许Macro沿K维分工，跨Macro部分和由Tile处理；
- 同一K-wave内规约延迟认为被Macro计算掩盖，规约动态能量保留；
- Tile内部输入广播、控制和同步暂不单独计费；
- 不同lane沿N维并行，活动lane数不超过4；
- 所有`P/K/N`任务必须覆盖，填充和尾块产生的无效计算计入利用率。

### 5.3 Core约束

- Core内部Tile同构；
- Tile之间只做N维并行，不允许跨Tile切分K；
- 特征入口带宽为512 bit/cycle，权重入口带宽为64 bit/cycle；两类传输不能同周期发生；
- 跨K-wave部分和由Core缓存和向量模块处理；
- `psum_width <= 32 bit`，单个8通道psum向量不超过256 bit；
- 峰值在途psum容量不超过1 MiB Core缓存；
- Core输出直接从Tile输出寄存器拉出，不增加Core级输出汇聚模块。

### 5.4 System约束

- System内部Core同构；
- Core之间只按P和N维划分任务，当前不跨Core切分K；
- `C_P <= P`，`C_N <= ceil(N/8)`；
- 每层必须覆盖全部输出，所有活动Core完成后该层才完成；
- 权重和输入从System SRAM读取后通过二维Mesh组播，输出在层末写回System SRAM；
- 4 MiB SRAM按分块工作集进行容量检查，不要求完整展开层一次性驻留；
- 当前不做跨层流水，完整网络延迟为21层延迟之和。

### 5.5 面积处理

面积不是当前主硬约束和主优化目标。DSE必须输出：

\[
A_{core\_compute}=T A_{tile}
\]

\[
A_{system\_compute}=C A_{core\_compute}
\]

可选输入`area_sanity_limit_mm2`只用于排除明显无法物理实现的配置，不用于定义主Pareto。
System SRAM、NoC和控制器面积不计入`system_compute_area_mm2`，但应在结果中明确标记这一
面积边界。

### 5.6 当前不建模项

- DRAM动态能量和外存带宽；
- 卷积halo通信；
- NoC详细仲裁、拥塞和虚通道；
- 时钟树、泄漏和电源网络；
- Tile广播、控制和同步的面积与能量；
- 跨层流水并行；
- 跨Core K维并行及其部分和规约。

## 6. 成本模型

### 6.1 能量

完整网络动态能量为：

\[
E_{network}=\sum_l(
E_{macro\_compute}
+E_{weight\_write}
+E_{tile\_psum}
+E_{core\_bus}
+E_{core\_cache/vector/control}
+E_{system\_SRAM}
+E_{NoC}
+E_{system\_control})_l
\]

固定参考外设只负责对SRAM访问、bit-hop、广播和控制事件定价。各事件数量独立输出，便于
将来改变外设参数后重新评价候选。

Tile DSE主结果不单独计入总线能量和额外外围开销，采用二者对大、小 Macro 呈相反趋势、
在候选排序中一阶抵消的统一抽象。该假设用于相对比较，不解释为物理电路中的精确抵消。

### 6.2 延迟

Tile层按 N-block/K-wave 批次计时。每个批次先串行完成权重加载，随后允许该批次的
特征加载与计算理想重叠：

```text
tile_layer_latency = sum_over_batches(
    weight_load_latency[batch]
    + max(feature_load_latency[batch], compute_latency[batch])
)
```

Tile不保存完整层输入特征，所需特征按批次重复流入。同一 bank 禁止权重加载与计算
重叠；ViT 投影与注意力位于不同 bank 时允许跨 bank 的权重加载/计算重叠。当前模型
不加入首次填充和末尾输出排空延迟。

扩展到Core层后，还需计入Core外设路径：

```text
core_layer_latency = max(
    tile_makespan,
    psum_cache_latency,
    vector_latency
) + core_dispatch_and_finish
```

System层采用：

```text
system_layer_latency = max(
    core_makespan,
    system_input_and_weight_transfer,
    core_cache_staging
) + output_writeback + system_dispatch_and_finish
```

并行Core的计算阶段由最慢Core决定。

### 6.3 有效算力与利用率

峰值算力只作为资源预算。候选的工作负载有效算力定义为：

\[
\phi_{effective}=\frac{OP_{network}}{L_{inference}}
\]

同时输出：

\[
utilization=\frac{\phi_{effective}}{\phi_{system}}
\]

该利用率包含阵列填充、任务粒度、带宽和空闲资源造成的综合损失。

## 7. 优化目标与Pareto定义

### 7.1 最终System目标

在以下硬约束下：

\[
\phi_{system}\leq\phi_{max}
\]

执行双目标优化：

\[
\min E_{inference},\qquad \min L_{inference}
\]

由于工作负载计算量固定，最小化`E_inference`等价于最大化：

\[
EE=\frac{OP_{network}}{E_{inference}}
\]

最终对外展示坐标为：

```text
maximize energy_efficiency_TOPS_per_W
minimize single_inference_latency_ms
```

面积、峰值算力占用率、TOPS/mm2、利用率和外设压力作为输出和候选解释指标，不加入主
Pareto坐标。若两个候选能量和延迟相同，优先保留峰值算力更小、面积更小的候选。

### 7.2 Tile和Core中间剪枝

Tile和Core层的局部Pareto只用于降低搜索量，不定义最终最优。中间状态至少按以下特征
进行多目标保留：

```text
compute_energy
latency
peak_compute
compute_area
weight_writes
traffic
utilization
```

不得只保留单一外设配置下能量或延迟最优的一个调度。每个架构、每个网络层保留若干
计算-流量-延迟折中状态，System层重新选择并完成全网评价。

### 7.3 代表候选选择

从最终Pareto及其epsilon邻域中选择：

1. 最大能效比候选；
2. 最小延迟候选；
3. 能效比-延迟knee候选；
4. 最高TOPS/mm2候选；
5. 低带宽压力候选；
6. 使用不同Macro组合、Macro/Tile、Tile/Core和Core/System粒度的结构多样性候选。

代表候选用于架构设计参考，不等价于在完整芯片PPA上已经最优。

## 8. 搜索与剪枝方法

第一版采用分层搜索但最终回到System重评，避免逐层贪心确定唯一结构：

1. 使用当前已评价的48个Tile作为第一版候选池；
2. 从能量、延迟、利用率、算力、面积和结构多样性维度选取代表Tile；
3. 对`Tile类型 x 每Core Tile数`使用贝叶斯优化和精确整数邻域搜索生成Core候选；
4. 对每个Core候选枚举算力预算允许的System Core数量；
5. 每层搜索全部合法`C_P x C_N`，保留层级能量-延迟状态；
6. 使用动态规划聚合21层；
7. 对接近System Pareto但在Tile/Core阶段被剪枝的邻域配置进行回填评价；
8. 输出最终Pareto和代表性候选。

贝叶斯采样数、epsilon和状态上限是运行预算参数，不是架构硬约束，必须写入结果元数据。

## 9. 实现状态与设计决策

### 9.1 当前可复用部分

- 5种Macro及其计算、面积和权重写入表征；
- 已评价的48个Tile候选；
- Tile/Core逐层调度和部分和模型；
- System逐层`P x N`空间并行模型；
- 固定Core总线、System SRAM和二维Mesh NoC成本模型；
- 完整网络动态规划聚合与能量-延迟Pareto工具。

### 9.2 已按本规范完成的部分

- 用`peak_compute_budget_TOPS = 4.0`替代System面积预算作为主硬约束；
- 已为Tile、Core和System输出峰值算力及峰值算力密度；
- Core Tile数和System Core数的默认上界已改为由峰值算力预算推导；
- System输出改为峰值算力约束下的Pareto文件；
- Core贝叶斯代理目标不再使用面积和Macro数量，面积仅保留为诊断输出；

### 9.3 仍需按本规范修改的部分

- 取消旧Tile基准容量/面积`+-20%`作为理论候选生成约束；当前48个Tile仍保留其历史来源；
- 将System SRAM容量检查从“完整层驻留”改为“分块工作集驻留”；
- 最终输出候选中保留计算面积、TOPS/mm2和外设需求特征；
- 正式重跑Core/System DSE并生成新版本结果。

### 9.4 已确认决策

1. 峰值算力统一采用`2 * rows * columns / VMM_latency`计算；
2. 当前48个固定Tile作为第一版唯一Tile候选池；
3. 取消旧Tile容量和面积`+-20%`窗口作为总体设计硬约束；
4. Core Tile数和System Core数的上界由峰值算力预算推导；
5. 保留Core内部同构Tile和System内部同构Core；
6. System层只搜索`P x N`并行，暂不搜索跨Core K切分；
7. 面积只作为输出和可选sanity filter；
8. 最终主Pareto为能效比-单次推理延迟双目标。

### 9.5 Tile尺寸解释边界

第一版使用48个固定Tile，因此Tile大小是离散候选输入，而不是从无局部边界的完整Macro
数量空间中重新搜索得到。旧窗口不再用于后续Core/System筛选，但已经影响了这48个Tile
的覆盖范围。第一版结果必须标记为：

> 在当前48个Tile候选池内，由最大System峰值算力约束和完整网络Pareto选择Tile大小。

若后续目标升级为真正搜索Tile大小，可以保留48个Tile的Macro组成比例作为模板，并沿
Macro数量尺度生成扩展候选：

\[
\mathbf n_{tile}(s)=s\hat{\mathbf n}_{tile},\qquad s=1,2,\ldots
\]

其中`n_hat_tile`是约分后的同构类型或双类型Macro比例，`s`的上界仍由
`phi_tile(s) <= phi_max`确定。该扩展不属于第一版搜索。

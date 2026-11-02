# CIM DSE中的Macro、Tile、Core与System层级

## 1. 层级划分的目的

本项目按照以下物理层次探索计算资源配置：

```text
System
  -> 多个同构 Core
       -> 多个同构 Tile
            -> 多个 SRAM-CIM Macro
```

每一级的核心设计变量分别是：

```text
Tile 内配置多少个、哪些类型的 Macro
Core 内配置多少个同构 Tile
System 内配置多少个同构 Core
```

这些硬件层级不与神经网络的软件层级一一对应。一个 Macro 不等于一个卷积核，一个
Tile 也不固定对应一个神经网络层。Macro和Tile是Core内部可跨层复用的计算资源；
Core是能够独立执行完整神经网络的最小完备模块；System只负责集成多个独立Core，
提供共享供电、时钟、存储接口和任务分发。

### 1.1 外设资源的归属原则

每一级不只包含计算资源，也包含协调其下一级计算单元所需的存储、互连、控制、时钟
和电源资源。外设的归属采用下面的原则：

> 一个外设应放在共享它的所有计算单元的最低公共层级。

因此，一个Macro独占的驱动和寄存器属于Macro；多个Macro共享的广播和归约网络属于
Tile；多个Tile共享的总线和调度器属于Core；多个Core共享的存储和NoC属于System。

完整数据路径为：

```text
DRAM
  -> System内存接口 / 多Core集成互连
    -> Core控制器 / Core缓存 / 向量处理模块
      -> Core共享总线 / Tile广播
        -> Tile局部互连 / Macro输入分发 / 输出路由
          -> Macro输入驱动 / 权重阵列 / 本地输出接口
```

返回路径中，同一个K wave内的多个Macro结果先在Tile完成归约，形成每条lane的8元素
部分和向量。如果K过大而需要多个wave，Tile再将每个wave的结果送到Core向量处理模块，
由Core完成跨wave累加并将中间状态保存在Core缓存中。System不参与psum归约。

同一种资源可以在多个层级出现，例如寄存器和缓冲，但它们保存的数据范围、共享对象和
访问次数不同，不能混为同一个资源或重复计费。

## 2. 统一的神经网络计算表示

当前研究将卷积和全连接统一表示为矩阵乘形式：

```text
Y[p, n] = sum_k X[p, k] * W[k, n]
```

其中：

| 维度 | 含义 | 卷积中的对应量 |
|---|---|---|
| `K` | 规约维度 | `Cin * R * S` |
| `N` | 输出维度 | 输出通道 `Cout` |
| `P` | 独立输出位置 | `Batch * Hout * Wout` |

当前每个 Macro 固定产生8个输出通道，因此输出维度进一步划分为：

```text
N_blocks = ceil(N / 8)
```

四个硬件层级的区别，本质上是它们覆盖 `K`、`N_blocks` 和 `P` 的粒度不同，以及它们
负责的存储、通信和控制范围不同。

## 3. 层级总览

| 层级 | 计算资源 | 该层特有的外设 | 主要设计变量 | 神经网络工作粒度 |
|---|---|---|---|---|
| Macro | 一个SRAM-CIM阵列 | 行列驱动、权重写入、本地输入和输出接口 | 行数、列数、精度、周期 | 一个K片段、8个输出通道、一个位置 |
| Tile | 多个Macro | Macro输入接口、跨Macro归约、lane控制和端点寄存器 | Macro类型与数量 | 产生每个K wave的lane部分和 |
| Core | 多个同构Tile | Core缓存、跨wave向量处理、共享总线、广播和完整控制器 | Tile类型与物理Tile数 | 合并K waves并执行完整网络 |
| System | 多个完备Core | 多Core互连、共享SRAM、任务分发、芯片级时钟/电源 | Core类型与物理Core数 | 逐层P×N联合推理，也支持独立执行 |

## 4. Macro层级

### 4.1 物理含义

Macro 是当前模型中最小的黑盒CIM计算单元。候选形状为：

```text
m32x8, m64x8, m128x8, m256x8, m512x8
```

`mRx8` 表示：

- 最多同时覆盖 `R` 个K维权重；
- 同时产生8个输出通道的部分和；
- 输入和权重均为8 bit；
- 输入采用1 bit/cycle的位串行计算；
- 一次完整8-bit VMM需要8个Macro周期。

Macro级外设紧贴阵列，并随物理Macro数量一起复制：

- SRAM权重存储单元及写入电路；
- wordline、bitline和列选择驱动；
- 位串行输入驱动；
- 阵列内部乘法、加法树和shift-add；
- 本地输出锁存或累加接口；
- Macro级使能和时钟门控。

这些资源只服务一个Macro，不负责Macro之间的数据广播或部分和归约。

### 4.2 对应的神经网络工作

一次 Macro VMM 对应：

```text
一个输出位置 p
一个8通道输出块 n_block
最多R个K维元素
```

如果 `K > R`，一个 Macro 必须经过多个VMM wave才能完成规约。如果多个 Macro 共同
处理同一输出块，它们分别计算不同K片段，并在该Tile内部先完成跨Macro归约。如果单个
Tile仍无法在一个wave中覆盖完整K，则各wave的Tile结果再送往Core向量处理模块累加。

Macro 保存的是某个K片段和8个输出通道对应的权重块，不保存或执行一个完整神经网络
层。它主要决定：

- K划分粒度；
- 行利用率；
- 单次VMM能效；
- 单位面积吞吐率；
- 权重容量。

### 4.3 当前Macro模型边界

Macro的能效和面积效率来自表征表。所有候选暂时采用共同的 `2.5 ns` 周期（400 MHz）。内部
SRAM、驱动、乘法、加法树和位串行电路保持黑盒，不在当前DSE中重新综合。因为这些
内部外设原则上已经包含在Macro黑盒TOPS/W和TOPS/mm2中，上层不能再次计费。当前需要
额外补充的是权重编程能耗；现有模型只计算了权重装载延迟。

## 5. Tile层级

### 5.1 物理含义

Tile 由多个 Macro 及其接口资源组成。当前 Tile 可以包含至多两种 Macro 类型，同一种
类型可以有多个实例。Tile 的物理计算资源数为：

```text
M_tile = sum_i macro_count_i
```

当前 Tile 还包含：

- 每个Macro的两组ping-pong输入寄存器；
- 每个Macro的一组输出寄存器；
- 最多4条逻辑输出lane；
- 同lane部分和累加单元。

物理实现中仍会存在Macro输入连线、权重选择、lane发射和ready/done信号，但当前假设
Tile内Macro数量较少，因此不单独计算这些网络和控制逻辑的能耗、延迟或面积。

### 5.2 对应的神经网络工作

Tile 内存在两种并行关系：

1. K维并行：同一lane中的多个Macro处理同一输出块、同一个K wave内的不同K片段，
   跨Macro部分和在该Tile内部累加。
2. N维并行：不同lane处理不同的8通道输出块，结果互不相加。

一条lane可以顺序处理多个输出块。所有输出位置 `P` 也按时间顺序执行，同一组权重在
不同输出位置之间复用。因此，一个 Tile 在某一时刻负责的是：

```text
一个K wave内由多个Macro共同覆盖的K片段
若干个8通道输出块
一个或一批输出位置
```

如果完整K需要多个wave，Tile只负责分别生成每个wave的已归约结果，不在Tile内长期
保存所有跨wave部分和。跨wave状态由Core缓存和Core向量处理模块负责。

Tile 并不固定绑定某个神经网络层。当前 Tile DSE 让同一个 Tile 依次执行ResNet-18的
21个Conv/FC层，用全网累计能量和延迟比较不同Macro配置。

### 5.3 Tile DSE回答的问题

Tile层主要回答：

> 为了兼顾大K层、小K层、行利用率、能效和面积，一个Tile应该配置多少个、哪些类型
> 的Macro？

主要设计变量为各Macro类型的数量。运行时调度变量包括活动Macro分组、活动lane数、
K划分方式和每条lane承担的输出块数。

当前 Tile DSE采用以下有意简化：

```text
Macro间输入广播能耗、延迟和面积 = 0
Tile控制与同步能耗、延迟和面积 = 0
同一Tile内跨Macro归约延迟         = 0
```

每组活动Macro只有在所需输入和权重全部就绪后才开始计算，因此不再增加组内同步延迟。
同一lane、同一wave的跨Macro归约与Macro计算重叠，归约延迟被Macro计算掩盖。对应的
psum动态能耗仍然保留，因为延迟被掩盖不代表归约不消耗能量。

当前 Tile 面积和动态能耗边界为：

```text
A_tile = sum(A_macro) + A_input_register + A_output_register

E_tile = E_macro
       + E_input_register
       + E_output_register
       + E_cross_macro_psum
```

权重装载延迟仍计入Tile makespan，权重装载能耗当前为0。时钟和泄漏继续排除。

## 6. Core层级

### 6.1 物理含义

Core 由多个完全相同的 Tile 组成：

```text
T_core = physical Tile count per Core
```

当前 Core-v1 默认搜索 `T_core = 1..8`，并区分：

- `core_tile_count`：物理配置的Tile数，决定面积和控制器规模；
- `active_tile_count`：某一层实际启用的Tile数，决定并行度和动态开销。

Core 还包含：

- 可独立执行完整网络层序列的Core控制模块；
- 保存输入、权重、中间激活和跨wave psum的Core级缓存；
- 执行跨wave累加及其他向量操作的向量处理模块；
- 特征/权重共享的512 bit/cycle总线；
- 输入特征广播网络；
- 512-bit源端寄存器；
- 总线请求仲裁和地址计数；
- 每个Tile的任务描述符；
- active、ready和done mask；
- Tile调度、dispatch和完成同步逻辑；
- Core级时钟和电源控制；
- 可选的Core共享输入缓冲或卷积line buffer。

Core输入缓冲与Tile输入寄存器不是同一个资源：前者保存可供多个Tile共享的数据，后者
是每个Tile/Macro的计算端点。Core输入缓冲读一次后可以广播，而Tile输入寄存器仍需在
各接收端写入。

### 6.2 对应的神经网络工作

Tile 之间只做输出维 `N` 并行：

```text
Tile0 -> 一组输出通道块
Tile1 -> 另一组输出通道块
...
```

所有活动 Tile 使用相同输入特征，因此输入可以广播。不同 Tile 负责不同输出通道，
对应权重互不相同，因此权重按N块分发而不复制。Tile之间不切分同一个输出块的K，
所以没有跨Tile部分和归约；但单个Tile可以针对自己的输出块执行多个K wave，这些wave
之间的部分和由Core向量处理模块累加。

从神经网络视角，一个 Core 可以完成一个Conv/FC层的全部输出，过程是：

```text
将该层N_blocks分配给活动Tile
  -> Core缓存提供输入和当前K wave权重
  -> 输入广播、权重按输出通道送往各Tile
  -> 每个Tile在内部完成同一wave的跨Macro归约
  -> Tile将每个wave的lane结果送回Core
  -> Core向量模块累加跨wave psum
  -> Core缓存保存中间psum和最终输出
```

Core向量模块只累加同一个Tile、同一个输出通道块的不同K wave，不在不同Tile之间做
psum归约。各Tile负责的输出通道互不重叠。完成一个层后，最终输出保存在Core缓存中，
供该Core独立执行下一层。

因此Core是最小的完备神经网络执行模块：它自行管理层序、数据驻留、Tile调度、部分和
状态及向量处理，不依赖System参与单个网络内部的计算或归约。

### 6.3 大K时的部分和路径

设单个Tile在一个wave中激活的Macro总行数为 `K_wave`：

```text
n_waves = ceil(K / K_wave)
```

部分和按以下顺序处理：

```text
每个K wave：
  Macro0..MacroM分别计算K片段
    -> Tile内部完成跨Macro归约
    -> 得到一个8元素lane psum向量

多个K waves：
  wave0 Tile结果写入Core缓存
  wave1..waveN由Core向量模块读取、累加、写回
  最后一个wave生成最终输出
```

部分和位宽为：

```text
psum_width = input_bits + weight_bits + ceil(log2(K))
```

若Core同时保留 `P_block` 个输出位置和 `B_active` 个8通道输出块，所需Core psum容量为：

```text
core_psum_bits = P_block * B_active * 8 * psum_width
```

跨Macro归约能耗属于Tile；跨wave向量累加、psum缓存读写、缓存容量和面积属于Core。
两类归约都不产生跨Tile或跨Core数据依赖。

### 6.4 Core DSE回答的问题

Core层主要回答：

> 在Core缓存、向量处理、共享带宽、广播、控制和面积约束下，一个完备Core应该配置
> 几个同构Tile？

Core DSE同时选择代表Tile设计和物理Tile数。增加Tile数可以降低层延迟，但会增加：

- Core面积；
- 广播树规模和扇出；
- 控制端点数量；
- 对共享特征/权重总线的竞争。
- Core缓存的并发psum容量和访问量；
- Core向量处理模块的累加工作量。

卷积line buffer放置位置也是Core外设设计的一部分：

| 放置方式 | 优点 | 代价 |
|---|---|---|
| 每Tile一个line buffer | Tile接口独立、调度简单 | N并行时复制相同输入和容量 |
| Core共享line buffer | 输入只保存一次，可直接向Tile广播 | 共享端口、广播和仲裁更复杂 |

当前Tile之间采用N并行并共享相同输入，因此更适合将line buffer或输入scratchpad放在
Core层，再由Core广播网络送入各Tile端点。Core-v1已经加入默认1 MiB、256-bit、4-bank
的Core SRAM，其容量、面积及跨wave psum动态访问由AccelForge表征，参数可通过命令行
调整。当前尚未将输入、权重和激活的容量占用及动态流量加入该缓存，也未搜索独立的
line buffer容量。

当前研究阶段的面积指标只服务计算资源分布探索，因此Core DSE仅输出：

```text
core_compute_area_mm2 = core_tile_count * tile_area_mm2
```

SRAM、向量模块、控制器和总线面积暂不输出，也不进入Core面积Pareto目标。它们的容量、
带宽和动态能耗约束仍保留。上文的完整Core面积公式属于后续物理实现阶段，而不是当前
计算资源DSE的评价口径。

## 7. System层级

### 7.1 物理含义

System 将由多个完全相同的 Core 组成：

```text
C_system = physical Core count per System
```

必须区分：

- 物理 `core_count`；
- 某一层参与联合推理的 `active_core_count`。

System只负责将多个完备Core集成为芯片或加速器，包含：

- Core接入互连、router和链路；
- DRAM控制器、DMA和外存接口；
- 4 MiB、512-bit、16-bank共享SRAM；
- 512-bit/cycle 2D mesh NoC；
- 算子分片到Core的分发、完成通知和同步；
- 芯片级时钟、电源和I/O；
- 多Core共享内存带宽和功率预算的仲裁。

每个Core内部已经包含执行完整网络所需的控制、缓存和向量处理资源，但System可以把
一个层拆给多个Core联合执行。System不保存跨层psum，也不进行跨Core K维部分和归约。

### 7.2 对应的神经网络工作

Core具备独立执行完整网络的能力，但当前System调度按层生成二维任务网格：

```text
Core(i,j) -> [P_i, N_j] x 完整K
```

每层独立选择 `C_P x C_N`，满足 `C_P*C_N<=physical_core_count`。权重按N切分并在P组
之间复制，因此权重复制倍数为C_P；输入按P切分并在N组之间广播，因此输入复制倍数为
C_N。完整K始终留在一个Core内，输出无需跨Core数值规约。

System DSE除了搜索Core数量，还需要探索或约束以下外围配置：

```text
可选共享缓存容量、bank数和端口数
Core接入互连拓扑、链路宽度和工作频率
DRAM有效带宽
DMA并发请求数
任务队列深度和Core分配策略
```

当前第一版System DSE精确枚举 `system_core_count=1..32`，并对每层枚举合法P/N网格。
每个Core保持一个N分片权重驻留并连续完成其P区域。每层输出写回System共享SRAM，下一层
再由共享SRAM和NoC重新分发。动态能量包括：

```text
Macro SRAM权重行写
Core计算、缓存和总线
System共享SRAM读写
NoC多播与输出回传bit-hop
System dispatch与完成同步
```

System固定外围计入动态能量和带宽，但面积目标仍只输出
`system_core_count*core_compute_area`。在每个计算面积预算内寻找：

```text
最大能效比 TOPS/W
最小单次推理延迟
```

不同层可以使用不同活动Core数和P/N网格。增加P并行会复制权重，增加N并行会复制输入，
所以Core数量能够形成真实的能效比-延迟权衡。当前仍排除halo、DRAM能量、泄漏、时钟和
详细NoC拥塞仲裁。

## 8. 配置资源与活动资源必须分开

每个层级都要区分物理配置和运行时活动资源：

| 层级 | 物理配置 | 运行时活动量 |
|---|---|---|
| Tile | `tile_macro_count` | 活动Macro数、活动lane数 |
| Core | `core_tile_count` | `active_tile_count` |
| System | `system_core_count` | `active_core_count` |

物理配置决定面积、容量、端口和静态开销；活动资源决定某一层的动态能耗、并行度和
带宽需求。未活动的资源仍占面积，未来加入泄漏和时钟树后也可能继续消耗能量。

整个系统的物理Macro总数为：

```text
M_system = M_tile * T_core * C_system
```

总面积和能耗不能只由这个计算资源乘积得到。更完整的资源表达为：

```text
A_tile   = sum(A_macro) + A_tile_register
A_core   = T_core * A_tile + A_core_control_cache_vector_bus
A_system = C_system * A_core + A_system_integration(C_system, interconnect, DMA)
```

动态能耗也应按访问发生的位置分层计费：

```text
E_total = E_macro_compute
        + E_tile_register_cross_macro_psum
        + E_core_cache_cross_wave_vector_bus_control
        + E_system_memory_interconnect_control
```

每一笔数据移动只在跨越相应层级边界时计费一次。例如Macro输入寄存器写入属于Tile
端点，Tile间广播线属于Core，Core间链路属于System。

但有效吞吐率不能只由这个乘积决定，还受到以下因素限制：

- 层的K/N/P尺寸是否能填满资源；
- Tile内最多4条输出lane；
- Core内共享总线带宽；
- System共享内存和Core接入互连带宽；
- 权重与输入的复制或广播策略；
- 层间依赖和负载均衡。

因此完整的层次化DSE变量应逐级扩展：

| DSE层级 | 主要计算资源变量 | 同级外围变量或约束 |
|---|---|---|
| Tile | Macro类型与数量 | 输入/输出寄存器、跨Macro psum；广播、控制与归约延迟固定为0 |
| Core | Tile类型与数量 | Core缓存、跨wave向量模块、总线、广播、完整网络控制 |
| System | 完备Core类型与数量 | Core接入互连、共享内存接口、DRAM/DMA和任务分发 |

## 9. 一个卷积层的层次化映射示例

考虑：

```text
K = 576
N = 128
P = 3136
N_blocks = 128 / 8 = 16
```

一种可能的执行方式为：

```text
Macro：计算某个K片段到8个输出通道的部分和

Tile：多个Macro共同覆盖K=576；最多4条lane同时处理4个N块；
      在每个K wave内部完成跨Macro归约

Core：若启用4个Tile，可将16个N块平均分为每Tile 4块；
      输入向4个Tile广播，权重按输出通道分发；
      Core缓存和向量模块保存并累加多个K wave，随后继续执行下一层

System：若配置多个Core，每个Core独立执行一个完整网络实例；
        System负责共享内存接口、任务分发和多Core集成
```

这个例子说明，一个神经网络层在单个Core内部跨越Macro、Tile和Core三个计算层级；
System只把整个网络实例作为任务交给某个Core，并提供共享内存与集成接口。

## 10. 当前实现状态

| 层级 | 当前状态 | 已建模重点 | 尚未覆盖重点 |
|---|---|---|---|
| Macro | 已有候选库 | 含内部外围的黑盒能效、面积、周期、VMM | 权重编程能耗、不同形状独立时序 |
| Tile | 已实现DSE | Macro组成、K/N调度、端点寄存器、跨Macro psum能耗、权重装载延迟 | 未计Tile控制/同步、归约延迟、时钟与泄漏 |
| Core | 计算层DSE已实现 | Tile数量、N并行、512-bit总线、广播、事件控制、Core SRAM容量/访问、跨wave向量累加 | 输入/权重/激活缓存流量、完整网络控制、非CIM算子、时钟与泄漏 |
| System | 计算资源DSE已实现 | 同构Core数量、计算面积、Macro数、能效比/单样本延迟Pareto、饱和吞吐/功率 | 接入互连、共享内存/DRAM、任务队列、实际带宽与外围能耗 |

当前工作负载只统计ResNet-18的21个Conv/FC计算，尚未包含BN、ReLU、池化、残差加法
和完整激活依赖图。这些完整网络功能应补入Core控制、缓存和向量处理模型，而不是留给
System。因此现有结果仍属于带Core psum外围约束的计算资源DSE，不能直接视为完备
Core或完整芯片评估。

Core搜索采用分层贝叶斯剪枝：外层从25个代表Tile类型中选择子集，内层对每个被选中的
Tile精确枚举 `1..8 Tile/Core`。默认评估12个Tile类型，因此最多精确评估96个Core计算
资源配置，而不是穷举200个组合。BO不会在Tile数量维度预测或剪枝；所有已评估Core的
全网Pareto状态都会继续传递到System层，避免逐层只保留单一最优点。

## 11. Innovation 1 当前 Tile DSE 搜索规范

本节集中记录创新点一当前正式使用的 Tile 级 Macro 组合搜索条件、约束、目标和方法。
它只适用于当前的 Tile-only 研究阶段；Core/System 的层级搜索属于后续扩展。若本节与
前文较早的通用 Core/System 设想冲突，以本节和 `experiments/docs/tile_DSE/` 下的冻结规范
为当前 Innovation 1 实验口径。

### 11.1 搜索条件

| 项目 | 当前设置 |
|---|---|
| 搜索对象 | 单个数字 SRAM-CIM Tile |
| Macro 类型 | `32x8`、`64x8`、`128x8`、`256x8`、`512x8` |
| 输入/权重精度 | 8 bit / 8 bit |
| 工作频率 | 400 MHz |
| Macro bit-cycle | 2.5 ns |
| 参考峰值算力 | 0.4608 TOPS |
| 峰值算力窗口 | `0.4608 * (1 +- 20%)`，即 0.36864--0.55296 TOPS |
| CNN 工作负载 | ResNet-18、ResNet-50 等，batch size = 1 |
| Transformer 工作负载 | ViT-B/16 或 ViT-L/16 的 GEMM/CIM 部分 |
| Tile 输出接口 | 最多 4 条物理 lane，固定 32 个输出/psum 寄存器 |
| 特征总线 | 512 bit/cycle |
| 权重总线 | 64 bit/cycle |
| Macro 静态功耗比例 | 5% |
| 总线动态能量 | 当前为 0 J/bit，仅统计总线流量和周期 |
| 必测研究锚点 | `1x64x8 + 1x512x8` |
| 等算力同构基线 | `9x64x8` |

Macro 的峰值算力、VMM 延迟、满行能量、面积和权重写入能量来自编译器表征及
SRAM/寄存器组件模型。DSE 不重新搜索 Macro 内部 bitcell、工艺或精度。

### 11.2 搜索约束

#### Macro 组合约束

- 每种 Macro 的数量是非负整数，至少包含一种 Macro。
- 一个 Tile 最多包含两种 Macro 类型，因此既允许同构组合，也允许双类型异构组合。
- 组合峰值算力必须落入 0.36864--0.55296 TOPS。
- 面积和容量不设硬上限，只作为后续评价和诊断字段。
- `64+512` 必须实际评价，但不能在候选生成、调度剪枝或目标函数中获得优待。

在上述条件下，理论候选池为 138 个：14 个同构候选和 124 个双类型异构候选。

#### Tile 映射约束

- 不同 lane 沿输出维度 `N` 并行；同一 lane 内的 Macro 沿归约维度 `K` 协同。
- 同一 Macro 同一时刻只能执行一个 VMM，并且只能属于一条 lane。
- Macro 可以按实际 `K` 维只开启部分阵列行。
- 闲置 Macro 不产生动态计算能量，但仍支付静态功耗。
- 同 wave 内的跨 Macro psum 归约计入动态能量。
- 跨 wave psum 保存在 Tile 输出寄存器中。

#### 总线与时序约束

- 共享入口每周期只能处于 `FEATURE`、`WEIGHT` 或 `IDLE` 模式。
- `FEATURE` 周期只能传特征，带宽为 512 bit/cycle。
- `WEIGHT` 周期只能传权重，带宽为 64 bit/cycle，每周期最多传输一行 64-bit 权重。
- 特征与权重不能在同一周期同时传输。
- 同一 bank 内禁止权重装载与 Macro VMM 重叠。
- 总线传输可以与计算重叠，但必须满足数据依赖。
- 一个 N-block 的 K 维权重装载一次，并在输出位置间复用。

#### 有界状态搜索约束

为避免映射状态爆炸，层内和跨层采用有界 Pareto 动态规划：

| 参数 | 设置 |
|---|---:|
| 层内 latency cap ratio | 4.0 |
| 层内 Pareto epsilon | 0.001 |
| 层内 state cap | 4096 |
| 跨层 aggregate epsilon | 0.001 |
| 跨层 aggregate state cap | 512 |

因此正式结果应标记为 bounded-DP Pareto，而不是无界的精确调度枚举。

### 11.3 搜索目标

网络级主 Pareto 同时最小化：

```text
minimize E_network
minimize T_network
```

其中 `E_network` 是完整网络推理能量，`T_network` 是完整网络推理延迟。

同时在实际评价结果上生成三目标 E-T-A Pareto：

```text
minimize E_network
minimize T_network
minimize A_tile
```

Tile 面积口径为：

```text
A_tile = sum(A_macro)
       + sum(2 * rows * A_8bit_input_register)
       + 32 * A_25bit_output_register
```

面积不进入层内调度 DP，因为同一 Tile 的面积不随层调度变化；E-T-A Pareto 在网络级
结果上后处理生成。

EDP、EDAP、TOPS/W、TOPS/mm2、Macro 数量、容量、利用率和总线流量是摘要或诊断指标。
其中 EDP 主要用于 Bayesian 采集，不作为最终独立 Pareto 维度。

跨网络筛选使用 Pareto regret，而不是只看某个网络的最低 EDP。冻结判据如下：

| 判定项 | 阈值 |
|---|---:|
| 单网络近优 regret | <= 5% |
| 跨网络通过率 | >= 80% |
| 跨网络中位 regret | <= 3% |
| 最坏网络 regret | <= 10% |
| 严格命中 E-T 或 E-T-A Pareto 的网络比例 | >= 50% |

### 11.4 搜索方法

搜索由“候选枚举、Bayesian 选点、内层物理调度、跨网络判定”四步组成：

1. **候选组合枚举**：递归枚举五种 Macro 的整数数量，累加峰值算力；超过上限时
   提前剪枝，最终保留至少一种 Macro、最多两种 Macro 类型且位于算力窗口内的组合。

2. **Bayesian 外层选点**：138 个理论候选不全部进行昂贵的网络级评价，正式只实际评价
   48 个候选：初始覆盖 16 个、Bayesian 采集 24 个、预测前沿验证 8 个，每批最多 4 个。
   初始集合包含全部 14 个同构候选、`9x64`、`64+512` 以及用于扩大结构覆盖的异构候选。

3. **候选特征和 GP 预测**：每个候选使用五种 Macro 数量、相对容量和 Macro 总数作为
   特征。Gaussian Process 分别预测 `log(energy)`、`log(latency)` 和 `log(EDP)`，采集
   函数同时考虑预测值、不确定度和候选多样性。未实际评价候选的预测值不进入最终 Pareto、
   regret 或跨网络合格名单。

4. **内层物理约束调度**：对选中的组合搜索 lane 数、Macro-to-lane 分组、N-block 分配、
   K-wave 划分、部分行激活以及权重/特征时序，并计算 Macro 动态/静态能量、寄存器能量、
   psum 归约能量、权重写入能量、总线周期和层延迟，保留层内 E-T Pareto 状态。

5. **网络级聚合与跨负载判定**：CNN 按层顺序聚合；Transformer 额外搜索投影 bank 与
   注意力 bank 的 Macro 分工和 lane 分配，稳态延迟取投影、注意力和共享总线路径的最大值。
   最后生成 E-T/E-T-A Pareto，计算单网络和跨网络 regret，判断 `64+512` 是否属于实测
   候选中的稳健近优组合。

当前结论的范围应限定为“Bayesian 实际评价的 48 个候选中的稳健近优结构”，不能外推为
138 个理论候选中的全局最优。

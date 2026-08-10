# SRAM-CIM Tile DSE Handoff

## 2026-07-22: 400 MHz operating point

- `experiments/SRAM_config/system_digital_sram_4x4.yaml` now sets
  `macro_cycle_period = 2.5e-9 s` (400 MHz).
- Macro VMMs, the default Core shared bus, and the default System NoC use this shared
  period. Existing DSE CSVs generated at `1.4278 ns` are historical and must be
  regenerated before use with the 400 MHz configuration.

## 1. 当前目标

当前工作已经从早期 GLB/NoC/System 能耗调试转向 **Macro -> Tile -> Core -> System**
层次化设计空间探索。现阶段重点是 Tile 级计算资源组成，并准备进入 Core 级搜索。

最终研究目标是证明：

> 由不同规约深度 Macro 组成的异构 Tile，在完整网络和系统资源约束下，可以比单一
> Macro 类型组成的同构 Tile 获得更好的能量、延迟和面积折中。

当前结论：Tile DSE 的简化架构和调度模型已经基本自洽，可以开始设计 Core 调度器；
但正式运行 Core Pareto 前，建议先把剩余 Tile 候选补全到 `137/137`，再冻结 Tile 库。

旧版 HANDOFF 中关于 T0 GLB 输出流量的诊断已经不是当前主线，不要据此继续工作。

## 2. 当前层次边界

当前 `sram_tile_dse.py` 评估的是 **一个 Tile** 在完整 ResNet-18 上的行为：

```text
一个 Tile 组成
  -> 每层枚举 Macro 分组和输出块分配
  -> 保留层内能量-延迟 Pareto 调度
  -> 组合 ResNet-18 的21层
  -> 得到完整网络的能量、延迟、面积和资源状态
```

虽然 `system_digital_sram_4x4.yaml` 仍含4-Tile系统层次、line FIFO、GLB等旧系统节点，
当前 Tile DSE 不会把结果按4个 Tile 缩放，也不计算 GLB/NoC/DRAM。YAML 在此阶段主要
提供 Macro 表、物理参数、寄存器契约和 ResNet-18 映射所需的架构外壳。

## 3. 工作负载

- AccelForge 自带 ResNet-18 workload。
- Batch size：1。
- 评估21个 Conv/FC einsum，名称为 `T0` 到 `T20`。
- 总 MAC 数：`1,814,073,344`。
- 网络层顺序执行，层能量和延迟逐层相加。
- 未建模 ReLU、BN、pooling、residual add 和完整激活依赖图。

## 4. Macro 类型

当前只探索8列 Macro，输入和权重均为8 bit：

| Macro | Rows | Columns | TOPS/W | TOPS/mm2 |
|---|---:|---:|---:|---:|
| `m32x8` | 32 | 8 | 6.08 | 2.23 |
| `m64x8` | 64 | 8 | 7.66 | 2.20 |
| `m128x8` | 128 | 8 | 11.28 | 2.31 |
| `m256x8` | 256 | 8 | 13.13 | 2.28 |
| `m512x8` | 512 | 8 | 17.03 | 2.24 |

共同参数：

- 工艺：22 nm。
- 电压：0.72 V。
- 周期：1.4278 ns。
- 输入位串行：1 bit/cycle。
- 一个8-bit完整 VMM：8 cycles，即11.4224 ns。
- Macro 数据源：`experiments/docs/SRAM_DATA/cim_macro_efficiency.csv`。

## 5. Tile 组成约束

正式约束已经写入 `experiments/SRAM_config/system_digital_sram_4x4.yaml`：

```yaml
max_macro_types_per_tile: 2
tile_output_lanes: 4
macro_weight_load_rows_per_cycle: 1
```

含义：

- 每个 Tile 最多包含两种不同 Macro 类型。
- 同一类型的 Macro 实例数不限，仍受容量和面积边界约束。
- 例如 `3xm64x8+1xm256x8` 满足“两种类型”规则。
- 每个 Tile 最多4条输出 Lane。
- 每层激活1到4条 Lane。
- 同一 Lane 内的 Macro 结果相加。
- 不同 Lane 的结果不相加，计算不同输出通道块。
- 当前允许每层重新选择 Macro 到 Lane 的分组，连接重配置成本为0。
- 用户明确决定暂不设置 `max_lane_fan_in`。

当前调度器把 Lane 数作为 DP 资源参与中间状态剪枝，不是生成完调度后再过滤。

## 6. 候选比较边界

参考 Tile：

```text
9 x m64x8
容量：4608个逻辑权重
Macro面积：0.366743496 mm2
寄存器面积：0.024137525 mm2
Tile总面积：0.390881021 mm2
```

候选容量和 **Tile总面积** 都必须在参考值的 +/-20% 内：

```text
容量：3686.4 到 5529.6 个逻辑权重
面积：0.312704817 到 0.469057225 mm2
```

候选池：

- 总数：137。
- 同构：14。
- 双类型异构：123。
- Macro 数范围：1到21。

`3xm64x8+1xm256x8` 虽然满足类型规则，但容量低22.2%、总面积低约24.0%，所以仍被
当前 +/-20% 比较边界排除。不要把这种排除误认为类型数量限制。

## 7. Tile 内映射语义

每个 Macro 一次产生8个输出，因此：

```text
N_blocks = ceil(N / 8)
```

调度器枚举 Macro group 作为一条 Lane：

- 同 Lane 的 Macro 可以共同覆盖 K，结果执行部分和累加。
- 如果 group 总行数小于 K，同一组 Macro 会经过多个 VMM wave。
- 不同 Lane 并行处理不同 `N_blocks`。
- 同一 Lane 可顺序处理多个输出块。
- 允许部分 Macro 空闲。
- 所有保留调度满足 `1 <= active_lane_count <= 4`。

`k_parallel`、`n_parallel`、`mixed` 只是调度结果标签：

- `k_parallel`：只有一条 Lane。
- `n_parallel`：多条 Lane 且每条 Lane 只占一个 Macro。
- `mixed`：多条 Lane 且至少一条 Lane 含多个 Macro。

它们不是三套互斥 mapper。

## 8. 输入和输出寄存器

每个 Macro 的 Tile 级寄存器结构：

```text
输入：2 banks x macro_rows entries x 8 bit
输出：1 bank x 8 entries x 25 bit
```

- 输入采用 ping-pong。
- 输出不采用 ping-pong。
- 25-bit输出寄存器保存 Macro 局部部分和。
- 更宽的跨 Macro/多 wave psum 由 Tile 向量累加器处理。

寄存器使用 AccelForge 的 `AladdinRegister` 在22 nm下缩放估计。

动态能量假设，每次物理 Macro VMM invocation：

```text
输入：一个活动 Bank 的全部物理 rows，各读取一次、写入一次
输出：8个25-bit entry，各读取一次、写入一次
```

动态寄存器能量/调用：

| Macro | Input reg | Output reg | Total |
|---|---:|---:|---:|
| `m32x8` | 1.560 pJ | 1.218 pJ | 2.778 pJ |
| `m64x8` | 3.119 pJ | 1.218 pJ | 4.338 pJ |
| `m128x8` | 6.238 pJ | 1.218 pJ | 7.457 pJ |
| `m256x8` | 12.477 pJ | 1.218 pJ | 13.695 pJ |
| `m512x8` | 24.953 pJ | 1.218 pJ | 26.172 pJ |

寄存器面积已经进入 Tile 面积和 +/-20% 面积过滤：

```text
A_tile = sum(A_macro)
       + sum(2 x rows x A_8bit_register)
       + N_macro x 8 x A_25bit_register
```

Aladdin 模型局限：来源为40 nm数据后缩放，不直接建模0.72 V，register leakage为0，
也不包含 Bank 固定开销、clock tree、ping-pong控制和物理布线。

## 9. 权重驻留和装载

采用权重驻留：每组权重在所有输出位置间复用。

当前简化权重装载模型：

- 每个 Macro 有独立权重装载端口。
- 每个周期装载一行，一行包含该 Macro 的8个权重。
- 只装载实际参与 K 规约的有效行。
- 同一 Macro 的装载和计算不重叠。
- 不同 Macro/Lane 可以并行装载。
- 权重装载能量为0。
- 权重装载延迟计入总 makespan。

一个 Lane 的输出块延迟：

```text
T_job = output_positions x T_VMM_compute + T_weight_load
```

层延迟：

```text
T_layer = max(各 Lane 的 jobs x T_job)
```

逐层结果应满足：

```text
weight_load_rows = K x ceil(N / 8)
latency = compute_only_latency + weight_load_added_latency
```

CSV 同时记录：

- `weight_load_values`
- `weight_load_rows`
- `weight_load_events`
- `weight_load_makespan`
- `weight_load_added_latency`
- `weight_load_energy = 0`

## 10. 当前能量、延迟和面积模型

能量：

```text
E_fabric = E_macro
         + E_input_register
         + E_output_register
         + E_psum
```

权重装载能量当前为0。

部分和位宽：

```text
psum_width = input_bits + weight_bits + ceil(log2(K))
```

部分和能量由22 nm `DigitalSramShiftAccumulator` 估计，向量宽度为8个输出。
psum 延迟仍为0，不进入 makespan。

面积：

```text
A_tile = A_macro + A_input_register + A_output_register
```

延迟：

```text
T_layer = VMM makespan + non-overlapped weight-load contribution
```

当前明确不包含：

- max Lane fan-in 约束。
- Lane 归约延迟和能量。
- psum 累加延迟。
- 权重装载能量。
- ping-pong控制和 Bank 固定开销。
- 输入广播、局部互连、多 Macro 同步。
- 控制器、时钟树和 leakage。
- line FIFO、GLB、NoC、DRAM。
- 激活函数、pooling 和 residual add。

这些是用户同意的当前简化边界，不要在未讨论的情况下突然加入固定成本。

## 11. 搜索算法

外层使用自定义 NumPy Gaussian Process 贝叶斯搜索；每个 Tile 内部仍使用精确/近似
Pareto DP 枚举 Macro grouping 和 N-block 分配。

当前参数：

```text
capacity tolerance: 20%
area tolerance: 20%
design budget: 48
initial designs: 16
batch size: 4
workers: 2
seed: 7
latency cap ratio: 2.0
local Pareto epsilon: 0.001 (0.1%)
aggregate state cap: 512 per Tile
```

初始点强制包含：

- `9xm64x8`
- `1xm64x8+1xm512x8`
- 全部14个同构候选

随后用 GP 预测最小能量、最小延迟、最小 EDP，并结合不确定性和批内距离采样。

注意：面积和 Macro 数量目前是 GP 输入特征，但不是独立预测目标。完成137个全枚举可以
避免贝叶斯采集漏掉 Core 面积预算下有价值的 Tile。

模型版本写入 checkpoint，旧模型结果会自动失效：

```text
energy: macro_psum_aladdin_register_v1
architecture: max_two_macro_types_four_output_lanes_v1
area: macro_plus_aladdin_register_area_v1
latency: vmm_plus_parallel_macro_row_weight_load_v1
```

## 12. 当前搜索进度

当前检查点：`48/137`。

- 同构样本：14，已全部覆盖。
- 异构样本：34。
- 含 `m32` 的已评估设计：28。
- 逐层调度 CSV 数据行：14,545。
- 全网聚合状态：10,151。
- 四资源 Pareto 状态：2,381。
- 能量-延迟 Pareto 状态：728。
- 架构摘要：48个 Tile。

所有48个设计均已验证：

- 覆盖21层。
- 每层活动 Lane 为1到4。
- 每个 Tile 最多两种 Macro。
- `weight_load_rows = K x ceil(N/8)`。
- 权重装载能量为0。
- 总延迟等于 compute-only 延迟加 weight-load-added 延迟。

## 13. 当前关键结果

完整 ResNet-18 极值：

| Objective | Tile | Energy | Latency |
|---|---|---:|---:|
| Minimum energy | `1xm128x8+1xm512x8` | 0.273843 mJ | 9.159303 ms |
| Minimum latency | `21xm32x8` | 0.635554 mJ | 4.075707 ms |
| Minimum EDP | `1xm32x8+5xm128x8` | 0.370959 mJ | 4.674046 ms |

两个最初基线：

| Tile | Minimum energy | Minimum latency |
|---|---:|---:|
| `9xm64x8` | 0.506317 mJ | 4.888387 ms |
| `1xm64x8+1xm512x8` | 0.279494 mJ | 6.785537 ms |

权重装载对两个基线最低延迟的增加：

```text
9xm64x8:                +0.232103 ms
1xm64x8 + 1xm512x8:    +0.896604 ms
```

`m32` 仍然频繁出现：

- 48个样本中28个包含 `m32`。
- 44个四资源 Pareto 架构中24个包含 `m32`。

寄存器面积和权重装载延迟已经削弱小 Macro 的优势，但独立装载端口和细粒度调度仍使
其在低延迟区域占主导。不要直接得出 `m32` 一定最优；仍缺控制、互连和 Lane fan-in
成本，但用户当前决定先不加入这些项。

高维 Pareto 的解释：

- 仅 Energy/Latency 前沿：728 states，14个 Tile 架构。
- Energy/Latency/Area 前沿：40个 Tile 架构。
- Energy/Latency/Area/Macro-count 前沿：44个 Tile 架构。

面积和 Macro 数量使多数 Tile 成为高维非支配点。进入 Core 前不能直接把2,381个状态
全部作为离散 Tile 类型组合，否则 Core 搜索空间会爆炸。

## 14. 结果文件

- `experiments/results/tile_DSE/tile_dse_bayesian_samples.csv`
  - 48个已评估 Tile 和模型版本。
- `experiments/results/tile_DSE/tile_dse_layers.csv`
  - 每个 Tile、每层的局部调度状态和能量/延迟分解。
- `experiments/results/tile_DSE/tile_dse_aggregate.csv`
  - 完整 ResNet-18 聚合状态。
- `experiments/results/tile_DSE/tile_dse_aggregate_pareto.csv`
  - Energy/Latency/Area/Macro-count 前沿。
- `experiments/results/tile_DSE/tile_dse_full_resnet_energy_latency_pareto.csv`
  - 二维能量-延迟前沿。
- `experiments/results/tile_DSE/tile_dse_architecture_summary.csv`
  - 每个 Tile 的最低能量、最低延迟和 knee 摘要。
- `experiments/results/tile_DSE/tile_dse_full_resnet_pareto.pdf`
  - 调度级矢量图。
- `experiments/results/tile_DSE/tile_dse_architecture_summary.pdf`
  - 架构级矢量图。

PNG 版本也存在于同一目录。`experiments/results/` 被 Git 忽略。

## 15. 关键代码

- `experiments/SRAM_config/system_digital_sram_4x4.yaml`
  - Macro 表、寄存器结构、Lane数、类型数和权重装载速率。
- `experiments/RUN/sram_macro_types.py`
  - 加载 Macro characterization 和 Tile 接口参数。
- `experiments/RUN/tile_DSE/sram_tile_dse.py`
  - Tile 候选枚举、寄存器面积/能量、权重装载时序、Lane受限调度 DP。
- `experiments/RUN/tile_DSE/run_sram_tile_dse.py`
  - 全枚举 runner 和完整网络状态聚合。
- `experiments/RUN/tile_DSE/run_sram_tile_bayesian_dse.py`
  - 贝叶斯外层搜索、并行执行和 checkpoint/resume。
- `experiments/RUN/tile_DSE/generate_sram_tile_dse_pareto.py`
  - CSV、PNG、PDF 可视化。
- `experiments/docs/tile_DSE/TILE_DSE_SEARCH_CONDITIONS.md`
  - 当前搜索条件的详细说明。
- `experiments/README.md`
  - 简要运行说明。

## 16. 可复现命令

继续从48个样本补全到全部137个 Tile：

```bash
third_party/accelforge/.venv/bin/python -u \
  experiments/RUN/tile_DSE/run_sram_tile_bayesian_dse.py \
  --design-budget 137 \
  --initial-designs 16 \
  --batch-size 4 \
  --workers 2 \
  --seed 7 \
  --capacity-tolerance 0.20 \
  --area-tolerance 0.20 \
  --latency-cap-ratio 2.0 \
  --local-pareto-epsilon 0.001 \
  --resume
```

重新生成图表：

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/tile_DSE/generate_sram_tile_dse_pareto.py
```

全枚举 runner 也可使用，但会重新计算全部候选：

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/tile_DSE/run_sram_tile_dse.py \
  --all-layers \
  --capacity-tolerance 0.20 \
  --area-tolerance 0.20 \
  --latency-cap-ratio 2.0 \
  --local-pareto-epsilon 0.001
```

## 17. 进入 Core DSE 前的下一步

### 必须先做

1. 将 Tile 搜索从 `48/137` 补全到 `137/137`。
2. 重新生成图表并检查极值和 Pareto 是否稳定。
3. 从完整 Tile 库构造约15到25个代表 Tile 的 coreset：
   - 保留全部同构候选。
   - 保留 `9xm64x8` 和 `1xm64x8+1xm512x8`。
   - 保留最低能量、最低延迟、最低 EDP 和 knee。
   - 对 Energy/Latency/Area 做 epsilon/grid 压缩。
4. 不要直接把全部全网聚合 schedule states 当作 Core 的 Tile 类型。

### 第一版 Core 模型建议

- 固定4个 Tile，延续此前用户选择。
- Core 可由多个 Tile 类型组成。
- Core 面积为各 Tile 总面积之和，先不加 Core 固定外围面积。
- 每层仍顺序执行，暂不做跨层流水。
- 用户已确定权重复制规则：

```text
若单个Tile能容纳一层全部权重：
    Tile间可以复制权重以提高并行度
否则：
    Tile尽量保存不同权重块，减少复制，提高权重复用率
```

- 第一版优先支持 Tile 间 N/P 并行，结果互不相加。
- 不要允许免费的跨 Tile K 并行。如果允许 K 划分，必须至少记录
  `inter_tile_psum_vectors` 并加入归约延迟；否则会人为偏好权重碎片化。
- Core 能量为活动 Tile 能量之和。
- Core 延迟为每层 Tile 调度 makespan，再逐层相加。
- GLB/NoC/DRAM 继续排除，以保持“计算资源分配优先”的研究边界。

## 18. 尚未解决但当前可暂缓的问题

- max Lane fan-in：用户明确要求当前不考虑。
- 权重装载能量：当前为0。
- Lane/跨 Macro 归约延迟。
- psum 累加延迟。
- Lane crossbar 或静态绑定成本。
- input broadcast、clock、control、leakage。
- Aladdin register 模型的电压和物理精度。
- BO 对面积/Macro-count 的独立多目标采集。

这些不是开始 Core 框架的阻塞项，但会影响最终论文级结论，需要后续灵敏度分析或明确
列为建模边界。

## 19. 工作树安全

当前工作树是 dirty 的：

```text
 M HANDOFF.md
 M experiments/README.md
 M experiments/RUN/sram_macro_types.py
 M experiments/SRAM_config/system_digital_sram_4x4.yaml
 m third_party/accelforge
?? experiments/RUN/tile_DSE/generate_sram_tile_dse_pareto.py
?? experiments/RUN/tile_DSE/run_sram_tile_bayesian_dse.py
?? experiments/RUN/tile_DSE/run_sram_tile_dse.py
?? experiments/RUN/tile_DSE/sram_tile_dse.py
?? experiments/docs/tile_DSE/TILE_DSE_SEARCH_CONDITIONS.md
```

不要 reset、checkout 或删除这些路径。`third_party/accelforge` 的修改可能是用户工作；
除非用户明确要求，否则不要改动或还原。`experiments/results/` 是生成结果且被 Git 忽略。

## 20. 最近验证

最近验证均通过：

- Python `py_compile` 通过。
- `git diff --check` 通过。
- 48个设计均覆盖21层。
- 活动 Lane 全部在1到4之间。
- 每个 Tile 的 Macro 类型数不超过2。
- 所有逐层权重装载行数正确。
- 权重装载能量全部为0。
- 聚合延迟分解一致。
- 两个 PDF 均为一页 PDF 1.4 矢量文件。

下一段对话应从“是否先补全137个 Tile，还是先搭 Core 数据结构”继续。推荐先补全
137个 Tile，然后开始 Core DSE。

## 21. 2026-07-21 Core-v1更新（覆盖第17节旧建议）

用户重新明确了研究主线：探索层级计算资源配置，即 Tile 内 Macro 数、Core 内 Tile 数、
System 内 Core 数。当前不补全137个 Tile，而是使用已评估48个候选中选出的25个代表
Tile。Core 内 Tile 同构，Tile 之间只做输出维并行。

Core-v1不再固定4个物理Tile。默认搜索 `tile_count=1..8`，并将物理Tile数与每层活动
Tile数分开。特征和权重共享一条512 bit/cycle总线，总线周期标称等于Macro周期
1.4278 ns。特征可向活动Tile广播；权重对应不同输出通道，不复制。输出直接保留在
Tile输出寄存器，Core层不计算输出汇聚。K不跨Tile切分，因此无跨Tile psum归约。

新增Core开销：

- AccelForge `AladdinRegister` 源端读和 `Wire` 广播树动态能耗；
- 特征与权重共享总线beat和带宽下界；
- Aladdin register/counter/comparator事件控制器；
- 一拍dispatch和一拍layer-finish同步；
- 默认1 MiB、256-bit、4-bank Core SRAM的psum容量、跨wave读写能耗和吞吐；
- 一个8x32-bit Core向量累加器的跨wave能耗和吞吐。

当前通信时序是允许ping-pong完美重叠的下界：

```text
T_core_layer = max(T_tile_makespan,
                   T_shared_bus,
                   T_core_psum_cache,
                   T_core_vector) + T_dispatch + T_finish
```

Core-v1代码为 `RUN/sram_core_dse.py` 和 `RUN/run_sram_core_dse.py`。聚合状态已支持动态
`A1..A_tile_count`，不再硬编码四个Tile。System层尚未实现；下一层应使用Core coreset，
搜索物理 `core_count` 和每层活动Core数，继续保持同构Core与输出维并行。

### Tile内部外围的最新决定

用户决定当前不增加Tile内部外围成本：

- Tile内Macro数量较少，不计算Macro间输入广播能耗、延迟和面积；
- 活动Macro在所需数据全部就绪后统一开始计算，不计算Tile控制与同步开销；
- psum归约与Macro计算重叠，归约延迟视为被Macro计算掩盖；
- psum动态能耗继续保留；
- 权重装载延迟继续保留，权重装载能耗仍为0。

因此无需回到Tile层增加 `TilePeripheralCostModel`。当前48个Tile结果可以继续用于代表
Tile选择，下一步直接运行正式Core-v1 DSE。

### 2026-07-21 最终层级边界（覆盖本节此前的System/psum描述）

- 单个K wave内，同一Tile中多个Macro产生的部分和由该Tile完成归约；
- 如果K过大而需要多个wave，Tile输出每个wave的lane部分和，跨wave累加由Core向量
  处理模块完成，中间psum保存在Core级缓存；
- Tile之间只做N并行，不对同一输出块切K，因此没有跨Tile psum归约；
- Core包含完整控制模块、Core级缓存和向量处理模块，是可独立执行完整神经网络的最小
  完备模块；
- System只集成多个完备Core，不把一个网络的层或张量拆到多个Core。多个Core默认执行
  独立网络实例，增加Core数主要提升系统吞吐率而非单样本延迟。

当前代码已经按该边界拆分 `cross_macro_psum_vectors` 和
`cross_wave_psum_vectors`：前者的动态能耗属于Tile，后者驱动Core SRAM读写和8-lane
向量累加。Core调度同时检查psum峰值容量、缓存访问宽度和向量位宽，并把缓存/向量
吞吐加入层延迟下界。默认Core SRAM为1 MiB、256-bit、4-bank，参数可由Core runner
调整。

当前Core SRAM只计跨wave psum容量和动态流量；输入、权重、激活、最终输出的缓存占用
与访问，非CIM算子，泄漏及时钟能耗尚未建模。正式Core资源结论前应对缓存容量、带宽
和向量吞吐做灵敏度分析，并补齐完整网络数据流。

### 2026-07-21 Core面积口径更新

当前重点是计算资源分布，Core CSV和Pareto只使用：

```text
core_compute_area_mm2 = core_tile_count * tile_area_mm2
```

不再输出 `core_area_mm2`、`core_overhead_area_mm2`、SRAM/向量/控制器面积拆分，外围面积
也不参与Core资源Pareto。Core SRAM和向量模块仍保留容量、位宽、吞吐和动态能耗约束。

### 2026-07-21 Core贝叶斯搜索更新

正式Core DSE不再穷举25个代表Tile与8个Tile数量的全部200个组合。新增
`RUN/run_sram_core_bayesian_dse.py`：

- BO外层从25个代表Tile类型中选择子集，默认预算12个、初始8个、batch为2；
- 每评估一个Tile类型时，内层精确枚举 `1..8 Tile/Core`，不对整数Tile数量做预测；
- 代理目标包含全网最小能耗、延迟、EDP、计算面积和Macro数；
- 每个Tile完成后保存原始layer/aggregate checkpoint，每批更新全局Pareto；
- 支持 `--resume`，模型和Core资源参数不兼容时自动重新开始；
- System层直接使用所有已评估Core的全网资源Pareto，不逐层选择唯一最优Core。

正式默认运行已完成：从25个代表Tile中评估12个Tile类型，并对每个Tile精确评估
`1..8 Tile/Core`，共96个Core架构。结果包含17,877个layer状态、39,366个完整网络
aggregate状态和8,834个四目标Core资源Pareto状态。12个采样Tile类型均至少贡献一个
Pareto状态。采样顺序为：

```text
9xm64x8
4xm128x8
2xm256x8
1xm512x8
1xm64x8+1xm512x8
15xm32x8
5xm32x8+2xm256x8
21xm32x8
1xm32x8+5xm128x8
7xm32x8+7xm64x8
2xm128x8+1xm256x8
5xm32x8+1xm512x8
```

结果文件为 `experiments/results/core_dse_*.csv`，checkpoint为
`core_dse_bayesian_samples.csv`。完成态使用同参数 `--resume` 不会重复调度或重算Pareto。

### 2026-07-21 System计算资源DSE

新增 `RUN/sram_system_dse.py`、`RUN/run_sram_system_dse.py` 和对应测试。System由多个
同构完备Core组成，每个Core执行独立网络实例；默认饱和任务队列使全部物理Core活动。
System不增加SRAM、NoC、DMA、控制、外围面积或外围能耗。

默认精确枚举 `1..32 Core/System`，指标为：

```text
system_compute_area = core_count * core_compute_area
system_macro_count = core_count * core_macro_count
single_inference_latency = core_latency
energy_per_inference = core_energy
saturated_throughput = core_count / core_latency
saturated_compute_power = core_count * core_energy / core_latency
energy_efficiency_TOPS_per_W = 2 * MACs / core_energy / 1e12
```

正式运行读取8,834个Core Pareto输入，生成282,688个System状态。System优化目标修正为：
在1、2、4、8、16、32 mm2计算面积硬预算内，最大化能效比TOPS/W并最小化单次推理
延迟。饱和吞吐仍输出，但不再参与Pareto判定。

独立Core复制不改变能效比和单次延迟，因此每个预算内对每种Core状态保留可容纳的最大
Core数作为规范化代表。这不是第三个优化目标。新双目标前沿共3,238个预算Pareto状态，
并为每个预算输出最大能效比端点。

结果文件：

```text
experiments/results/system_dse_states.csv
experiments/results/system_dse_area_budget_pareto.csv
experiments/results/system_dse_area_budget_best.csv
```

`core_source_index`精确指向 `core_dse_aggregate_pareto.csv` 的源行，用于从System结果回溯
Core与Tile调度。当前System结果是计算资源模型，不是含共享带宽和片外存储的芯片模型。

### 2026-07-21 System P×N联合推理模型（覆盖此前独立Core复制模型）

System中的Core功能完备，但可联合执行同一个网络。每层枚举 `C_P x C_N`，每个Core负责
`[P_tile,N_tile] x 完整K`，禁止跨Core split-K。权重按N切分并在P组间复制；输入按P切分
并在N组间多播。每层输出写回System共享SRAM。

固定System外围为4 MiB、512-bit、16-bank共享SRAM和512-bit/cycle 2D mesh NoC。共享
SRAM对唯一输入/权重只读取一次，NoC按确定性路由多播树bit-hop计费；Core缓存收发、
System控制同步和Macro权重SRAM行写也计入动态能量。Macro权重写采用同容量、64-bit写宽
单bank SRAM的写能耗。面积仍只输出 `core_count * core_compute_area`。

`RUN/sram_system_dse.py`、`RUN/run_sram_system_dse.py` 已实现该模型。小规模验证使用1个
Core架构和 `1..2 Core/System`，生成298个层状态和127个全网状态，输出位于
`/tmp/cim_system_pn_smoke`。正式DSE未运行。此前Core重算在checkpoint阶段中断，当前
`experiments/results/core_dse_*` 和旧System CSV不能视为完整正式结果。

### 2026-07-22 峰值算力约束更新

整体DSE主约束改为最大System峰值算力，当前第一版固定为：

```text
peak_compute_budget_TOPS = 4.0
```

Macro峰值算力统一采用：

```text
2 * rows * columns / VMM_latency
```

Tile、Core和System峰值算力分别为Macro峰值算力的组成和层级复制结果。Core内Tile数和
System物理Core数默认由4 TOPS预算动态推导，面积只作为输出和可选sanity filter。

Core Bayesian runner默认读取全部48个已评价Tile，选择12个Tile类型进行贝叶斯剪枝；每个
Tile类型精确枚举4 TOPS预算允许的所有Core Tile数量。System runner默认从Core峰值算力
推导合法Core数量，逐层搜索`P x N`，输出：

```text
system_dse_peak_compute_pareto.csv
system_dse_peak_compute_best.csv
```

峰值约束代码已完成单元测试和小规模System验证；正式Core/System DSE尚未按4 TOPS重新运行。

### 2026-07-23 Tile-only等算力搜索（当前主线）

当前工作范围已主动收敛为**独立Tile DSE**。暂停Core和System搜索，既有48个Tile、
Core/System CSV、Pareto图和任何700.38 MHz结果均为历史数据，不能用于当前结论。

#### 已确认的实验定义

```text
workload                 = ResNet-18, batch_size = 1, 21 Conv/FC layers
operating frequency      = 400 MHz
bit-cycle                = 2.5 ns
Tile peak-compute budget = 0.4608 TOPS
Macro types per Tile     = 1 or 2
active Tile output lanes = 1..8 (hardware maximum = 8)
objectives               = minimize full-network energy and latency
derived metric           = EE = fixed ResNet-18 operations / energy
not optimized            = area, Core topology, System topology, peripheral variants
```

Tile峰值算力必须**严格等于**0.4608 TOPS，而不是仅小于上限。该预算由两个等算力对照
组合确定：

```text
9xm64x8                 = 9 * 0.0512 TOPS = 0.4608 TOPS
1xm64x8 + 1xm512x8     = 0.0512 + 0.4096 TOPS = 0.4608 TOPS
```

目标是在同等Tile峰值算力下验证`1xm64x8 + 1xm512x8`是否位于全网络能量-延迟Pareto前沿。
若成立，结论必须表述为“在0.4608 TOPS等算力Tile预算下的异构Tile优势”，不能外推为所有
Tile尺寸下的全局结论。

当前模型中，给定Tile微架构后的工作负载EE上界不是单纯的Macro峰值TOPS/W，而是：

```text
EE_max(model) = ResNet-18 fixed operations / minimum allowed Tile-schedule energy
```

即由工作负载形状、合法K/N调度、寄存器、部分和和权重写事件共同决定；它不包含未来Core/
System通信、缓存、NoC和泄漏能耗。

#### 已完成的代码和文档变更

- `experiments/SRAM_config/system_digital_sram_4x4.yaml`
  - `macro_cycle_period = 2.5e-9`，即400 MHz；
  - `tile_output_lanes = 8`。
- `experiments/RUN/tile_DSE/sram_tile_dse.py`
  - `enumerate_tile_designs()`改为按`peak_compute_budget_tops`和严格容差枚举；
  - 移除了候选合法性中的旧容量/面积`+-20%`窗口；
  - 面积仍作为诊断字段输出，但不参与Tile候选资格。
- `experiments/RUN/tile_DSE/run_sram_tile_dse.py`
  - 新参数：`--tile-peak-compute-tops`（默认0.4608）与
    `--peak-compute-tolerance-tops`；
  - 输出使用`experiments/results/tile_DSE/tile_only_dse_*.csv`前缀，避免覆盖历史结果。
- `experiments/RUN/tile_DSE/run_sram_tile_bayesian_dse.py`
  - 特征和全局Pareto不再包含面积或Macro数量支配关系；
  - 同样使用`tile_only_dse_*.csv`检查点和输出；
  - 初始点仅在当前算力预算下存在时保留`9xm64x8`和`1xm64x8+1xm512x8`，支持其它预算。
- 新增`experiments/RUN/tile_DSE/test_sram_tile_dse.py`。
- 已更新`experiments/docs/tile_DSE/CURRENT_DSE_SPECIFICATION.md`和`experiments/README.md`的Tile-only说明。

#### 当前候选池与验证状态

在400 MHz、0.4608 TOPS严格等算力条件下，理论候选池为**24个Tile**，包含上述两个对照
组合。候选数很小，正式运行应穷举全部24个，不需要Tile层贝叶斯剪枝；
`run_sram_tile_dse.py --all-layers --tile-peak-compute-tops 0.4608`是推荐生产入口。

已通过：

```text
26 unit tests passed
py_compile passed
git diff --check passed
```

CACTI依赖仍会输出既有的`ResourceWarning`，不影响测试通过。

曾以`run_sram_tile_bayesian_dse.py`尝试完整评估24个候选，当前检查点已完成7个设计并写入：

```text
experiments/results/tile_DSE/tile_only_dse_bayesian_samples.csv
experiments/results/tile_DSE/tile_only_dse_layers.csv
experiments/results/tile_DSE/tile_only_dse_aggregate.csv
```

长运行会话在环境中断前未生成`tile_only_dse_aggregate_pareto.csv`，因此这些文件是**不完整
检查点，不是正式结果**。恢复命令如下：

```bash
third_party/accelforge/.venv/bin/python -u \
  experiments/RUN/tile_DSE/run_sram_tile_bayesian_dse.py \
  --design-budget 48 --initial-designs 1 --batch-size 1 --workers 1 \
  --tile-peak-compute-tops 0.4608 --local-state-cap 4096 --resume
```

如果当前环境继续中断长Bayesian runner，应改用`run_sram_tile_dse.py --all-layers`逐层穷举，
或为复杂候选增加可审计的调度状态上限；不能把7个检查点直接画成最终Pareto。

### 2026-07-23 Tile-only全量结果（最新状态）

24个严格等算力候选已经全部评价完成。由于复杂的`m32/m64`组合会使未受限层内DP占用
超过6 GiB内存，当前正式结果统一使用：

```text
local Pareto epsilon       = 0.001
local scheduling state cap = 4096
aggregate state cap        = 512
latency cap ratio          = 2.0
```

层内状态上限采用按已完成输出块进度分层的确定性流式采样；CSV字段
`local_schedule_state_cap_applied`记录某层是否触发。24个设计中23个至少有一层触发，因此
当前结果是统一搜索预算下的**近似Pareto**，不能表述为精确全调度枚举。模型版本为
`equal_peak_compute_streaming_bounded_dp_max_two_macro_types_eight_output_lanes_v4`。

正式结果：

```text
evaluated Tile designs       = 24 / 24
layer schedule rows          = 6,720
full-network aggregate rows  = 8,030
energy-latency Pareto states = 804
Pareto Tile architectures    = 18
```

全网络架构极值：

| Objective | Tile | Energy | Latency |
|---|---|---:|---:|
| Minimum energy | `1xm64x8+1xm512x8` | 0.280650 mJ | 15.848178 ms |
| Minimum latency | `18xm32x8` | 0.634548 mJ | 8.326208 ms |
| Minimum EDP | `1xm64x8+2xm256x8` | 0.330240 mJ | 9.969450 ms |

`1xm64x8+1xm512x8`贡献245个全局Pareto状态并给出全局最低能量；`9xm64x8`贡献16个
全局Pareto状态。由此可支持的限定结论是：在400 MHz、0.4608 TOPS严格等算力、当前
Tile能量/延迟模型和bounded-DP搜索预算下，指定异构Tile位于ResNet-18能量-延迟前沿。

结果审计已通过：24个候选各覆盖21层，活动lane为1到8，峰值算力均为0.4608 TOPS，
`weight_load_rows = K * ceil(N/8)`，逐层与聚合能量/延迟分解一致。28个DSE单元测试、
`py_compile`和`git diff --check`均通过。当前继续保持Tile-only范围，不启动Core/System DSE。

### 2026-07-23 Tile DSE目录整理

Tile级文件已按职责集中到以下目录：

```text
experiments/RUN/tile_DSE/       Python实现、runner、测试和可视化
experiments/results/tile_DSE/   历史及当前Tile DSE结果
experiments/docs/tile_DSE/      当前规范和搜索条件
```

共享的Macro库、AccelForge加载器和组件模型仍保留在`experiments/RUN/`，Core/System通过
`tile_DSE.sram_tile_dse`导入共享的Tile数据结构。新的正式运行入口为：

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/tile_DSE/run_sram_tile_dse.py \
  --all-layers --tile-peak-compute-tops 0.4608 --local-state-cap 4096
```

### 2026-07-23 Tile算力窗口与4-lane配置（当前最新定义）

用户取消了严格等于0.4608 TOPS的限制。当前把0.4608 TOPS解释为搜索中心，允许峰值算力
相对偏差`+-20%`，同时把每层活动Tile输出lane硬件上限从8降为4：

```text
peak-compute reference       = 0.4608 TOPS
relative tolerance           = 0.20
theoretical window           = [0.36864, 0.55296] TOPS
actual discrete window       = [0.384, 0.5376] TOPS
active Tile output lanes     = 1..4
Macro types per Tile         = 1 or 2
```

Macro峰值算力以0.0256 TOPS为粒度。新候选池共138个，其中14个同构、124个双类型异构，
实际分布在7个峰值算力档位：0.384、0.4096、0.4352、0.4608、0.4864、0.512和0.5376 TOPS。
Macro实例数为1到21，逻辑权重容量为3840到5376 values。

代码接口与输出隔离：

- `enumerate_tile_designs()`使用`peak_compute_reference_tops`、
  `peak_compute_tolerance_ratio`和`peak_compute_atol_tops`；
- 两个Tile runner默认`--peak-compute-tolerance-ratio 0.20`；
- YAML中的`tile_output_lanes = 4`；
- 新模型版本为
  `peak_compute_window_streaming_bounded_dp_max_two_macro_types_four_output_lanes_v5`；
- 新结果使用`experiments/results/tile_DSE/tile_window_dse_*`前缀。

现有`tile_only_dse_*`覆盖的是24个严格0.4608 TOPS、8-lane候选，仅保留作历史回归基线，
不得resume或解释成当前结果。当前138候选/4-lane正式结果尚未生成。推荐全枚举命令：

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/tile_DSE/run_sram_tile_dse.py \
  --all-layers --tile-peak-compute-tops 0.4608 \
  --peak-compute-tolerance-ratio 0.20 \
  --local-pareto-epsilon 0.001 --local-state-cap 4096
```

允许算力变化后，全局能量-延迟Pareto同时包含资源规模效应与Macro组成效应。汇报时必须
同时给出候选峰值TOPS，并建议按上述7个离散算力档位分别比较同构/异构候选；否则不能把
高算力候选的低延迟直接归因于异构组成。

正式长运行前还有一个待决口径：`_schedule_front()`把跨K-wave psum能量代理计入DP状态
能量，但`_evaluate_partition()`输出的`fabric_energy_j`只包含跨Macro、同wave的psum能量。
应选择“把跨wave能量纳入Tile目标”或“从Tile DP剪枝目标移除它”之一，使搜索目标和报告
目标一致；在此决定前未启动138候选全量运行。

随后已按用户要求删除全部10个`experiments/results/tile_DSE/tile_only_dse_*`旧结果文件。
更早的`tile_dse_*`历史结果未删除；当前`tile_window_dse_*`正式结果仍未生成。

### 2026-07-25 Tile DSE冻结模型闭环（阶段状态）

此前“跨wave口径待决”的状态已经关闭。当前实现以
`experiments/docs/tile_DSE/TILE_DSE_SEARCH_CONDITIONS.md` v2为准：

- 22 nm、400 MHz、4条输出lane、固定32个25-bit输出寄存器；
- 特征和权重共用512-bit/cycle总线，周期严格互斥，总线能量暂为0；
- 权重周期最多同时携带8个Macro的各一行64-bit权重；
- Macro动态能量使用每种物理规模独立的`E_fixed[q] + E_rows[q] * active_rows`；
- 5%静态功耗由所有物理Macro按完整层延迟支付；
- 同wave跨Macro和跨wave累加均在Tile内计能，延迟由VMM掩盖；
- 跨wave psum保存在Tile输出寄存器，不再产生Core cache/vector成本；
- 正式`latency_cap_ratio=4`，局部state cap=4096，aggregate cap=512；
- Bayesian预算拆为16个初始、24个采集、8个预测前沿验证，batch=4；
- 最终同时输出实际评价样本的E-T和E-T-A Pareto，并单独输出GP预测与不确定度。

模型版本更新为：

```text
energy:       partial_row_macro_static_tile_psum_register_weight_bus_v3
architecture: fixed_32_output_registers_shared_512b_bus_four_lanes_v6
latency:      ideal_dependency_shared_feature_weight_bus_v2
```

38个RUN层回归测试通过。代表层T16的4096/0.001/4x与8192/0/no-cap对照中，同构基线
最低能量和最低延迟分别变化0.088%和0.725%，异构基线均为0%。异构基线21层聚合在
512/1024 cap下最低能量与最低延迟一致；T15到T20在aggregate epsilon 0.001/0下结果
完全一致。正式48点`tile_window_dse_*`结果尚未启动或生成。

### 2026-07-26 Tile DSE正式搜索与可视化（当前最新状态）

已按冻结参数完成48/138个Tile的Bayesian评价：初始覆盖16、采集24、
预测前沿复评8，batch=4。运行中在38/48中断后通过`--resume`恢复，
最终检查点的模型版本和阶段数量均一致。

正式结果：

```text
aggregate feasible points = 10224
E-T Pareto points         = 1098
E-T-A Pareto points       = 2851
lowest energy             = 1xm128x8+1xm512x8, 0.262338 mJ, 13.956590 ms
lowest latency            = 17xm32x8+2xm64x8, 0.604695 mJ, 7.448548 ms
E-T knee                  = 1xm32x8+5xm128x8, 0.358644 mJ, 8.331530 ms
minimum Tile area         = 15xm32x8, 0.190778 mm2
```

结果CSV、二维E-T图、架构摘要图和E-T-A三目标图均位于
`experiments/results/tile_DSE/`，使用`tile_window_dse_*`前缀。可视化脚本为
`experiments/RUN/tile_DSE/generate_sram_tile_dse_pareto.py`。

小Macro数量多的双类型异构点存在明显长尾：原因是Macro子集、4-lane任务分配和
局部Pareto DP状态的组合增长，不是能量模型或进程死锁。后续性能优化应优先
做增量状态计数、资源子集缓存、lane状态规范化、层级checkpoint和层级并行；
不建议为提速直接降低正式`local_state_cap`或`latency_cap_ratio`。

39个RUN层回归测试通过，结果CSV的能量、延迟和面积字段全部为有限正值。

### 2026-07-26 Tile局部DP无损加速

已在`experiments/RUN/tile_DSE/sram_tile_dse.py`中实现增量状态计数、资源子集缓存、
lane时序key规范化，以及共享权重总线周期的分段闭式求和。新增
`profile_sram_tile_dse.py`输出分组数、转移数、剪枝数、峰值状态、cap次数和缓存命中。

在`5xm32x8+4xm128x8`/T16上，单层时间从270.375 s降为7.988 s，约33.8x；
Pareto行数仍为17，完整结果SHA-256一致。完整21层输出169行，与优化前正式
CSV的能量、延迟和调度签名哈希一致。回归测试增加到41个并全部通过。
该优化不改变模型语义或冻结版本，现有`tile_window_dse_*`正式结果仍然有效。

### 2026-07-26 Tile DSE核心研究目标

用户明确：当前 DSE 的长期目的是证明（更严格地说是用数据验证）
`1xm64x8+1xm512x8` 在面对多种网络负载时，是被搜索选出的 Pareto 最优组合之一。

后续工作必须围绕该假设组织，但不得通过候选空间、剪枝或目标权重偏向目标组合。
对每个网络应报告其 Pareto 资格、归一化 regret、面积和相对等算力`9xm64x8`的收益，
并与`1xm128x8+1xm512x8`等实际竞争者对照。“最优之一”的阈值和跨网络聚合规则必须在
多负载 DSE 前冻结。当前 ResNet-18 只是第一个负载，不足以单独证明跨网络结论。

判定阈值已冻结在`TILE_DSE_SEARCH_CONDITIONS.md`第1.2节：对每个网络分别计算
E-T和E-T-A乘法型Pareto regret，联合`r=max(r_ET,r_ETA)`。单网络通过线为5%；
跨网络通过率至少80%；中位regret不超3%；最坏regret不超10%；严格命中
E-T或E-T-A前沿的网络比例至少50%。各网络等权，阈值不得根据后续结果事后放宽。

用户随后明确最终统计口径：不穷举138个理论候选，只使用Bayesian优化选出的同一组
48个实测Tile，并在每个纳入统计的网络上对这48个Tile完成实际评价。对48个候选逐一
应用冻结阈值，最终只列出全算力窗口内所有达标组合，不单列严格0.4608 TOPS名单。
未实测候选的GP预测值不得进入Pareto、regret或最终名单。结论必须限定为“Bayesian
实测48候选中的稳健近优组合”，不得表述为138个候选的穷举全局最优。

当前ResNet-18的`r_ET=1.861618%`、`r_ETA=0%`，单网络通过；目标组合严格命中
E-T-A前沿，并在严格0.4608 TOPS子空间同时命中E-T和E-T-A前沿。

### 2026-07-26 ResNet-50固定48点评价

已扩展`run_sram_tile_bayesian_dse.py`，支持选择workload、按网络隔离结果目录和从CSV
固定实测Tile面板。ResNet-50复用ResNet-18 Bayesian搜索选出的同一组48个Tile，未重新
选择候选，也未使用GP预测值进入结果。正式配置保持batch size 1、4 lane、
`latency_cap_ratio=4`、`local_pareto_epsilon=0.001`和`local_state_cap=4096`。

ResNet-50 workload包含54个Conv/FC einsum、21种唯一`K/N/P`形状和
4,089,184,256 MAC。正式运行完成48/48个Tile，得到11,973个聚合状态、832个E-T
Pareto状态和3,450个E-T-A Pareto状态。审计确认48个设计与ResNet-18集合完全一致，
每个设计均覆盖54层，全部聚合状态的MAC和workload元数据一致。

`1xm64x8+1xm512x8`在ResNet-50上的结果为：

```text
r_ET                  = 2.048785892%
r_ETA                 = 0%
r                     = 2.048785892%  -> 单网络通过
exact E-T hit         = no
exact E-T-A hit       = yes
```

全局最低能量组合为`1xm128x8+1xm512x8`，最低能量0.636757098 mJ；全局最低延迟
组合为`17xm32x8+2xm64x8`，最低延迟18.19284 ms。结果与PNG/PDF可视化位于
`experiments/results/tile_DSE/resnet50/`。

### 2026-07-27 Tile调度复杂度修复与VGG16 v2结果

VGG16旧搜索在`K=25088,N=4096,P=1`层反复触发WSL内存耗尽。根因不是连接超时或
worker数量本身，而是局部DP生成不可完成状态，并使用无界`timing_cache`保存数百万个
lane/job时序键。

当前代码实现了：

- 固定131072条目的LRU timing cache；
- 输出lane占满但未完成全部输出块的状态剪枝；
- 剩余Macro资源无法容纳后续分组时的不可完成状态剪枝；
- 上述lane/资源死路的批量计数与剪枝；
- `schedule_search_version=reachable_state_bounded_timing_cache_v2`，并写入partial key、
  样本CSV和resume兼容检查。

代表层`5xm32x8+4xm128x8`由8.00 s降至2.52 s；病理缩小层
`11xm32x8+5xm64x8,K=25088,N=512,P=1`由27.82 s降至13.32 s。完整最坏形状
`N=4096`在批量资源剪枝前已能于5分40秒完成，峰值RSS为327476 KiB；最终批量剪枝
进一步减少Python内层循环。18个Tile DSE测试、`py_compile`和`git diff --check`通过。

使用同一组Bayesian选出的48个Tile，在新目录
`experiments/results/tile_DSE/vgg16_reachable_dp_v2/`完成VGG16干净复评：

```text
evaluated designs       = 48/48
aggregate states        = 10838
E-T Pareto states       = 786
E-T-A Pareto states     = 3148
service restarts        = 0
```

完整性审计确认48个设计唯一、每个设计覆盖16层、全部聚合能量/延迟/面积为有限正值，
且全部样本使用v2搜索版本。目标`1xm64x8+1xm512x8`的VGG16结果为：

```text
r_ET                  = 1.303777447%
r_ETA                 = 0%
r                     = 1.303777447%  -> 单网络通过
exact E-T hit         = no
exact E-T-A hit       = yes
```

最低能量为`1xm128x8+1xm512x8`的2.151620231 mJ；最低延迟为65.9791425 ms，
由`19xm32x8+1xm64x8`与`21xm32x8`并列达到。PNG/PDF可视化已经生成并完成视觉检查。

重要：v2移除了此前占用4096 state cap的不可完成状态，因此其有界搜索结果可能与旧版
不同。旧ResNet-18/50 regret不能和VGG16 v2直接做跨网络统计；下一步必须在同一48点
面板上用v2重跑ResNet-18和ResNet-50。

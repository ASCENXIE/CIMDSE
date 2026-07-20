# SRAM-CIM Tile DSE Handoff

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

- `experiments/results/tile_dse_bayesian_samples.csv`
  - 48个已评估 Tile 和模型版本。
- `experiments/results/tile_dse_layers.csv`
  - 每个 Tile、每层的局部调度状态和能量/延迟分解。
- `experiments/results/tile_dse_aggregate.csv`
  - 完整 ResNet-18 聚合状态。
- `experiments/results/tile_dse_aggregate_pareto.csv`
  - Energy/Latency/Area/Macro-count 前沿。
- `experiments/results/tile_dse_full_resnet_energy_latency_pareto.csv`
  - 二维能量-延迟前沿。
- `experiments/results/tile_dse_architecture_summary.csv`
  - 每个 Tile 的最低能量、最低延迟和 knee 摘要。
- `experiments/results/tile_dse_full_resnet_pareto.pdf`
  - 调度级矢量图。
- `experiments/results/tile_dse_architecture_summary.pdf`
  - 架构级矢量图。

PNG 版本也存在于同一目录。`experiments/results/` 被 Git 忽略。

## 15. 关键代码

- `experiments/SRAM_config/system_digital_sram_4x4.yaml`
  - Macro 表、寄存器结构、Lane数、类型数和权重装载速率。
- `experiments/RUN/sram_macro_types.py`
  - 加载 Macro characterization 和 Tile 接口参数。
- `experiments/RUN/sram_tile_dse.py`
  - Tile 候选枚举、寄存器面积/能量、权重装载时序、Lane受限调度 DP。
- `experiments/RUN/run_sram_tile_dse.py`
  - 全枚举 runner 和完整网络状态聚合。
- `experiments/RUN/run_sram_tile_bayesian_dse.py`
  - 贝叶斯外层搜索、并行执行和 checkpoint/resume。
- `experiments/RUN/generate_sram_tile_dse_pareto.py`
  - CSV、PNG、PDF 可视化。
- `experiments/TILE_DSE_SEARCH_CONDITIONS.md`
  - 当前搜索条件的详细说明。
- `experiments/README.md`
  - 简要运行说明。

## 16. 可复现命令

继续从48个样本补全到全部137个 Tile：

```bash
third_party/accelforge/.venv/bin/python -u \
  experiments/RUN/run_sram_tile_bayesian_dse.py \
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
  experiments/RUN/generate_sram_tile_dse_pareto.py
```

全枚举 runner 也可使用，但会重新计算全部候选：

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/run_sram_tile_dse.py \
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
?? experiments/RUN/generate_sram_tile_dse_pareto.py
?? experiments/RUN/run_sram_tile_bayesian_dse.py
?? experiments/RUN/run_sram_tile_dse.py
?? experiments/RUN/sram_tile_dse.py
?? experiments/TILE_DSE_SEARCH_CONDITIONS.md
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

# 当前 Tile 级设计空间探索条件与限制

> 更新时间：2026-07-20  
> 对应检查点：寄存器能量/面积、Lane 约束和权重装载延迟启用后，48 / 137 个 Tile 组成已评估

## 1. 当前探索目标

当前阶段只比较 **SRAM-CIM Macro 如何组成一个 Tile**，目标是在完整
ResNet-18 工作负载上寻找能量、延迟、面积和 Macro 数量之间的 Pareto 前沿，
并研究同构 Tile 与异构 Tile 的系统级能效差异。

当前问题的核心是：

- 小 Macro 的规约粒度更细，通常具有更高的行利用率和调度灵活性；
- 大 Macro 的单次 VMM 能效更高，但在小规约层中可能出现较低的行利用率；
- 异构 Tile 希望在大 Macro 的高能效和小 Macro 的细粒度之间取得平衡。

## 2. 当前评估边界

当前 `sram_tile_dse.py` 实际评估的是 **一个 Tile**：

```text
一个 Tile 组成
  -> 对每个 ResNet-18 层枚举 Macro 分组和输出通道块分配
  -> 对 21 层的局部 Pareto 调度进行组合
  -> 得到完整网络的能量-延迟状态
```

虽然 `system_digital_sram_4x4.yaml` 中仍描述了 4 个 Tile 组成的系统层次，
但当前 Tile DSE 没有将结果按 4 个 Tile 并行缩放，也没有建模 Tile 间的层分配、
流水和通信。因此，目前的结果不能直接解释为 4-Tile Core 或完整系统结果。

## 3. 工作负载

| 项目 | 当前设置 |
|---|---:|
| 网络 | AccelForge ResNet-18 |
| 批量 | Batch = 1 |
| 参与评估的层 | 21 个 Conv/FC einsum，T0-T20 |
| 网络总 MAC 数 | 1,814,073,344 |
| 执行方式 | 各层顺序执行，层延迟与能量逐层相加 |

当前未包含 ReLU、BN、池化、残差加法和完整张量依赖关系。

## 4. Macro 候选库

当前只探索列数为 8 的 Macro，输入和权重均为 8 bit。

| Macro | 规约行数 | 输出列数 | 能效 (TOPS/W) | 面积效率 (TOPS/mm2) | 单次完整 VMM 能量 | Macro 面积 (mm2) |
|---|---:|---:|---:|---:|---:|---:|
| `m32x8` | 32 | 8 | 6.08 | 2.23 | 84.211 pJ | 0.020101 |
| `m64x8` | 64 | 8 | 7.66 | 2.20 | 133.681 pJ | 0.040749 |
| `m128x8` | 128 | 8 | 11.28 | 2.31 | 181.560 pJ | 0.077618 |
| `m256x8` | 256 | 8 | 13.13 | 2.28 | 311.957 pJ | 0.157278 |
| `m512x8` | 512 | 8 | 17.03 | 2.24 | 481.033 pJ | 0.320173 |

共同物理参数：

- 工艺：22 nm；
- 电压：0.72 V；
- Macro 周期：1.4278 ns；
- 串行输入位宽：1 bit/cycle；
- 一次 8-bit 完整 VMM 需要 8 个周期，即 11.4224 ns。

Macro 数据来自 `experiments/docs/SRAM_DATA/cim_macro_efficiency.csv`。面积由
Macro 峰值吞吐率除以面积效率得到，而不是直接采用 YAML 中旧系统基线的固定
`macro_area`。

## 5. Tile 候选的硬约束

参考 Tile 为：

```text
9 x m64x8
容量 = 9 x 64 x 8 = 4608 个逻辑权重
Macro面积 = 0.366743496 mm2
寄存器面积 = 0.024137525 mm2
Tile总面积 = 0.390881021 mm2
```

候选 Tile 不要求容量和面积绝对相同，只要求二者都落在参考值的 +/-20% 范围内。

| 约束 | 下限 | 上限 |
|---|---:|---:|
| 逻辑权重容量 | 3686.4 | 5529.6 |
| Tile 总面积 | 0.312704817 mm2 | 0.469057225 mm2 |

每个 Tile 最多包含两种不同的 Macro 类型，但同一类型可以出现多个实例。例如
`3xm64x8+1xm256x8` 满足类型数量规则，但仍必须同时通过下面的容量和面积边界。

枚举结果：

- 候选总数：137；
- 同构候选：14；
- 双类型异构候选：123；
- 单 Tile Macro 数量范围：1-21；
- 当前没有直接限制 `m32` 数量；
- 当前没有限制独立 Macro 端点数、调度器发射宽度或局部互连端口数。

容量与面积约束必须同时满足。Macro 数量目前不是候选过滤的硬约束。
`3xm64x8+1xm256x8` 的容量和总面积低于 `9xm64x8` 参考值约22.2%和24.0%，
所以在当前 +/-20% 比较边界下不会进入候选池；这不是类型数量规则造成的排除。

## 6. Tile 内映射模型

### 6.1 基本计算单位

每个 Macro 一次产生 8 个输出通道，因此一层被划分为：

```text
N_blocks = ceil(N / 8)
```

对于每个输出通道块，调度器枚举一个或多个 Macro 组成的 lane：

- 一个 Macro 可以独立完成 K 规约；
- 多个 Macro 可以共同完成同一个 K 规约，结果由 Tile 内向量单元累加；
- 不同 lane 可以并行计算不同的输出通道块；
- 同一种 lane 可以分配多个输出通道块，负载在其副本之间尽量均衡；
- 不强制所有 Macro 被使用，允许空闲 Macro；
- 同一组 Macro 在不同 VMM wave 中可以继续完成超过组总行数的 K 规约。

Tile 固定最多提供4条输出 Lane，每层激活1-4条。每条 Lane 对应8个输出通道：

- 分配到同一 Lane 的 Macro 结果相加，用于共同完成 K 规约；
- 分配到不同 Lane 的 Macro 结果不相加，用于计算不同输出通道块；
- 调度器可以针对不同层重新选择 Macro 分组；
- DP 把活动 Lane 数作为受限资源，所有保留调度满足 `1 <= active_lane_count <= 4`。

当前使用广义 Macro 分组与 N-block 分配的动态规划，不再把
`K-parallel`、`N-parallel` 和 `mixed` 当作三套互斥的搜索策略。

### 6.2 映射标签的含义

`K/N/mixed` 是对已经生成的调度进行的事后分类：

- `k_parallel`：调度中只有一个 lane；
- `n_parallel`：有多个 lane，且每个 lane 只占用一个 Macro；
- `mixed`：有多个 lane，且至少一个 lane 由多个 Macro 组成。

因此这些标签只用于结果解释和着色，不能理解为三种受到同等资源约束的独立数据流。

### 6.3 权重驻留

模型采用权重驻留思想：同一组已装载权重在全部输出位置之间复用。

- `weight_reuse_per_load = output_positions`；
- `weight_fits_tile` 仅判断一层全部权重是否小于 Tile 容量；
- `weight_residency_epochs = ceil(layer_weight_values / tile_capacity_values)`；
- 权重值数量、装载事件和驻留轮次会写入 CSV；
- 权重装载延迟计入目标函数，权重装载能量当前为0。

每个 Macro 具有独立装载端口，每周期装载一行、每行包含该 Macro 的8个权重。
同一 Macro 的装载与计算不重叠，不同 Macro 和 Lane 可以并行装载；只装载实际参与
K 规约的有效行。每组权重在全部输出位置之间驻留复用，因此装载次数不随
`output_positions` 重复。

当前模型也没有实现多个 Tile 之间“权重尽量不复制，只有整层权重能装入单 Tile 时
才复制”的分配规则，这属于下一阶段 Core 级调度。

## 7. 能量模型

当前全网能量为：

```text
E_fabric = E_macro + E_input_register + E_output_register + E_psum
```

其中：

```text
E_macro = sum(各类 Macro 调用次数 x 该 Macro 的完整 VMM 能量)

E_input_register = sum(
    Macro调用次数 x Macro行数
    x (8-bit寄存器读取能量 + 8-bit寄存器写入能量)
)

E_output_register = sum(
    Macro调用次数 x 8个输出通道
    x (25-bit寄存器读取能量 + 25-bit寄存器写入能量)
)

psum_width = input_bits + weight_bits + ceil(log2(K))
E_psum = psum_vector_count x psum_vector_energy
```

每个 8 列输出向量在合并 `n` 个 Macro 部分和时，记为 `n - 1` 次向量累加。
向量累加能量由 22 nm 的 `DigitalSramShiftAccumulator` 模型估计。

寄存器使用 AccelForge `AladdinRegister` 的 22 nm 缩放结果。每次 Macro VMM
调用对一个活动输入 Bank 计一次完整物理深度的读和写，包括未使用行；另对单组
`8 x 25-bit` 输出寄存器计一次读和写。第二个输入 Bank 只代表 ping-pong 物理容量，
不会让单次动态访问能量翻倍。当前寄存器模型不计 leakage，也不增加寄存器访问延迟。
输入寄存器按两个物理 Bank 计面积，输出寄存器按单 Bank 计面积。

当前明确没有计入：

- ping-pong 控制和 Bank 固定开销；
- 输入广播、输出归约和多 Macro 同步能量；
- 调度器发射、状态机和同步能量；
- 每个 Macro 独立控制端点的固定能量；
- 权重装载及重编程能量；
- Macro 泄漏、时钟树和电源管理能量；
- Tile 内输入广播和局部互连能量；
- line FIFO、GLB、NoC 和 DRAM 能量；
- 激活函数、池化和残差加法能量。

## 8. 延迟模型

每条 Lane 的一个输出块延迟为：

```text
T_job = output_positions x T_VMM_compute + T_weight_load
```

同一 Lane 的多个输出块顺序执行，不同 Lane 并行执行；层延迟为所有 Lane 总时间的
makespan，完整网络延迟为21层之和。每个 Macro 每周期装载一个有效权重行，装载能量为0。

当前没有计入：

- 部分和向量累加延迟；
- 调度器 issue、barrier 和同步延迟；
- 输入、输出和中间数据搬运延迟；
- 多 Tile 通信与流水延迟。

为避免搜索大量明显过慢的分配，每层调度设置了相对延迟上限：

```text
latency_cap = 2.0 x 该 Tile 在该层上的简单可行调度延迟
```

该上限是相对每个 Tile 自身的，不是统一的绝对延迟约束。

## 9. Pareto 剪枝与近似

- 层内局部 Pareto epsilon：0.001，即 0.1%；
- 全网逐层组合状态上限：每个 Tile 512 个状态；
- 超过上限时自适应提高能量-延迟压缩 epsilon；
- 当前检查点实际最大聚合 epsilon：0.00225，即 0.225%；
- 最终资源 Pareto 目标：能量、延迟、Tile 总面积、Macro 数量。

最终 Pareto 中 Macro 数量是一个独立资源维度，但它不会转换为能耗或延迟。

## 10. 贝叶斯搜索设置

当前搜索命令对应的配置为：

```bash
third_party/accelforge/.venv/bin/python \
  experiments/RUN/run_sram_tile_bayesian_dse.py \
  --design-budget 48 \
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

外层贝叶斯搜索使用的特征包括：

- 五种 Macro 的数量；
- 相对容量；
- 相对面积；
- Macro 总数量。

初始样本强制包含 `9xm64x8`、`1xm64x8+1xm512x8` 和所有同构候选，
剩余初始点使用最远点采样补齐。后续采集基于三个设计摘要：

- 最小完整网络能量；
- 最小完整网络延迟；
- 最小完整网络 EDP。

Gaussian Process 同时考虑预测值、不确定性和批内样本距离。注意：面积和 Macro 数量
虽然作为 GP 输入特征，但目前不是采集函数的预测目标。

## 11. 当前搜索进度与结果文件

| 项目 | 当前值 |
|---|---:|
| 已评估 Tile | 48 / 137 |
| 同构/异构样本 | 14 / 34 |
| 全网聚合调度状态 | 10,151 |
| 四资源 Pareto 状态 | 2,381 |
| 能量-延迟 Pareto 状态 | 728 |
| 能量模型 | Macro + 输入/输出寄存器 + psum |
| 检查点版本 | `macro_psum_aladdin_register_v1` |
| 架构约束版本 | `max_two_macro_types_four_output_lanes_v1` |
| 面积模型 | Macro + Aladdin输入/输出寄存器 |
| 延迟模型 | VMM + 每Macro每周期一行权重装载 |

主要输出：

- `results/tile_dse_layers.csv`：逐层调度结果；
- `results/tile_dse_aggregate.csv`：完整网络组合状态；
- `results/tile_dse_aggregate_pareto.csv`：能量/延迟/面积/Macro 数四目标前沿；
- `results/tile_dse_architecture_summary.csv`：每个 Tile 的最小能量与最小延迟摘要；
- `results/tile_dse_full_resnet_pareto.pdf`：全网调度级矢量图；
- `results/tile_dse_architecture_summary.pdf`：Tile 架构级矢量图。

## 12. 为什么当前结果偏好大量 m32

当前模型已经完整计入 `m32` 带来的收益：

- 更细的 K 划分；
- 更容易填补剩余规约；
- 更多可并行分配的 lane；
- 更高的行利用率和负载均衡灵活性。

但没有计入其主要代价：

- 更多独立 Macro 需要更多调度端点；
- 更多发射、同步和控制事件；
- 更复杂的局部互连；
- 更多时钟和寄存器固定集成面积；
- 潜在的更高权重装载与重配置开销。

当前已经计入寄存器动态读写能量与面积，以及权重装载延迟；仍未计入寄存器 leakage、
时钟树、控制端点和 Lane 归约成本，因此对大量 `m32` 的惩罚仍不完整。当前按用户约定
不设置 `max_lane_fan_in`。

`tile_macro_count` 只作为最终 Pareto 的一个坐标，不能降低某个设计的 TOPS/W，也不会
增加其延迟；贝叶斯采集仍只优化能量、延迟和 EDP。因此当前高频出现 `m32` 是模型边界
导致的系统性偏好，不能直接作为“小 Macro Tile 一定更优”的架构结论。

## 13. 下一步建议

在继续扩大样本数之前，优先加入参数化的 Tile 调度开销模型：

```text
E_scheduler = N_round x E_round
            + N_activation x E_endpoint
            + N_weight_load x E_load_setup

T_scheduler = N_round x T_issue
            + N_barrier x T_sync

A_tile = sum(A_macro)
       + N_endpoint x A_endpoint
       + A_local_interconnect(N_endpoint)
```

建议先做灵敏度扫描，而不是直接限制 `m32` 数量：

- Macro 激活固定能量：相对一次 `m64` VMM 的 0%、1%、2%、5%；
- 端点固定面积：相对一个 `m64` 面积的 0%、1%、2%、5%；
- 调度发射延迟：0、0.25、0.5、1 个 Macro 周期；
- 最大独立端点数：8、12、16。

这样可以观察异构 Tile 的 Pareto 优势是否在合理调度成本区间内仍然成立，避免通过
人为禁止 `m32` 预先决定搜索结论。

## 14. 相关代码

- `SRAM_config/system_digital_sram_4x4.yaml`：Macro 类型表和物理参数；
- `RUN/sram_macro_types.py`：读取 Macro 表并计算 VMM 能量与面积；
- `RUN/sram_tile_dse.py`：Tile 候选枚举、Macro 分组和层内调度；
- `RUN/run_sram_tile_bayesian_dse.py`：外层贝叶斯搜索与检查点；
- `RUN/generate_sram_tile_dse_pareto.py`：Pareto CSV、PNG 和 PDF 可视化。

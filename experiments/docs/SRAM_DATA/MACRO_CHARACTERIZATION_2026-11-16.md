# 当前Macro参数与能量口径

原始数据来自 2026 年 11 月 16 日的模拟器表征，面积与完整 INT8 VMM 能量为计算依据。
现行模型使用 22 nm、0.72 V、W8A8、8 输出列和 250 MHz，周期为 4 ns，完整 VMM 为 8 周期（32 ns）。表中的面效按 250 MHz 下的峰值算力和原始面积计算。
模型版本：`partial_row_macro_static_register_gb22_33p6_psum32_macro_refresh_v14_bus4_250mhz`。

| Macro | 面积 µm² | 完整INT8 VMM总能量 pJ | TOPS/W | 250 MHz TOPS/mm² |
|---|---:|---:|---:|---:|
| 32×8 | 29828 | 83.83 | 6.10759871 | 0.53640874 |
| 64×8 | 54106 | 139.08 | 7.36266897 | 0.59143163 |
| 128×8 | 92078 | 187.01 | 10.95128603 | 0.69506288 |
| 256×8 | 160224 | 325.25 | 12.59338970 | 0.79888157 |
| 512×8 | 297630 | 475.18 | 17.23978282 | 0.86012835 |

```text
ops = 2*rows*8; peak_TOPS = ops/(32e-9)/1e12
TOPS/W = ops/energy_pJ; TOPS/mm² = peak_TOPS/(area_um²/1e6)
E_dynamic_full = .95*E_full; P_static = .05*E_full/(32e-9)
E_dynamic(0)=0; E_dynamic(r>0)=E_fixed + r*E_row
```

能量公式使用J，表中pJ须乘1e-12。静态按全部已安装Macro功率乘实际总执行时间计入，
不按VMM重复收取。外部输入/输出寄存器、权重写入、总线和GB按[规范](../tile_DSE/TILE_DSE_EXPERIMENT_SPECIFICATION_V5.md)另计。

| Macro | 固定动态项 pJ | 每活动行动态项 pJ/行 |
|---|---:|---:|
| 32×8 | 50.73939625305969 | 0.9030969920918848 |
| 64×8 | 96.14639789310044 | 0.5621812829203057 |
| 128×8 | 112.74069282572697 | 0.5071781810490081 |
| 256×8 | 208.50056513718505 | 0.39252708930787095 |
| 512×8 | 294.57601810755614 | 0.30633785525867935 |

部分行能量沿用 512 行 Macro 的归一化形状 `P(r)=4.793+0.004984375*r`，以满行动态能量 451.421 pJ 定标。其他尺寸先取满行动态能量随物理行数的家族中心差分
（端点单侧差分），再乘512归一化斜率/家族斜率比0.5505902118969338，最后`E_fixed=E_dynamic_full-E_row*rows`。
部分行成本来自外推。当前表征提供满行端点。

唯一计算来源是[cim_macro_efficiency.csv](cim_macro_efficiency.csv)中的PC=WB=IB=8五行，
以及[硬件YAML](../../SRAM_config/system_digital_sram_4x4.yaml)的时序和部分行系数。
`sram_helper_functions`查CSV，`sram_macro_types.load_sram_macro_library`恢复Macro能量/面积并检查满行端点。
AccelForge YAML端点和Macro图表生成器共用同一CSV，不维护第二套PPA。

GB每512-bit读22 pJ、写33.6 pJ，不做bankscale。
总线按 4 级 512-bit 串联寄存器估算，与 Macro 输入寄存器分别计费。
沿用`hwcomponents_library==1.0.58`的`AladdinRegister` 22-nm读写代理：
每级每512 bit读/写各1.5595915900695365 pJ，穿越4级共12.476732720556292 pJ，
等效 0.024368618594836508 pJ/有效 bit。加载 YAML 时核对级数、位宽和读写模型。
该参数描述有效位的寄存器访问能量。时钟树和布线能耗属于模型边界。服务时序按各通路的带宽计算，主 EDAP 使用 Tile 本地面积。
跨K波次和跨KV输出状态32 bit，最终输出8 bit，局部结果25 bit。
Macro 本体面积为 `64+512` 的 0.351736 mm² 和 `4×128` 的 0.368312 mm²，前者低 4.5005%。整网 EDAP 使用包含本地寄存器的面积，以及同一条调度的能量和延迟。 [四网实验结论](../tile_DSE/EXPERIMENT_CONCLUSION.md)

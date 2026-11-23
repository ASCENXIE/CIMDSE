# v5 CNN 输入几何

输入约定：batch=1，预处理后的图像为 224×224×3（2026-11-07 用户确认）。
这不是对 ImageNet 所有原始图像分辨率的声明。

- `resnet18_224.json`：21 个 Conv/FC，1,814,073,344 MAC。
- `resnet50_224.json`：54 个 Conv/FC，4,089,184,256 MAC；bottleneck 的下采样 stride 位于 3×3 卷积。

几何由 `experiments/RUN/tile_DSE/v5/baseline_cnn.py` 显式建立，再与本地 AccelForge
原始负载的 kernel、stride、通道、P/K/N、层数及 MAC 总数逐项比较，不用输出 P/K/N 猜 padding。
所用标准 ResNet 规则与 Torchvision 官方 ResNet 源码文档核对：stem 7×7/s2/p3、
maxpool 3×3/s2/p1、3×3 卷积 p1、1×1 卷积 p0、最终全局平均池化。
参考地址已记录在每份 JSON 的 `architecture_reference` 中。

`shape_steps` 保存池化、残差加法和卷积的形状传播；`input_source` 记录真实块入口及分支来源。
`dependencies` 则约束本轮采用的 Conv/FC 顺序执行，不应误解为逐层真实张量数据流。
非 CIM 算子的形状影响保留，时间与能量仍按 v5 理想化。

重新生成时使用空目录并审查差异：

```bash
conda activate cim-dse
python experiments/RUN/tile_DSE/v5/baseline_cnn.py --output-dir /tmp/resnet_geometry_review
```

原始工作负载哈希发生变化时，准备入口会拒绝直接沿用旧几何文件。
这些文件关闭了 CNN 输入几何缺失问题，不代表已经完成全网络映射优化或性能评价。
ViT 仍使用独立的 192-token 工作负载定义，本目录的 224 输入约定仅用于两个 CNN 基线。

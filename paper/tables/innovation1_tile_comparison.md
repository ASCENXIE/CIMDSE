# Homogeneous and Heterogeneous Tile Comparison

Each row uses one internally consistent minimum-EDP mapping. The Pareto flag
is computed in the energy versus area-delay-product plane.

| Workload | Role | Design | Class | Energy (mJ) | Latency (ms) | Area (mm2) | EDP (mJ ms) | ADP (mm2 ms) | E-ADP Pareto |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-18 | Best-EDP homogeneous | `5xm128x8` | homogeneous | 0.3603 | 9.0599 | 0.2458 | 3.2645 | 2.2272 | 0 |
| ResNet-18 | Best-EDP heterogeneous | `1xm32x8+5xm128x8` | heterogeneous | 0.3675 | 8.2376 | 0.2584 | 3.0276 | 2.1289 | 0 |
| ResNet-18 | Proposed 64+512 | `1xm64x8+1xm512x8` | heterogeneous | 0.2885 | 13.1983 | 0.2281 | 3.8084 | 3.0102 | 1 |
| ResNet-18 | Equal-TOPS 9x64 | `9xm64x8` | homogeneous | 0.5042 | 8.9983 | 0.2314 | 4.5365 | 2.0822 | 0 |
| ResNet-18 | Single 512 | `1xm512x8` | homogeneous | 0.3067 | 17.1940 | 0.2026 | 5.2740 | 3.4828 | 0 |
| ViT-B/16 | Best-EDP homogeneous | `5xm128x8` | homogeneous | 3.3756 | 89.0993 | 0.2458 | 300.7647 | 21.9037 | 0 |
| ViT-B/16 | Best-EDP heterogeneous | `1xm32x8+5xm128x8` | heterogeneous | 3.4184 | 71.7862 | 0.2584 | 245.3951 | 18.5522 | 1 |
| ViT-B/16 | Proposed 64+512 | `1xm64x8+1xm512x8` | heterogeneous | 2.6884 | 127.8181 | 0.2281 | 343.6296 | 29.1523 | 1 |
| ViT-B/16 | Equal-TOPS 9x64 | `9xm64x8` | homogeneous | 4.8319 | 85.7797 | 0.2314 | 414.4771 | 19.8495 | 0 |
| ViT-B/16 | Single 512 | `1xm512x8` | homogeneous | 2.7722 | 146.8206 | 0.2026 | 407.0226 | 29.7400 | 0 |

# interpolation：PyTorch 移植与 CPU/NPU 评测

## 原始实现位置（Pointcept df36980）

| 组件 | 路径 |
|------|------|
| Python API | `libs/pointops/functions/interpolation.py` |
| CUDA kernel | `libs/pointops/src/interpolation/interpolation_cuda_kernel.cu` |
| pointops2 同名 | `libs/pointops2/functions/pointops.py` |

**接口**

```python
new_feat = pointops.interpolation(xyz, new_xyz, feat, offset, new_offset, k=3)
# xyz: (m, 3), new_xyz: (n, 3), feat: (m, c)
# offset / new_offset: (B,) 累积上界
# 返回: (n, c)
```

**算法**

1. `idx, dist = knn_query(k, xyz, offset, new_xyz, new_offset)` → `(n, k)`。
2. 反距离权重：`weight[n,i] = (1/(dist+1e-8)) / sum_j(1/(dist+1e-8))`。
3. 输出：`output[n,c] = sum_i feat[idx[n,i], c] * weight[n,i]`（与 CUDA `interpolation_forward_cuda` 一致）。

`interpolation2` 为 `torch.autograd.Function`，forward 走 CUDA gather，backward 用 `atomicAdd` 累加梯度。

## PyTorch 实现

见 `interpolation.py`：复用 `knn_query` + 向量化 gather/加权求和，支持 **CPU / CUDA / NPU**。

```python
from pointops_torch import interpolation

out = interpolation(xyz, new_xyz, feat, offset, new_offset, k=3)
```

核心逻辑：

```python
idx, dist = knn_query(k, xyz, offset, new_xyz, new_offset)
weight = (1 / (dist + 1e-8)) / (1 / (dist + 1e-8)).sum(dim=1, keepdim=True)
return (feat[idx.long()] * weight.unsqueeze(-1)).sum(dim=1)
```

## 精度对比（Ascend 容器，warmup=5 repeat=30）

| Case | Shape | vs fp64 参考 | max abs err | CPU vs NPU | max abs err |
|------|-------|--------------|-------------|------------|-------------|
| small | N=4096 M=512 C=32 k=3 | PASS | 2.98e-07 | PASS | 3.58e-07 |
| medium | N=16384 M=2048 C=64 k=3 | PASS | 4.77e-07 | PASS | 4.77e-07 |
| large | N=65536 M=8192 C=128 k=3 | — | — | PASS | 4.77e-07 |

说明：large 规模下 fp64 堆式 `knn_query` 参考过慢（O(M·N) Python 循环），仅做 CPU/NPU 一致性对比；small/medium 与 fp64（fp64 knn + fp64 加权 gather）在 `rtol=1e-4, atol=1e-6` 下 PASS。

## 性能对比（steady-state，ms/call）

| Case | CPU (ms) | NPU (ms) | CPU/NPU |
|------|----------|----------|---------|
| small | 31.88 | 1.13 | **28.2×** |
| medium | 269.93 | 3.06 | **88.3×** |
| large | 2499.75 | 17.26 | **144.9×** |

端到端包含 `knn_query` + 加权插值；NPU 上矩阵化 `topk` 与 gather 收益明显。

## 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/d00883276/cuda_to_npu/Pointcept/libs/pointops_torch
python3 bench_interpolation.py --device-id 0
```

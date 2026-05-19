# knn_query：PyTorch 移植与 CPU/NPU 评测

## 原始实现位置

| 组件 | 路径 |
|------|------|
| Python API | `libs/pointops/functions/query.py` |
| CUDA kernel | `libs/pointops/src/knn_query/knn_query_cuda_kernel.cu` |

**接口**

```python
idx, dist = pointops.knn_query(nsample, xyz, offset, new_xyz=None, new_offset=None)
# xyz: (N, 3), new_xyz: (M, 3)  默认 new_xyz=xyz, new_offset=offset
# offset / new_offset: (B,) 累积上界
# idx: (M, nsample), dist: (M, nsample) 欧氏距离 = sqrt(dist2)
```

**算法**

1. 对每个 query 点 `pt_idx`，用 `new_offset` 确定 batch `bt_idx`。
2. 仅在 `xyz[offset[bt-1]:offset[bt]]` 内搜索（与 query 同 batch 的源点）。
3. 维护大小为 `nsample` 的最大堆，保留 k 个最小平方距离；最后 `heap_sort` 升序输出。
4. Python 层对 `dist2` 开方返回 `dist`。（CUDA 数组上限 `nsample <= 128`。）

## PyTorch 实现

见 `knn_query.py`：按 batch 用 `(M_b, N_b)` 距离矩阵 + `torch.topk(..., largest=False, sorted=True)`，语义与堆选 k 近邻一致（同距离时 tie-break 可能与 CUDA 不同，本评测数据下与 fp64 堆参考 100% 一致）。

```python
from pointops_torch import knn_query

idx, dist = knn_query(16, xyz, offset, new_xyz, new_offset)
```

## 精度对比（Ascend 容器）

| Case | N | M | k | idx vs fp64 堆 | dist vs fp64 | CPU vs NPU idx | CPU vs NPU dist |
|------|---|---|---|----------------|--------------|----------------|-----------------|
| small | 4096 | 512 | 16 | PASS 100% | PASS | PASS 100% | PASS |
| medium | 16384 | 2048 | 16 | PASS 100% | PASS | PASS 100% | PASS |
| large | 65536 | 8192 | 16 | PASS 100% | PASS | PASS 100% | PASS |

## 性能对比（warmup=5, repeat=30）

| Case | CPU (ms) | NPU (ms) | CPU/NPU |
|------|----------|----------|---------|
| small | 34.95 | 1.05 | **33.4×** |
| medium | 279.66 | 2.83 | **98.7×** |
| large | 2077.83 | 16.70 | **124.4×** |

按 batch 的矩阵化 `topk` 在 NPU 上收益显著（相对 FPS 的 Python 循环实现）。

## 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/d00883276/cuda_to_npu/Pointcept/libs/pointops_torch
python3 bench_knn_query.py --device-id 0
```

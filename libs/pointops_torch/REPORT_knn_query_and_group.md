# knn_query_and_group：PyTorch 移植与 CPU/NPU 评测

## 原始实现位置

| 组件 | 路径 |
|------|------|
| Python API | `libs/pointops/functions/utils.py` |
| KNN query | `libs/pointops/functions/query.py` + `knn_query_cuda_kernel.cu` |
| Grouping | `libs/pointops/functions/grouping.py` + `grouping_cuda_kernel.cu` |
| 调用示例 | `pointcept/models/point_transformer/point_transformer_seg.py` L48–59 |

**接口**

```python
grouped, idx = pointops.knn_query_and_group(
    feat, xyz, offset, new_xyz, new_offset,
    idx=None, nsample=16, with_xyz=False,
)
# feat: (N, C), xyz: (N, 3)
# offset / new_offset: (B,) 累积上界
# idx: (M, nsample)  可选，预计算时跳过 knn_query
# grouped: (M, nsample, C) 或 with_xyz=True 时为 (M, nsample, 3+C)
```

**算法**

1. 若 `idx is None`：调用 `knn_query(nsample, xyz, offset, new_xyz, new_offset)` 得到 `(M, k)` 近邻索引。
2. 调用 `grouping(idx, feat, xyz, new_xyz, with_xyz)`：
   - 按 `idx` gather 特征 `(M, k, C)`；
   - `with_xyz=True` 时 gather 坐标、减去 `new_xyz`、对 `idx=-1` 置零，再与特征 concat。

## PyTorch 实现

见 `knn_query_and_group.py`（组合 `knn_query.py` + `grouping.py`），**同一套代码**可在 CPU / CUDA / NPU 上运行：

```python
from pointops_torch import knn_query_and_group

grouped, idx = knn_query_and_group(
    feat, xyz, offset, new_xyz, new_offset,
    nsample=16, with_xyz=True,
)
# 复用 idx、仅 grouping（Point Transformer 中 x_v 分支）:
grouped_v, _ = knn_query_and_group(
    feat_v, xyz, offset, new_xyz, new_offset,
    idx=idx, nsample=16, with_xyz=False,
)
```

## 精度对比（Ascend 容器 `cuda_migration_test`）

| Case | N | M | C | k | idx vs fp64 | out vs fp64 | CPU vs NPU idx | CPU vs NPU out |
|------|---|---|---|---|-------------|-------------|----------------|----------------|
| small | 4096 | 512 | 32 | 16 | PASS 100% | PASS | PASS 100% | PASS |
| medium | 16384 | 2048 | 64 | 16 | PASS 100% | PASS | PASS 100% | PASS |
| large | 65536 | 8192 | 128 | 16 | PASS 100% | PASS | PASS 100% | PASS |

`with_xyz=True` 全路径评测；CPU/NPU fp32 输出与 fp64 参考完全一致（max |out_diff| = 0）。

## 性能对比（warmup=5, repeat=30）

| Case | CPU (ms) | NPU (ms) | CPU/NPU |
|------|----------|----------|---------|
| small | 39.55 | 4.70 | **8.4×** |
| medium | 296.88 | 40.88 | **7.3×** |
| large | 2242.20 | 349.33 | **6.4×** |

KNN 矩阵化 `topk` + gather 在 NPU 上相对 CPU 有稳定加速；大规格时 grouping 的索引 gather 占比上升，加速比略降但仍显著。

## 复现

```bash
docker exec -it cuda_migration_test bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/d00883276/cuda_to_npu/Pointcept/libs/pointops_torch
python3 bench_knn_query_and_group.py --device-id 0
```

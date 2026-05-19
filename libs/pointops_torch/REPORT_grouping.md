# grouping：PyTorch 移植与 CPU/NPU 评测

## 原始实现位置

| 组件 | 路径 |
|------|------|
| Python API（本任务对齐对象） | `libs/pointops/functions/grouping.py` |
| 底层 CUDA gather | `libs/pointops/src/grouping/grouping_cuda_kernel.cu` |

**接口**（[Pointcept df36980](https://github.com/Pointcept/Pointcept/tree/df36980119f4636beb2d02d04ef3b2fec0fddfba/libs)）

```python
out = pointops.grouping(idx, feat, xyz, new_xyz=None, with_xyz=False)
# idx: (m, nsample), feat: (n, c), xyz: (n, 3), new_xyz: (m, 3)
# out: (m, nsample, c) 或 with_xyz 时 (m, nsample, 3+c)
```

**语义**

1. 在 `xyz` / `feat` 末尾各拼接一行零向量（`idx == -1` 时取该行）。
2. `grouped_feat = feat[idx].view(m, nsample, c)`。
3. `with_xyz=True`：`grouped_xyz = (xyz[idx] - new_xyz.unsqueeze(1)) * sign(idx+1)`，再与 `grouped_feat` 在最后一维拼接。

公开 API 在原始仓库中已是 **纯 PyTorch 索引**；底层 `grouping2 = Grouping.apply` 才是 CUDA kernel 的 `(n,c)` gather。Point Transformer 等模型调用的是上面的 `grouping()`。

## PyTorch 实现

见 `grouping.py`，与官方 Python 逻辑一致，可在 CPU / CUDA / NPU 上运行：

```python
from pointops_torch import grouping

out = grouping(idx, feat, xyz, new_xyz, with_xyz=True)
```

## 精度对比（Ascend 容器 `cuda_migration_test`）

idx 由同分布 `knn_query` 生成（含 `-1` padding），与 fp64 逐元素参考对比。

### feat only（`with_xyz=False`）

| Case | m | k | c | vs fp64 ref | CPU vs NPU | max \|diff\| |
|------|---|---|---|-------------|------------|-------------|
| small | 512 | 16 | 32 | PASS | PASS | 0 |
| medium | 2048 | 16 | 64 | PASS | PASS | 0 |
| large | 8192 | 16 | 128 | PASS | PASS | 0 |

### with_xyz=True

| Case | m | k | c | vs fp64 ref | CPU vs NPU | max \|diff\| |
|------|---|---|---|-------------|------------|-------------|
| small | 512 | 16 | 32 | PASS | PASS | 0 |
| medium | 2048 | 16 | 64 | PASS | PASS | 0 |
| large | 8192 | 16 | 128 | PASS | PASS | 0 |

## 性能对比（warmup=5, repeat=30）

评测环境无 NVIDIA GPU；CUDA 列在 Ascend 容器中为 N/A。GPU 路径与 NPU 相同（`tensor[idx]` advanced indexing），在有 CUDA 的机器上可将 `device=cuda` 复现。

### feat only

| Case | CPU (ms) | NPU (ms) | CPU/NPU |
|------|----------|----------|---------|
| small | 0.57 | 0.13 | **4.2×** |
| medium | 4.20 | 0.24 | **17.8×** |
| large | 48.78 | 0.93 | **52.4×** |

### with_xyz=True

| Case | CPU (ms) | NPU (ms) | CPU/NPU |
|------|----------|----------|---------|
| small | 1.20 | 0.38 | **3.1×** |
| medium | 8.20 | 0.40 | **20.7×** |
| large | 179.96 | 1.74 | **103.5×** |

`grouping` 为内存带宽型 gather，NPU 上 `index_select`/`gather` 类算子收益明显；`with_xyz=True` 在大规模时 CPU 额外做减法与 mask，差距更大。

## 复现

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/d00883276/cuda_to_npu/Pointcept/libs/pointops_torch
python3 bench_grouping.py --device-id 0
```

或在 Docker 内：

```bash
docker exec cuda_migration_test bash -lc \
  'source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
   cd /home/d00883276/cuda_to_npu/Pointcept/libs/pointops_torch && \
   python3 bench_grouping.py --device-id 0'
```

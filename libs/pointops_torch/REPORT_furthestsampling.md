# furthestsampling：PyTorch 移植与 CPU/NPU 评测

## 原始实现位置（Pointcept @ df36980）

| 组件 | 路径 |
|------|------|
| Python API | `libs/pointops2/functions/pointops.py` → `furthestsampling` |
| CUDA kernel | `libs/pointops2/src/sampling/sampling_cuda_kernel.cu` |
| 调用示例 | `pointcept/models/stratified_transformer/stratified_transformer_v1m2_refine.py` |

**接口**

```python
idx = pointops.furthestsampling(xyz, offset, new_offset)
# xyz: (N, 3) float, CUDA
# offset: (B,) 各 batch 输入点数的累积上界（exclusive end）
# new_offset: (B,) 各 batch 采样点数的累积上界
# idx: (M,) int32, M = new_offset[-1]
```

**算法（与 CUDA kernel 一致）**

1. 每个 batch 的第一个采样点固定为 `start_n`（该 batch 在 xyz 中的起始索引）。
2. 维护全局 `tmp[k]` = 点 k 到已选点集的最小平方距离（初值 `1e10`）。
3. 每轮以当前最远点 `old` 为圆心，更新 `tmp[k] = min(tmp[k], ||p_k - p_old||²)`，再取 `argmax(tmp)` 为下一个 `old`。
4. `offset` / `new_offset` 为累积边界，支持变长 batch。

## PyTorch 实现

见 `farthest_point_sampling.py`（`furthestsampling` 为其别名）：

```python
from pointops_torch import furthestsampling

idx = furthestsampling(xyz, offset, new_offset)  # device 与 xyz 一致
```

- **CPU / CUDA / NPU**：同一实现，`xyz` 在哪个 device 就在哪执行。
- **无自定义 CUDA / CANN 算子**，便于在 Ascend 上直接跑通。

## 精度对比（Ascend 容器 `model_migration_siglip_skill_test`，warmup=5）

| Case | N | B | M | vs fp64 参考 | CPU vs NPU | 索引一致率 |
|------|---|---|---|--------------|------------|------------|
| small | 4096 | 2 | 256 | PASS | PASS | 100% |
| medium | 16384 | 4 | 1024 | PASS | PASS | 100% |
| large | 65536 | 8 | 2048 | PASS | PASS | 100% |

fp32 CPU/NPU 与 float64 参考实现索引完全一致（确定性 FPS，无近似累加误差）。

## 性能对比（steady-state，warmup=5, repeat=30）

| Case | CPU (ms) | NPU (ms) | CPU/NPU 加速比 |
|------|----------|----------|----------------|
| small | 28.77 | 84.94 | 0.34×（NPU 更慢） |
| medium | 179.92 | 331.52 | 0.54× |
| large | 611.10 | 579.76 | **1.05×**（NPU 略快） |

说明：当前实现为**逐迭代 Python 循环 + 每轮全点距离向量**，小 batch 时 NPU 启动/同步开销占主导；点数与采样数变大后，NPU 算子吞吐优势才显现。若需逼近原 CUDA kernel，需 fused 多轮 FPS 或自定义 Ascend 算子。

## 复现

```bash
# 在已挂载 /home 的 Ascend 容器内
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/d00883276/cuda_to_npu/Pointcept/libs/pointops_torch
python3 bench_fps.py --device-id 0
```

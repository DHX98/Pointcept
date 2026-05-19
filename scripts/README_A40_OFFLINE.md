# Pointcept A40 离线性能对比包

在**不通外网**的 x86 + NVIDIA A40 机器上，从源码编译安装 Pointcept CUDA 算子，并与 Ascend NPU 迁移评测的**静态 CPU/NPU 数据**做三方对比。

## 包内容

| 路径 | 说明 |
|------|------|
| `install.sh` | 离线安装 + 首次跑 benchmark |
| `run_benchmark.sh` | 仅重新跑性能对比 |
| `libs/pointops/` | 原版 CUDA 扩展源码（knn_query、grouping、interpolation、FPS v1） |
| `libs/pointops2/` | 原版 CUDA 扩展源码（furthestsampling 等） |
| `libs/pointops_torch/` | PyTorch 移植 + `bench_compare_static.py` + REPORT 静态数据 |
| `wheels/` | 可选 `ninja` wheel（离线 pip 安装） |
| `VERSION.txt` | 打包时的 commit / 时间戳 |
| `MANIFEST.sha256` | 包内文件校验 |

官方源码对应：

https://github.com/Pointcept/Pointcept/tree/df36980119f4636beb2d02d04ef3b2fec0fddfba/libs

## A40 机器前置条件

安装包**不包含** PyTorch / CUDA 驱动，需目标机已具备：

1. **Python 3.8+**
2. **PyTorch（CUDA 版）**，且与系统 `nvcc` 主版本一致  
   ```bash
   python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
   ```
3. **CUDA toolkit**（含 `nvcc`）
4. **NVIDIA 驱动**（`nvidia-smi` 可见 A40）
5. **gcc/g++**
6. **ninja**（若未安装，包内 `wheels/` 会在 `install.sh` 中离线安装）

可选环境变量：

| 变量 | 默认 | 说明 |
|------|------|------|
| `TORCH_CUDA_ARCH_LIST` | `8.6` | A40 算力 |
| `MAX_JOBS` | CPU 核数 | 编译并行度 |
| `DEVICE_ID` | `0` | GPU 编号 |
| `SKIP_BENCHMARK` | `0` | `1` 则只安装不跑 benchmark |

## 快速开始（A40 离线机）

```bash
# 1. 校验压缩包（在拷贝完成后）
sha256sum -c pointcept_a40_offline_*.tar.gz.sha256

# 2. 解压
tar xzf pointcept_a40_offline_*.tar.gz
cd pointcept_a40_offline_*

# 3. 安装 CUDA 扩展并跑首次对比
bash install.sh

# 4. 之后仅重跑 benchmark
bash run_benchmark.sh
```

## 输出结果

安装与 benchmark 日志/报告在 `artifacts/`：

| 文件 | 内容 |
|------|------|
| `offline_install.txt` | 安装环境、GPU、torch 版本 |
| `PERF_CUDA_COMPARE.md` | **CUDA vs 静态 CPU/NPU** Markdown 表格 |
| `PERF_CUDA_COMPARE.log` | 终端完整输出 |

`PERF_CUDA_COMPARE.md` 中每个算子包含：

- **CPU (ms)** — 来自 Ascend 容器 REPORT（静态）
- **CUDA (ms)** — A40 上实测原版 `pointops` CUDA kernel
- **NPU (ms)** — 来自 REPORT（静态）
- **CPU/CUDA、CPU/NPU、CUDA/NPU** 加速比

覆盖算子：`knn_query`、`grouping`（feat / with_xyz）、`interpolation`、`knn_query_and_group`、`furthestsampling`。

## 从 GitHub 直接下载（已打包）

仓库 `releases/a40_offline/` 目录提供预构建离线包（与当前 `main` 提交一致）：

```bash
# 克隆后
cd Pointcept/releases/a40_offline
sha256sum -c pointcept_a40_offline_*.tar.gz.sha256
tar xzf pointcept_a40_offline_*.tar.gz
cd pointcept_a40_offline_* && bash install.sh
```

## 在有网络的机器上打包

在已包含本仓库的机器上执行：

```bash
cd /path/to/Pointcept
bash scripts/package_for_a40_offline.sh
# 输出: releases/a40_offline/pointcept_a40_offline_<commit>_<date>.tar.gz
#       releases/a40_offline/pointcept_a40_offline_<commit>_<date>.tar.gz.sha256
```

拷贝 `.tar.gz` 与 `.sha256` 到 A40 即可。

自定义输出目录：

```bash
OUTPUT_DIR=/data/scp bash scripts/package_for_a40_offline.sh
```

## 常见问题

**Q: `torch.cuda.is_available()` 为 False**  
A: 先修复 PyTorch/CUDA 环境，与 nvcc 版本对齐后再运行 `install.sh`。

**Q: 编译报 `ninja not found`**  
A: 确认 `wheels/ninja-*.whl` 存在；或手动安装 ninja 后重试。

**Q: 只想看静态 CPU/NPU 表、不测 CUDA**  
A: `cd libs/pointops_torch && TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 bench_compare_static.py --static-only`

**Q: GPU 不是 A40**  
A: 设置对应算力，例如 RTX 3090：`TORCH_CUDA_ARCH_LIST=8.6 bash install.sh`（与 A40 相同则为 8.6）。

**Q: `pointgroup_ops` 需要吗？**  
A: 性能对比不需要。默认 `INSTALL_POINTGROUP=0`；若需安装：`INSTALL_POINTGROUP=1 bash install.sh`（还需 sparsehash 头文件）。

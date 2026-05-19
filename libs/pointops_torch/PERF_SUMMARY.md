# PointOps Torch Migration — Static CPU/NPU Performance

_Ascend NPU container; CPU/NPU from offline REPORT*.md benchmarks. CUDA column: live when GPU + pointops available, else N/A._

| Operator | Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |
|----------|------|-------|----------|----------|---------|
| furthestsampling | small | N=4096 B=2 M=256 | 28.77 | 84.94 | 0.34x |
| furthestsampling | medium | N=16384 B=4 M=1024 | 179.92 | 331.52 | 0.54x |
| furthestsampling | large | N=65536 B=8 M=2048 | 611.10 | 579.76 | 1.05x |
| farthest_point_sampling | small | N=4096 B=2 M=256 | 29.45 | 72.99 | 0.40x |
| farthest_point_sampling | medium | N=16384 B=4 M=1024 | 185.18 | 296.15 | 0.63x |
| farthest_point_sampling | large | N=65536 B=8 M=2048 | 603.95 | 577.18 | 1.05x |
| knn_query | small | N=4096 M=512 k=16 | 34.95 | 1.05 | 33.29x |
| knn_query | medium | N=16384 M=2048 k=16 | 279.66 | 2.83 | 98.82x |
| knn_query | large | N=65536 M=8192 k=16 | 2077.83 | 16.70 | 124.42x |
| grouping_feat_only | small | m=512 k=16 c=32 | 0.57 | 0.13 | 4.38x |
| grouping_feat_only | medium | m=2048 k=16 c=64 | 4.20 | 0.24 | 17.50x |
| grouping_feat_only | large | m=8192 k=16 c=128 | 48.78 | 0.93 | 52.45x |
| grouping_with_xyz | small | m=512 k=16 c=32 | 1.20 | 0.38 | 3.16x |
| grouping_with_xyz | medium | m=2048 k=16 c=64 | 8.20 | 0.40 | 20.50x |
| grouping_with_xyz | large | m=8192 k=16 c=128 | 179.96 | 1.74 | 103.43x |
| interpolation | small | N=4096 M=512 C=32 k=3 | 31.88 | 1.13 | 28.21x |
| interpolation | medium | N=16384 M=2048 C=64 k=3 | 269.93 | 3.06 | 88.21x |
| interpolation | large | N=65536 M=8192 C=128 k=3 | 2499.75 | 17.26 | 144.83x |
| knn_query_and_group | small | N=4096 M=512 C=32 k=16 | 39.55 | 4.70 | 8.41x |
| knn_query_and_group | medium | N=16384 M=2048 C=64 k=16 | 296.88 | 40.88 | 7.26x |
| knn_query_and_group | large | N=65536 M=8192 C=128 k=16 | 2242.20 | 349.33 | 6.42x |

---

### furthestsampling

- Report: `REPORT_furthestsampling.md`
- Original CUDA: `pointops2/src/sampling/sampling_cuda_kernel.cu`
- Benchmark: warmup=5, repeat=30

| Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |
|------|-------|----------|----------|---------|
| small | N=4096 B=2 M=256 | 28.77 | 84.94 | 0.34x |
| medium | N=16384 B=4 M=1024 | 179.92 | 331.52 | 0.54x |
| large | N=65536 B=8 M=2048 | 611.10 | 579.76 | 1.05x |

### farthest_point_sampling

- Report: `REPORT.md`
- Original CUDA: `pointops/src/sampling/sampling_cuda_kernel.cu`
- Benchmark: warmup=5, repeat=30

| Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |
|------|-------|----------|----------|---------|
| small | N=4096 B=2 M=256 | 29.45 | 72.99 | 0.40x |
| medium | N=16384 B=4 M=1024 | 185.18 | 296.15 | 0.63x |
| large | N=65536 B=8 M=2048 | 603.95 | 577.18 | 1.05x |

### knn_query

- Report: `REPORT_knn_query.md`
- Original CUDA: `pointops/src/knn_query/knn_query_cuda_kernel.cu`
- Benchmark: warmup=5, repeat=30

| Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |
|------|-------|----------|----------|---------|
| small | N=4096 M=512 k=16 | 34.95 | 1.05 | 33.29x |
| medium | N=16384 M=2048 k=16 | 279.66 | 2.83 | 98.82x |
| large | N=65536 M=8192 k=16 | 2077.83 | 16.70 | 124.42x |

### grouping_feat_only

- Report: `REPORT_grouping.md`
- Original CUDA: `pointops/src/grouping/grouping_cuda_kernel.cu (high-level API is PyTorch gather)`
- Benchmark: warmup=5, repeat=30

| Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |
|------|-------|----------|----------|---------|
| small | m=512 k=16 c=32 | 0.57 | 0.13 | 4.38x |
| medium | m=2048 k=16 c=64 | 4.20 | 0.24 | 17.50x |
| large | m=8192 k=16 c=128 | 48.78 | 0.93 | 52.45x |

### grouping_with_xyz

- Report: `REPORT_grouping.md`
- Original CUDA: `pointops/src/grouping/grouping_cuda_kernel.cu (high-level API is PyTorch gather)`
- Benchmark: warmup=5, repeat=30

| Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |
|------|-------|----------|----------|---------|
| small | m=512 k=16 c=32 | 1.20 | 0.38 | 3.16x |
| medium | m=2048 k=16 c=64 | 8.20 | 0.40 | 20.50x |
| large | m=8192 k=16 c=128 | 179.96 | 1.74 | 103.43x |

### interpolation

- Report: `REPORT_interpolation.md`
- Original CUDA: `pointops/src/interpolation/interpolation_cuda_kernel.cu`
- Benchmark: warmup=5, repeat=30

| Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |
|------|-------|----------|----------|---------|
| small | N=4096 M=512 C=32 k=3 | 31.88 | 1.13 | 28.21x |
| medium | N=16384 M=2048 C=64 k=3 | 269.93 | 3.06 | 88.21x |
| large | N=65536 M=8192 C=128 k=3 | 2499.75 | 17.26 | 144.83x |

### knn_query_and_group

- Report: `REPORT_knn_query_and_group.md`
- Original CUDA: `knn_query_cuda_kernel.cu + grouping (with_xyz=True path)`
- Benchmark: warmup=5, repeat=30

| Case | Shape | CPU (ms) | NPU (ms) | CPU/NPU |
|------|-------|----------|----------|---------|
| small | N=4096 M=512 C=32 k=16 | 39.55 | 4.70 | 8.41x |
| medium | N=16384 M=2048 C=64 k=16 | 296.88 | 40.88 | 7.26x |
| large | N=65536 M=8192 C=128 k=16 | 2242.20 | 349.33 | 6.42x |

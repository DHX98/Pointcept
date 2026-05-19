"""
Pure-PyTorch farthest point sampling (FPS), equivalent to pointops2.furthestsampling.

Reference: Pointcept/libs/pointops2/src/sampling/sampling_cuda_kernel.cu
  (same algorithm as pointops.farthest_point_sampling in libs/pointops)
  - Per-batch cumulative offsets (offset / new_offset).
  - tmp[k] = min squared distance from any selected point to point k.
  - Iteratively pick argmax(tmp) as the next center.
  - First sample index per batch is always start_n (first point in the batch).

Works on CPU, CUDA, and Ascend NPU (same device as xyz).
"""

from __future__ import annotations

import torch


def farthest_point_sampling(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        xyz: (N, 3) point coordinates, float32/float16 on any supported device.
        offset: (B,) int cumulative end indices of input points per batch.
        new_offset: (B,) int cumulative end indices of output samples per batch.

    Returns:
        idx: (M,) int32 indices into xyz, M = new_offset[-1].
    """
    assert xyz.dim() == 2 and xyz.size(1) == 3, "xyz must be (N, 3)"
    assert xyz.is_contiguous(), "xyz must be contiguous"

    device = xyz.device
    offset = offset.to(device=device, dtype=torch.int64).reshape(-1)
    new_offset = new_offset.to(device=device, dtype=torch.int64).reshape(-1)
    b = offset.numel()
    assert b > 0 and new_offset.numel() == b

    m_total = int(new_offset[-1].item())
    idx = torch.empty(m_total, dtype=torch.int32, device=device)

    for bid in range(b):
        start_n = int(offset[bid - 1].item()) if bid > 0 else 0
        end_n = int(offset[bid].item())
        start_m = int(new_offset[bid - 1].item()) if bid > 0 else 0
        end_m = int(new_offset[bid].item())

        n_pts = end_n - start_n
        n_samp = end_m - start_m
        if n_samp == 0:
            continue

        pts = xyz[start_n:end_n]
        tmp = torch.full((n_pts,), 1e10, device=device, dtype=xyz.dtype)

        idx[start_m] = start_n
        old_abs = start_n

        for j in range(1, n_samp):
            center = xyz[old_abs : old_abs + 1]
            dist = ((pts - center) ** 2).sum(dim=1)
            tmp = torch.minimum(tmp, dist)
            old_local = int(torch.argmax(tmp).item())
            old_abs = start_n + old_local
            idx[start_m + j] = old_abs

    return idx


def farthest_point_sampling_reference_fp64(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
) -> torch.Tensor:
    """High-precision CPU reference (float64), for correctness checks only."""
    xyz64 = xyz.detach().cpu().double()
    offset = offset.detach().cpu().long()
    new_offset = new_offset.detach().cpu().long()
    b = offset.numel()
    m_total = int(new_offset[-1].item())
    idx = torch.empty(m_total, dtype=torch.int64)

    for bid in range(b):
        start_n = int(offset[bid - 1].item()) if bid > 0 else 0
        end_n = int(offset[bid].item())
        start_m = int(new_offset[bid - 1].item()) if bid > 0 else 0
        end_m = int(new_offset[bid].item())

        pts = xyz64[start_n:end_n]
        tmp = torch.full((end_n - start_n,), 1e30, dtype=torch.float64)
        idx[start_m] = start_n
        old_abs = start_n

        for j in range(1, end_m - start_m):
            center = xyz64[old_abs]
            dist = ((pts - center) ** 2).sum(dim=1)
            tmp = torch.minimum(tmp, dist)
            old_local = int(torch.argmax(tmp).item())
            old_abs = start_n + old_local
            idx[start_m + j] = old_abs

    return idx.to(torch.int32)


# pointops2 API alias
furthestsampling = farthest_point_sampling

"""
Pure-PyTorch KNN query, equivalent to pointops.knn_query.

Reference: Pointcept/libs/pointops/src/knn_query/knn_query_cuda_kernel.cu
  - Per query point, search only within the xyz batch slice [offset[b-1], offset[b]).
  - Batch id for query pt_idx comes from new_offset (same as CUDA get_bt_idx).
  - k smallest squared distances via topk (CUDA uses a size-k max-heap + heap_sort).
  - Returns (idx, sqrt(dist2)) matching the original Python wrapper.

Works on CPU, CUDA, and Ascend NPU. CUDA kernel supports nsample <= 128 only.
"""

from __future__ import annotations

import torch


def knn_query(
    nsample: int,
    xyz: torch.Tensor,
    offset: torch.Tensor,
    new_xyz: torch.Tensor | None = None,
    new_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        nsample: number of neighbors per query point (<= 128 for CUDA parity).
        xyz: (N, 3) database points.
        offset: (B,) cumulative end indices of xyz per batch.
        new_xyz: (M, 3) query points; defaults to xyz.
        new_offset: (B,) cumulative end indices of queries; defaults to offset.

    Returns:
        idx: (M, nsample) int32 indices into xyz.
        dist: (M, nsample) Euclidean distances (sqrt of squared distances).
    """
    if new_xyz is None or new_offset is None:
        new_xyz = xyz
        new_offset = offset

    assert xyz.dim() == 2 and xyz.size(1) == 3
    assert new_xyz.dim() == 2 and new_xyz.size(1) == 3
    assert xyz.is_contiguous() and new_xyz.is_contiguous()

    device = xyz.device
    dtype = xyz.dtype
    offset = offset.to(device=device, dtype=torch.int64).reshape(-1)
    new_offset = new_offset.to(device=device, dtype=torch.int64).reshape(-1)
    b = offset.numel()
    assert b > 0 and new_offset.numel() == b

    m = new_xyz.shape[0]
    idx = torch.empty(m, nsample, dtype=torch.int32, device=device)
    dist2 = torch.empty(m, nsample, dtype=dtype, device=device)

    for bid in range(b):
        start_n = int(offset[bid - 1].item()) if bid > 0 else 0
        end_n = int(offset[bid].item())
        start_m = int(new_offset[bid - 1].item()) if bid > 0 else 0
        end_m = int(new_offset[bid].item())

        n_pts = end_n - start_n
        n_q = end_m - start_m
        if n_q == 0:
            continue

        pts = xyz[start_n:end_n]
        queries = new_xyz[start_m:end_m]
        k = min(nsample, n_pts)

        d2 = ((queries.unsqueeze(1) - pts.unsqueeze(0)) ** 2).sum(dim=-1)
        topd2, topi = torch.topk(d2, k, dim=1, largest=False, sorted=True)
        idx[start_m:end_m, :k] = topi.to(torch.int32) + start_n
        dist2[start_m:end_m, :k] = topd2
        if k < nsample:
            idx[start_m:end_m, k:] = -1
            dist2[start_m:end_m, k:] = 1e10

    return idx, torch.sqrt(dist2)


def _heap_reheap(dist: list[float], idx: list[int], k: int) -> None:
    root = 0
    child = root * 2 + 1
    while child < k:
        if child + 1 < k and dist[child + 1] > dist[child]:
            child += 1
        if dist[root] > dist[child]:
            return
        dist[root], dist[child] = dist[child], dist[root]
        idx[root], idx[child] = idx[child], idx[root]
        root = child
        child = root * 2 + 1


def _heap_sort(dist: list[float], idx: list[int], k: int) -> None:
    for i in range(k - 1, 0, -1):
        dist[0], dist[i] = dist[i], dist[0]
        idx[0], idx[i] = idx[i], idx[0]
        _heap_reheap(dist, idx, i)


def knn_query_reference_fp64(
    nsample: int,
    xyz: torch.Tensor,
    offset: torch.Tensor,
    new_xyz: torch.Tensor | None = None,
    new_offset: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU float64 reference using the same max-heap logic as the CUDA kernel."""
    if new_xyz is None or new_offset is None:
        new_xyz = xyz
        new_offset = offset

    xyz64 = xyz.detach().cpu().double()
    new_xyz64 = new_xyz.detach().cpu().double()
    offset = offset.detach().cpu().long()
    new_offset = new_offset.detach().cpu().long()

    m = new_xyz64.shape[0]
    idx_out = torch.full((m, nsample), -1, dtype=torch.int64)
    dist2_out = torch.full((m, nsample), 1e10, dtype=torch.float64)

    def get_bt_idx(pt_idx: int) -> int:
        i = 0
        while True:
            if pt_idx < int(new_offset[i].item()):
                break
            i += 1
        return i

    for pt_idx in range(m):
        bt = get_bt_idx(pt_idx)
        start = int(offset[bt - 1].item()) if bt > 0 else 0
        end = int(offset[bt].item())
        qx, qy, qz = new_xyz64[pt_idx].tolist()

        best_dist = [1e30] * nsample
        best_idx = [-1] * nsample
        k = min(nsample, end - start)
        for i in range(start, end):
            x, y, z = xyz64[i].tolist()
            d2 = (qx - x) ** 2 + (qy - y) ** 2 + (qz - z) ** 2
            if d2 < best_dist[0]:
                best_dist[0] = d2
                best_idx[0] = i
                _heap_reheap(best_dist, best_idx, nsample)

        _heap_sort(best_dist, best_idx, nsample)
        idx_out[pt_idx, :k] = torch.tensor(best_idx[:k], dtype=torch.int64)
        dist2_out[pt_idx, :k] = torch.tensor(best_dist[:k], dtype=torch.float64)

    return idx_out.to(torch.int32), torch.sqrt(dist2_out.float())

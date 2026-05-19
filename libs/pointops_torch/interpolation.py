"""
Pure-PyTorch interpolation, equivalent to pointops.interpolation.

Reference: Pointcept/libs/pointops/functions/interpolation.py (commit df36980)
  - knn_query(k) -> idx (n, k), dist (n, k)
  - weight_i = (1 / (dist_i + 1e-8)) / sum_j(1 / (dist_j + 1e-8))
  - output[n, c] = sum_i feat[idx[n, i], c] * weight[n, i]

CUDA gather kernel (interpolation_forward_cuda) is equivalent to the vectorized
indexing below. Works on CPU, CUDA, and Ascend NPU.
"""

from __future__ import annotations

import torch
from torch.autograd import Function

try:
    from .knn_query import knn_query, knn_query_reference_fp64
except ImportError:
    from knn_query import knn_query, knn_query_reference_fp64


def _weights_from_dist(dist: torch.Tensor) -> torch.Tensor:
    dist_recip = 1.0 / (dist + 1e-8)
    return dist_recip / dist_recip.sum(dim=1, keepdim=True)


def _interp_gather(feat: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted KNN feature interpolation: (n, c)."""
    gathered = feat[idx.long()]  # (n, k, c)
    return (gathered * weight.unsqueeze(-1)).sum(dim=1)


def interpolation(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    feat: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """
    Args:
        xyz: (m, 3) source point coordinates.
        new_xyz: (n, 3) query coordinates.
        feat: (m, c) source features.
        offset: (B,) cumulative end indices for xyz batches.
        new_offset: (B,) cumulative end indices for new_xyz batches.
        k: number of neighbors (default 3).

    Returns:
        (n, c) interpolated features at new_xyz.
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    idx, dist = knn_query(k, xyz, offset, new_xyz, new_offset)
    weight = _weights_from_dist(dist)
    return _interp_gather(feat, idx, weight)


def interpolation_reference_fp64(
    xyz: torch.Tensor,
    new_xyz: torch.Tensor,
    feat: torch.Tensor,
    offset: torch.Tensor,
    new_offset: torch.Tensor,
    k: int = 3,
) -> torch.Tensor:
    """CPU float64 reference (fp64 knn + fp64 gather/weight, same formula as pointops)."""
    idx, dist = knn_query_reference_fp64(k, xyz, offset, new_xyz, new_offset)
    feat64 = feat.detach().cpu().double()
    idx64 = idx.detach().cpu().long()
    dist64 = dist.detach().cpu().double()
    weight = _weights_from_dist(dist64)
    gathered = feat64[idx64]
    return (gathered * weight.unsqueeze(-1)).sum(dim=1).float()


class Interpolation(Function):
    """Autograd wrapper matching pointops.interpolation2 (CUDA forward/backward)."""

    @staticmethod
    def forward(
        ctx,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor,
        input: torch.Tensor,
        offset: torch.Tensor,
        new_offset: torch.Tensor,
        k: int = 3,
    ) -> torch.Tensor:
        assert xyz.is_contiguous() and new_xyz.is_contiguous() and input.is_contiguous()
        idx, dist = knn_query(k, xyz, offset, new_xyz, new_offset)
        weight = _weights_from_dist(dist)
        output = _interp_gather(input, idx, weight)
        ctx.save_for_backward(idx, weight)
        ctx.m = input.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        idx, weight = ctx.saved_tensors
        m = ctx.m
        k = weight.shape[1]
        n, c = grad_output.shape

        grad_input = torch.zeros(m, c, device=grad_output.device, dtype=grad_output.dtype)
        for ki in range(k):
            src = idx[:, ki].long()
            grad_input.index_add_(0, src, grad_output * weight[:, ki].unsqueeze(-1))
        return None, None, grad_input, None, None, None


interpolation2 = Interpolation.apply

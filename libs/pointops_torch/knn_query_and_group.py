"""
Pure-PyTorch knn_query_and_group, equivalent to pointops.knn_query_and_group.

Reference: Pointcept/libs/pointops/functions/utils.py
  - knn_query(nsample, ...) when idx is None
  - grouping(idx, feat, xyz, new_xyz, with_xyz)

Works on CPU, CUDA, and Ascend NPU.
"""

from __future__ import annotations

import torch

try:
    from .grouping import grouping
    from .knn_query import knn_query, knn_query_reference_fp64
except ImportError:
    from grouping import grouping
    from knn_query import knn_query, knn_query_reference_fp64


def knn_query_and_group(
    feat: torch.Tensor,
    xyz: torch.Tensor,
    offset: torch.Tensor | None = None,
    new_xyz: torch.Tensor | None = None,
    new_offset: torch.Tensor | None = None,
    idx: torch.Tensor | None = None,
    nsample: int | None = None,
    with_xyz: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        feat: (n, c) point features.
        xyz: (n, 3) point coordinates.
        offset: (b,) cumulative end indices of xyz per batch.
        new_xyz: (m, 3) query points; defaults to xyz.
        new_offset: (b,) cumulative end indices of queries; defaults to offset.
        idx: optional precomputed (m, nsample) neighbor indices.
        nsample: k neighbors when idx is None.
        with_xyz: prepend relative xyz to grouped features.

    Returns:
        grouped: (m, nsample, c) or (m, nsample, 3+c).
        idx: (m, nsample) neighbor indices.
    """
    if idx is None:
        assert nsample is not None
        idx, _ = knn_query(nsample, xyz, offset, new_xyz, new_offset)
    return grouping(idx, feat, xyz, new_xyz, with_xyz), idx


def knn_query_and_group_reference_fp64(
    feat: torch.Tensor,
    xyz: torch.Tensor,
    offset: torch.Tensor | None = None,
    new_xyz: torch.Tensor | None = None,
    new_offset: torch.Tensor | None = None,
    idx: torch.Tensor | None = None,
    nsample: int | None = None,
    with_xyz: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU fp64 reference matching the original Python wrapper semantics."""
    if new_xyz is None or new_offset is None:
        new_xyz = xyz
        new_offset = offset

    xyz64 = xyz.detach().cpu().double()
    feat64 = feat.detach().cpu().double()
    new_xyz64 = new_xyz.detach().cpu().double()
    off = offset.detach().cpu()
    new_off = new_offset.detach().cpu()

    if idx is None:
        assert nsample is not None
        idx, _ = knn_query_reference_fp64(nsample, xyz64, off, new_xyz64, new_off)
    else:
        idx = idx.detach().cpu()

    m, nsample_k, c = idx.shape[0], idx.shape[1], feat64.shape[1]
    idx64 = idx.long()
    xyz_pad = torch.cat([xyz64, torch.zeros(1, 3, dtype=torch.float64)], dim=0)
    feat_pad = torch.cat([feat64, torch.zeros(1, c, dtype=torch.float64)], dim=0)
    flat_idx = idx64.view(-1)
    grouped_feat = feat_pad[flat_idx, :].view(m, nsample_k, c)

    if with_xyz:
        mask = torch.sign(idx64 + 1).double()
        grouped_xyz = xyz_pad[flat_idx, :].view(m, nsample_k, 3) - new_xyz64.unsqueeze(1)
        grouped_xyz = grouped_xyz * mask.unsqueeze(-1)
        grouped = torch.cat((grouped_xyz, grouped_feat), dim=-1)
    else:
        grouped = grouped_feat

    return grouped, idx.to(torch.int32)

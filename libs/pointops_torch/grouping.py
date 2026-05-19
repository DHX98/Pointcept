"""
Pure-PyTorch grouping, equivalent to pointops.grouping.

Reference: Pointcept/libs/pointops/functions/grouping.py (commit df36980)
  - Pad xyz/feat with one zero row (sentinel for idx == -1).
  - grouped_feat[m,s,c] = feat[idx[m,s], c]
  - with_xyz: relative coords (xyz[idx] - new_xyz) masked by sign(idx+1).

The low-level CUDA kernel (grouping_forward_cuda) gathers (n,c) by (m,nsample)
indices; this module matches the public Python API used by Point Transformer.

Works on CPU, CUDA, and Ascend NPU.
"""

from __future__ import annotations

import torch


def grouping(
    idx: torch.Tensor,
    feat: torch.Tensor,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor | None = None,
    with_xyz: bool = False,
) -> torch.Tensor:
    """
    Args:
        idx: (m, nsample) neighbor indices into xyz/feat (-1 for padding).
        feat: (n, c) point features.
        xyz: (n, 3) point coordinates.
        new_xyz: (m, 3) center coordinates; defaults to xyz.
        with_xyz: if True, concat masked relative xyz with grouped features.

    Returns:
        (m, nsample, c) or (m, nsample, 3+c) when with_xyz is True.
    """
    if new_xyz is None:
        new_xyz = xyz
    assert xyz.is_contiguous() and feat.is_contiguous()
    m, nsample, c = idx.shape[0], idx.shape[1], feat.shape[1]
    device, dtype = feat.device, feat.dtype

    xyz_pad = torch.cat([xyz, torch.zeros(1, 3, device=device, dtype=xyz.dtype)], dim=0)
    feat_pad = torch.cat([feat, torch.zeros(1, c, device=device, dtype=dtype)], dim=0)
    flat_idx = idx.reshape(-1).long()
    grouped_feat = feat_pad[flat_idx].view(m, nsample, c)

    if not with_xyz:
        return grouped_feat

    assert new_xyz.is_contiguous()
    mask = torch.sign(idx + 1).to(dtype)
    grouped_xyz = xyz_pad[flat_idx].view(m, nsample, 3) - new_xyz.unsqueeze(1)
    grouped_xyz = grouped_xyz * mask.unsqueeze(-1)
    return torch.cat((grouped_xyz, grouped_feat), dim=-1)


def grouping_reference_fp64(
    idx: torch.Tensor,
    feat: torch.Tensor,
    xyz: torch.Tensor,
    new_xyz: torch.Tensor | None = None,
    with_xyz: bool = False,
) -> torch.Tensor:
    """CPU float64 loop reference matching the original Python implementation."""
    idx64 = idx.detach().cpu().long()
    feat64 = feat.detach().cpu().double()
    xyz64 = xyz.detach().cpu().double()
    if new_xyz is None:
        new_xyz64 = xyz64
    else:
        new_xyz64 = new_xyz.detach().cpu().double()

    m, nsample, c = idx64.shape[0], idx64.shape[1], feat64.shape[1]
    n = feat64.shape[0]
    xyz_pad = torch.cat([xyz64, torch.zeros(1, 3, dtype=torch.float64)], dim=0)
    feat_pad = torch.cat([feat64, torch.zeros(1, c, dtype=torch.float64)], dim=0)

    out_feat = torch.empty(m, nsample, c, dtype=torch.float64)
    for mi in range(m):
        for si in range(nsample):
            ii = int(idx64[mi, si].item())
            if ii < 0:
                ii = n
            out_feat[mi, si] = feat_pad[ii]

    if not with_xyz:
        return out_feat.float()

    out_xyz = torch.empty(m, nsample, 3, dtype=torch.float64)
    for mi in range(m):
        for si in range(nsample):
            ii = int(idx64[mi, si].item())
            mask = 1.0 if ii >= 0 else 0.0
            if ii < 0:
                ii = n
            rel = xyz_pad[ii] - new_xyz64[mi]
            out_xyz[mi, si] = rel * mask
    return torch.cat((out_xyz, out_feat), dim=-1).float()

"""
Domain alignment utilities (CORAL).
"""

from __future__ import annotations

import torch
from torch import Tensor


def _covariance(feats: Tensor) -> Tensor:
    """Compute covariance matrix of features (B,C) -> (C,C)."""
    if feats.dim() != 2:
        feats = feats.reshape(feats.size(0), -1)
    mean = feats.mean(dim=0, keepdim=True)
    centered = feats - mean
    cov = centered.t().matmul(centered) / (feats.size(0) - 1 + 1e-5)
    return cov


def coral_loss(
    src: Tensor,
    tgt: Tensor,
    align_mean: bool = True,
) -> Tensor:
    """
    CORAL loss aligning covariances (and optionally means) between source and target.
    Args:
        src: (B,C) source features
        tgt: (B,C) target features
        align_mean: if True, include L2 mean alignment
    """
    if src.dim() != 2:
        src = src.reshape(src.size(0), -1)
    if tgt.dim() != 2:
        tgt = tgt.reshape(tgt.size(0), -1)
    cov_src = _covariance(src)
    cov_tgt = _covariance(tgt)
    cov_loss = torch.mean((cov_src - cov_tgt) ** 2)
    if not align_mean:
        return cov_loss
    mean_loss = torch.mean((src.mean(dim=0) - tgt.mean(dim=0)) ** 2)
    return cov_loss + mean_loss

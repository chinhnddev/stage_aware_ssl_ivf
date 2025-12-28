"""
WP4 heads: quality (binary) and optional stage head, plus multitask loss helpers.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class QualityHead(nn.Module):
    """Binary quality head; returns logits."""

    def __init__(self, in_dim: int, hidden: Optional[int] = None, dropout: float = 0.2):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class StageHead(nn.Module):
    """Optional stage classifier; returns logits over stage classes."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def focal_bce_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 0.0, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Binary focal BCE; gamma=0 reduces to BCEWithLogits."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    if gamma <= 0:
        return bce.mean()
    p = torch.sigmoid(logits)
    pt = torch.where(targets == 1, p, 1 - p)
    focal = (1 - pt) ** gamma * bce
    return focal.mean()


def compute_multitask_loss(
    quality_logits: torch.Tensor,
    quality_targets: torch.Tensor,
    stage_logits: Optional[torch.Tensor] = None,
    stage_targets: Optional[torch.Tensor] = None,
    stage_mask: Optional[torch.Tensor] = None,
    lambda_stage: float = 0.0,
    pos_weight: Optional[torch.Tensor] = None,
    focal_gamma: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Compute L_total = L_quality + lambda_stage * masked L_stage."""
    quality_targets = quality_targets.float()
    l_quality = focal_bce_loss(quality_logits, quality_targets, gamma=focal_gamma, pos_weight=pos_weight)
    total = l_quality
    l_stage = torch.tensor(0.0, device=quality_logits.device)
    if stage_logits is not None and stage_targets is not None and lambda_stage > 0:
        if stage_mask is not None:
            mask = stage_mask.float()
            if mask.sum() > 0:
                l_stage = (
                    F.cross_entropy(stage_logits, stage_targets, reduction="none") * mask
                ).sum() / mask.sum()
        else:
            l_stage = F.cross_entropy(stage_logits, stage_targets)
        total = total + lambda_stage * l_stage
    return {"total": total, "l_quality": l_quality, "l_stage": l_stage}

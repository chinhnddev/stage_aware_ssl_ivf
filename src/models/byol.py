"""
BYOL implementation (simplified) for stage-aware SSL.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOL(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        proj_dim: int = 256,
        pred_dim: int = 256,
        tau: float = 0.996,
    ) -> None:
        super().__init__()
        self.online_backbone = backbone
        self.online_projector = _build_mlp(feature_dim, feature_dim, proj_dim)
        self.online_predictor = _build_mlp(proj_dim, proj_dim, pred_dim)

        import copy

        self.target_backbone = copy.deepcopy(backbone)
        self.target_projector = _build_mlp(feature_dim, feature_dim, proj_dim)
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.tau = tau

    @torch.no_grad()
    def _update_target(self) -> None:
        for online, target in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
            target.data = target.data * self.tau + online.data * (1.0 - self.tau)
        for online, target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target.data = target.data * self.tau + online.data * (1.0 - self.tau)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Online path
        o1 = self.online_backbone(x1)
        o2 = self.online_backbone(x2)
        z1 = self.online_projector(o1)
        z2 = self.online_projector(o2)
        p1 = self.online_predictor(z1)
        p2 = self.online_predictor(z2)

        # Target path (no grad)
        with torch.no_grad():
            t1 = self.target_projector(self.target_backbone(x1))
            t2 = self.target_projector(self.target_backbone(x2))

        return p1, p2, t1.detach(), t2.detach()

    def update_moving_average(self) -> None:
        self._update_target()


def byol_loss(p1: torch.Tensor, p2: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    p1 = F.normalize(p1, dim=-1)
    p2 = F.normalize(p2, dim=-1)
    t1 = F.normalize(t1, dim=-1)
    t2 = F.normalize(t2, dim=-1)
    loss = 2 - 2 * (p1 * t2).sum(dim=-1).mean() - 2 * (p2 * t1).sum(dim=-1).mean()
    return loss * 0.5

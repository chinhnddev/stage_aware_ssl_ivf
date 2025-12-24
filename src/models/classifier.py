"""
Classifier head and backbone loader for supervised finetuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torchvision import models


def build_backbone(name: str = "resnet50") -> Tuple[nn.Module, int]:
    if name.lower() == "resnet50":
        backbone = models.resnet50(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feat_dim
    raise ValueError(f"Unsupported backbone {name}")


class LinearHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x
        else:
            x = self.pool(x).flatten(1)
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x
        else:
            x = self.pool(x).flatten(1)
        return self.net(x)


class EmbryoClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)


def load_pretrained_backbone(backbone: nn.Module, path: Path, from_byol_ckpt: bool = False) -> None:
    state = torch.load(path, map_location="cpu")
    if from_byol_ckpt:
        state = state.get("model", state)
        state = {k.replace("online_backbone.", ""): v for k, v in state.items() if k.startswith("online_backbone.")}
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing:
        print(f"[pretrain] missing keys: {missing}")
    if unexpected:
        print(f"[pretrain] unexpected keys: {unexpected}")


def freeze_backbone(backbone: nn.Module, until: str = "none") -> None:
    for p in backbone.parameters():
        p.requires_grad = False
    if until == "layer4":
        for name, module in backbone.named_children():
            if name == "layer4":
                for p in module.parameters():
                    p.requires_grad = True
    elif until == "all":
        for p in backbone.parameters():
            p.requires_grad = True

"""
Frozen backbone encoder utilities for linear probing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torchvision import models

try:
    import timm  # type: ignore
except ImportError:
    timm = None


class FrozenBackboneEncoder(nn.Module):
    """
    Wrap a backbone and expose frozen features for linear probing.

    - requires_grad=False for all backbone params
    - backbone kept in eval mode during forward
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        global_pool: str = "avg",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.global_pool = global_pool
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)
        if feats.dim() == 4 and self.global_pool == "avg":
            feats = feats.mean(dim=(-2, -1))
        return feats


def _build_resnet(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    name = name.lower()
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, dim
    raise ValueError(f"Unsupported torchvision backbone: {name}")


def _build_timm(name: str, pretrained: bool) -> Tuple[nn.Module, int, str]:
    if timm is None:
        raise ImportError("timm is not installed; cannot build timm backbone.")
    model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="")
    if not hasattr(model, "num_features"):
        raise ValueError(f"timm model {name} missing num_features attribute")
    return model, int(model.num_features), getattr(model, "global_pool", "avg")


def build_frozen_backbone(
    name: str,
    pretrained: bool = True,
    checkpoint: Optional[Path] = None,
    from_byol_ckpt: bool = False,
) -> FrozenBackboneEncoder:
    """
    Build a frozen backbone encoder. Supports torchvision ResNet or timm models (e.g., vit_base_patch16_224).
    If checkpoint is provided, it is loaded after model construction.
    """
    name_l = name.lower()
    if name_l.startswith("vit") or name_l.startswith("convnext") or (timm and name_l in timm.list_models()):
        backbone, dim, gp = _build_timm(name, pretrained=pretrained)
        global_pool = gp if isinstance(gp, str) else "avg"
    else:
        backbone, dim = _build_resnet(name, pretrained=pretrained)
        global_pool = "avg"

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        if from_byol_ckpt:
            state = state.get("model", state)
            state = {k.replace("online_backbone.", ""): v for k, v in state.items() if k.startswith("online_backbone.")}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            print(f"[frozen_backbone] missing keys: {missing}")
        if unexpected:
            print(f"[frozen_backbone] unexpected keys: {unexpected}")

    return FrozenBackboneEncoder(backbone, dim, global_pool=global_pool)

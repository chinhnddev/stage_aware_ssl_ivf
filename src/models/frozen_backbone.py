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
        if feats.dim() == 4:
            # (B, C, H, W) -> global average pool -> (B, C)
            feats = feats.mean(dim=(-2, -1))
        elif feats.dim() == 3:
            # (B, N, D) -> mean over tokens -> (B, D)
            feats = feats.mean(dim=1)
        elif feats.dim() == 2:
            # already (B, D), keep as is
            pass
        elif feats.dim() == 1:
            feats = feats.unsqueeze(1)
            raise ValueError("Backbone output is 1D (B,); expected feature tensor (B, D).")
        else:
            raise ValueError(f"Unexpected backbone output shape: {feats.shape}")
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


def _unwrap_state_dict(raw_state: dict) -> dict:
    """
    Try to unwrap common checkpoint containers and strip typical prefixes.
    """
    candidate = raw_state
    for key in ["state_dict", "model", "backbone", "encoder", "online_network", "student", "teacher"]:
        if isinstance(candidate, dict) and key in candidate:
            candidate = candidate[key]
    if not isinstance(candidate, dict):
        raise ValueError("Checkpoint is not a state_dict-like mapping.")
    # Strip common prefixes
    cleaned = {}
    for k, v in candidate.items():
        new_k = k
        for prefix in ["module.", "online_backbone.", "backbone.", "encoder.", "model."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix) :]
        cleaned[new_k] = v
    return cleaned


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
        raw_state = torch.load(checkpoint, map_location="cpu")
        state = _unwrap_state_dict(raw_state)
        if from_byol_ckpt:
            # Already stripped in unwrap; keep keys that look like backbone parameters
            state = {k: v for k, v in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        matched = len(state) - len(unexpected)
        total_expected = len(backbone.state_dict())
        match_ratio = matched / max(total_expected, 1)
        print(
            f"[frozen_backbone] load checkpoint {checkpoint} | matched={matched} missing={len(missing)} "
            f"unexpected={len(unexpected)} match_ratio={match_ratio:.3f}"
        )
        if missing:
            print(f"[frozen_backbone] missing keys (truncated): {missing[:10]}")
        if unexpected:
            print(f"[frozen_backbone] unexpected keys (truncated): {unexpected[:10]}")
        if match_ratio < 0.2:
            raise ValueError(
                f"Checkpoint appears incompatible with backbone '{name}': match_ratio={match_ratio:.3f}. "
                f"Check that ssl_checkpoint matches backbone_name and export format."
            )

    return FrozenBackboneEncoder(backbone, dim, global_pool=global_pool)

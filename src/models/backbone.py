"""
Backbone factory for SSL.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision import models

try:
    import timm  # type: ignore
except ImportError:
    timm = None


def build_backbone(name: str = "resnet50", pretrained: bool = False) -> Tuple[nn.Module, int]:
    """Factory for SSL backbone. Supports ResNet and timm ViT variants."""
    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, dim

    if name.startswith("vit"):
        if timm is None:
            raise ImportError("timm is required for ViT backbones.")
        model = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
        dim = getattr(model, "num_features", None)
        if dim is None:
            raise ValueError(f"timm model {name} missing num_features")
        return model, int(dim)

    raise ValueError(f"Unsupported backbone: {name}")

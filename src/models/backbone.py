"""
Backbone factory for SSL.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision import models


def build_backbone(name: str = "resnet50") -> Tuple[nn.Module, int]:
    if name == "resnet50":
        net = models.resnet50(weights=None)
        dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, dim
    raise ValueError(f"Unsupported backbone: {name}")

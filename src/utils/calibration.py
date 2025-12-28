"""
Calibration utilities: temperature scaling and ECE.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TemperatureScaler(nn.Module):
    """Learn a single temperature parameter for post-hoc calibration."""

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=1e-6)
        return logits / temp

    @torch.no_grad()
    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 1000, lr: float = 0.01) -> float:
        """Fit temperature on validation logits/targets (binary)."""
        opt = torch.optim.LBFGS([self.log_temp], lr=lr, max_iter=max_iter)
        targets = targets.float()

        def _closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self.forward(logits), targets)
            loss.backward()
            return loss

        opt.step(_closure)
        return float(self.temperature.detach().cpu())


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error for binary probs."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply learned temperature to numpy logits."""
    return logits / max(temperature, 1e-6)

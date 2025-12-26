"""
Checkpoint utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.serialization as ts


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    # Explicitly allow full objects since we trust our own checkpoints.
    # Add DataLoader to safe globals to be compatible with PyTorch 2.6+ default sandboxing.
    ts.add_safe_globals([torch.utils.data.dataloader.DataLoader])
    return torch.load(path, map_location=map_location, weights_only=False)

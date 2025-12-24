"""
Checkpoint utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)

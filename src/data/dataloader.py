"""
DataLoader utilities for IVF embryo datasets.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import IVFImageDataset
from .transforms import build_eval_transforms, build_train_transforms


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloaders(
    csv_paths: Dict[str, List[Path]],
    root: Path,
    mean: Sequence[float],
    std: Sequence[float],
    batch_size: int = 32,
    num_workers: int = 4,
    use_clahe: bool = False,
    clahe_threshold: float = 60.0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    csv_paths: dict with keys train/val/test -> list of CSV paths to combine.
    """
    set_seed(seed)

    loaders: Dict[str, DataLoader] = {}

    train_transform = build_train_transforms(mean, std, use_clahe=use_clahe, clahe_threshold=clahe_threshold)
    eval_transform = build_eval_transforms(mean, std, use_clahe=use_clahe, clahe_threshold=clahe_threshold)

    for split, paths in csv_paths.items():
        if not paths:
            continue
        datasets = [IVFImageDataset(p, root=root, transform=train_transform if split == "train" else eval_transform) for p in paths]
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            from torch.utils.data import ConcatDataset

            dataset = ConcatDataset(datasets)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        loaders[split] = loader

    return loaders

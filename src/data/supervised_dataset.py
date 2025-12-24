"""
Supervised dataset for embryo quality classification.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class IVFClassifDataset(Dataset):
    """
    Supervised dataset that resolves image paths via root_map/root_dir and returns (image, label).
    """

    def __init__(
        self,
        csv_path: Path,
        split: Optional[str],
        root_dir: Optional[Path] = None,
        root_map: Optional[Dict[str, Union[str, Path]]] = None,
        use_domain: Optional[str] = None,
        transform=None,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.root_map = {k: Path(v) for k, v in root_map.items()} if root_map else None
        self.transform = transform

        df = pd.read_csv(csv_path)
        if use_domain:
            df = df[df["domain"] == use_domain].reset_index(drop=True)
        if split:
            df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No samples for split={split} domain={use_domain} in {csv_path}")

        def resolve(image_path: str, domain: Optional[str]) -> Path:
            if os.path.isabs(image_path):
                return Path(image_path)
            if self.root_map is not None:
                if domain not in self.root_map:
                    raise KeyError(f"domain '{domain}' not in root_map keys={list(self.root_map.keys())}")
                base = self.root_map[domain]
            elif self.root_dir is not None:
                base = self.root_dir
            else:
                raise ValueError("Either root_dir or root_map must be provided.")
            # tolerate case differences for hospital paths (ed1 vs ED1)
            candidate = base / image_path
            if candidate.exists():
                return candidate
            lower_rel = Path(*[p.lower() for p in Path(image_path).parts])
            candidate2 = base / lower_rel
            if candidate2.exists():
                return candidate2
            raise FileNotFoundError(f"Image not found for {image_path} under {base}")

        df["abs_path"] = df.apply(lambda r: resolve(r["image_path"], r.get("domain")), axis=1)
        self.labels: List[float] = df["unified_label"].astype(float).tolist()
        self.paths: List[Path] = df["abs_path"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label

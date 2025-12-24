"""
Stage-aware self-supervised dataset for IVF embryo images.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def infer_day_from_path(path: str) -> Optional[int]:
    path_lower = path.lower()
    if "day3" in path_lower or "d3" in path_lower:
        return 3
    if "day5" in path_lower or "d5" in path_lower:
        return 5
    return None


def build_stage_id(day: Optional[int], stage_group: Optional[str]) -> str:
    if day is not None and stage_group:
        return f"D{day}_{stage_group}"
    if day is not None:
        return f"D{day}"
    return "unknown"


@dataclass
class SSLSample:
    img1: any
    img2: any
    stage_id: str
    index: int


class IVFSSLDataset(Dataset):
    """
    Returns two augmented views per image plus stage_id metadata.
    """

    def __init__(
        self,
        csv_paths: List[Path],
        root_dir: Path,
        transform1: Callable,
        transform2: Callable,
    ) -> None:
        self.root_dir = Path(root_dir)
        dfs = [pd.read_csv(p) for p in csv_paths]
        df = pd.concat(dfs, ignore_index=True)

        # Drop rows whose image files are missing to avoid worker FileNotFoundError.
        df["abs_path"] = df["image_path"].apply(lambda p: self.root_dir / str(p))
        exists_mask = df["abs_path"].apply(lambda p: p.exists())
        missing = (~exists_mask).sum()
        df = df[exists_mask].reset_index(drop=True)
        if missing:
            print(f"[IVFSSLDataset] Skipped {missing} missing files; remaining {len(df)} samples.")
        if len(df) == 0:
            raise ValueError(
                f"No images found after filtering. Check data.root_dir='{self.root_dir}' and csv image_path values."
            )
        # Allowed domains/datasets handled upstream; ignore label columns.
        self.df = df

        # Resolve stage metadata
        days = df["day"] if "day" in df.columns else None
        stage_groups = df["stage_group"] if "stage_group" in df.columns else None
        stage_ids: List[str] = []
        for _, row in df.iterrows():
            day = int(row["day"]) if days is not None and not pd.isna(row["day"]) else infer_day_from_path(
                str(row["image_path"])
            )
            stage_group = row["stage_group"] if stage_groups is not None and isinstance(row["stage_group"], str) else None
            stage_id = build_stage_id(day, stage_group)
            stage_ids.append(stage_id)
        self.stage_ids = stage_ids

        self.transform1 = transform1
        self.transform2 = transform2

        # Build stage -> indices map for potential use
        self.stage_to_indices: Dict[str, List[int]] = {}
        for idx, sid in enumerate(self.stage_ids):
            self.stage_to_indices.setdefault(sid, []).append(idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]
        img_path = row["abs_path"]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        img1 = self.transform1(img)
        img2 = self.transform2(img)
        stage_id = self.stage_ids[idx]
        return img1, img2, stage_id, idx

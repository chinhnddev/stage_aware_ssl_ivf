"""
Custom dataset for IVF embryo images backed by metadata CSVs.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset


class IVFImageDataset(Dataset):
    """
    Dataset that loads images based on a metadata CSV.
    Expects columns: image_path, label_id (or unified_label), dataset_id, domain, split.
    """

    def __init__(
        self,
        csv_path: Path,
        root: Path,
        transform: Optional[Callable] = None,
        label_col: str = "label_id",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root = Path(root)
        self.transform = transform
        self.label_col = label_col

        df = pd.read_csv(self.csv_path)
        if label_col not in df.columns:
            if "unified_label" in df.columns:
                df = df.rename(columns={"unified_label": label_col})
            else:
                raise ValueError(f"Label column '{label_col}' not found in {csv_path}")

        # Build class mapping if labels are non-numeric
        if not pd.api.types.is_integer_dtype(df[label_col]):
            classes = sorted(df[label_col].unique())
            self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
            df[label_col] = df[label_col].map(self.class_to_idx)
        else:
            self.class_to_idx = {}

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = Path(row["image_path"])
        img_path = self.root / rel_path
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row[self.label_col])
        return img, label


def load_metadata(csv_paths: List[Path]) -> pd.DataFrame:
    """Load and concatenate multiple metadata CSVs."""
    dfs = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(dfs, ignore_index=True)

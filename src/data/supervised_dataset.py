"""
Supervised dataset for embryo quality classification.
"""

from __future__ import annotations

import os
import warnings
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
        use_role: Optional[str] = None,
        transform=None,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.root_map = {k: Path(v) for k, v in root_map.items()} if root_map else None
        self.transform = transform

        df = pd.read_csv(csv_path)
        if use_role:
            if "role" not in df.columns:
                warnings.warn(f"use_role='{use_role}' requested but 'role' column missing in {csv_path}", stacklevel=2)
            else:
                df = df[df["role"] == use_role].reset_index(drop=True)
        if use_domain:
            df = df[df["domain"] == use_domain].reset_index(drop=True)
        if split:
            df = df[df["split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"No samples for split={split} domain={use_domain} role={use_role} in {csv_path}")

        def resolve(image_path: str, domain: Optional[str]) -> Optional[Path]:
            if os.path.isabs(image_path):
                p = Path(image_path)
                return p if p.exists() else None
            if self.root_map is not None:
                if domain in self.root_map:
                    base = self.root_map[domain]
                else:
                    domain_lower = domain.lower() if isinstance(domain, str) else ""
                    alias_map = {
                        "hv_clinical": ["hospital"],
                        "hospital": ["hv_clinical"],
                        "hv_kaggle": ["kaggle"],
                        "kaggle": ["hv_kaggle"],
                        "roboflow_ssl": ["public"],
                        "public": ["roboflow_ssl"],
                    }
                    alt = next((self.root_map[a] for a in alias_map.get(domain_lower, []) if a in self.root_map), None)
                    if alt is None:
                        raise KeyError(f"domain '{domain}' not in root_map keys={list(self.root_map.keys())}")
                    base = alt
            elif self.root_dir is not None:
                base = self.root_dir
            else:
                raise ValueError("Either root_dir or root_map must be provided.")

            rel = Path(image_path)
            candidate = base / rel
            if candidate.exists():
                return candidate
            # lowercase fallback
            lower_rel = Path(*[p.lower() for p in rel.parts])
            candidate2 = base / lower_rel
            if candidate2.exists():
                return candidate2
            # insert alldata if missing (EDx/1/0.png -> EDx/alldata/1/0.png)
            parts = list(lower_rel.parts)
            if parts and parts[0].startswith("ed") and (len(parts) < 2 or parts[1] != "alldata"):
                rel_with_alldata = Path(parts[0], "alldata", *parts[1:])
                candidate3 = base / rel_with_alldata
                if candidate3.exists():
                    return candidate3
            return None

        df["abs_path"] = df.apply(lambda r: resolve(r["image_path"], r.get("domain")), axis=1)
        before = len(df)
        df = df[df["abs_path"].notna()].reset_index(drop=True)
        skipped = before - len(df)
        if skipped:
            print(f"[IVFClassifDataset] Skipped {skipped} missing files; remaining {len(df)} samples.")
        if len(df) == 0:
            raise ValueError(f"No samples remain after filtering missing files in {csv_path}")
        label_col = None
        for col in ["quality_label", "label_id", "unified_label", "label"]:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            raise KeyError("No label column found. Expected one of: quality_label, label_id, unified_label, label")
        labels_series = pd.to_numeric(df[label_col], errors="coerce")
        missing_labels = labels_series.isna()
        if missing_labels.any():
            warnings.warn(
                f"Dropping {missing_labels.sum()} rows without labels from {csv_path}; check role/domain filtering.",
                stacklevel=2,
            )
            df = df[~missing_labels].reset_index(drop=True)
            labels_series = labels_series[~missing_labels].reset_index(drop=True)
        # ED3: folder 1 = blastocyst (should be positive), folder 2 = non-blastocyst (negative).
        # Invert to ensure blastocyst is mapped to 1, non-blastocyst to 0 even if raw labels are 0/1 or 1/2.
        if "dataset_id" in df.columns:
            mask_ed3 = df["dataset_id"].astype(str).str.upper() == "ED3"
            if mask_ed3.any():
                uniq = sorted(labels_series[mask_ed3].unique())
                if len(uniq) == 2:
                    mapping = {uniq[0]: 1.0, uniq[1]: 0.0}
                    labels_series.loc[mask_ed3] = labels_series.loc[mask_ed3].map(mapping)
                else:
                    warnings.warn(
                        f"ED3 detected but found {len(uniq)} unique labels ({uniq}); cannot invert reliably.",
                        stacklevel=2,
                    )
        # If multi-class, binarize: everything > min label is positive
        if labels_series.nunique() > 2:
            min_label = labels_series.min()
            labels_series = (labels_series > min_label).astype(float)
        self.labels = labels_series.astype(float).tolist()
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

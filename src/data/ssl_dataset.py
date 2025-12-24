"""
Stage-aware self-supervised dataset for IVF embryo images.
"""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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
        transform1: Callable,
        transform2: Callable,
        root_dir: Optional[Path] = None,
        root_map: Optional[Dict[str, Union[str, Path]]] = None,
        use_domains: Optional[List[str]] = None,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.root_map = {k: Path(v) for k, v in root_map.items()} if root_map else None
        self.use_domains = use_domains
        dfs = [pd.read_csv(p) for p in csv_paths]
        df = pd.concat(dfs, ignore_index=True)
        if self.use_domains is not None and "domain" in df.columns:
            df = df[df["domain"].isin(self.use_domains)].reset_index(drop=True)

        # Resolve full paths per-domain if root_map is provided; fallback to single root_dir.
        def _resolve_image_path(image_path: str, domain: Optional[str]) -> Path:
            if os.path.isabs(str(image_path)):
                return Path(image_path)
            if self.root_map is not None:
                if domain not in self.root_map:
                    raise KeyError(f"domain '{domain}' not found in root_map keys={list(self.root_map.keys())}")
                base_path = self.root_map[domain] / str(image_path)
            elif self.root_dir is not None:
                base_path = self.root_dir / str(image_path)
            else:
                raise ValueError("Either root_dir or root_map must be provided.")

            # Try case-insensitive fallback on the relative path only (e.g., ED1 -> ed1) for hospital.
            if not base_path.exists() and isinstance(domain, str) and domain.lower() == "hospital":
                rel_parts = [part.lower() for part in Path(image_path).parts]
                # Some rows may miss "alldata" in the path; insert if absent.
                if rel_parts and rel_parts[0].startswith("ed") and (len(rel_parts) == 2 or rel_parts[1] != "alldata"):
                    rel_with_alldata = [rel_parts[0], "alldata"] + rel_parts[1:]
                    candidate = (self.root_map[domain] if self.root_map else Path(".")) / Path(*rel_with_alldata)
                    if candidate.exists():
                        return candidate
                rel_lower = Path(*rel_parts)
                lower_path = self.root_map[domain] / rel_lower if self.root_map else base_path
                if lower_path.exists():
                    return lower_path
            return base_path

        df["abs_path"] = df.apply(
            lambda r: _resolve_image_path(r["image_path"], r["domain"] if "domain" in r else None), axis=1
        )

        # Drop rows whose image files are missing to avoid worker FileNotFoundError.
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

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
from torch.utils.data import Dataset, get_worker_info
import torch


def infer_day_from_path(path: str) -> Optional[int]:
    path_lower = path.lower()
    if "day3" in path_lower or "d3" in path_lower:
        return 3
    if "day5" in path_lower or "d5" in path_lower:
        return 5
    return None


def stage_group_from_raw(stage_raw: Optional[str]) -> Optional[str]:
    if not stage_raw:
        return None
    sr = str(stage_raw).upper()
    if sr.startswith("ED"):
        return "ED"
    if sr.startswith("D"):
        return "DAY"
    return None


def build_stage_id(stage_raw: Optional[str], stage_group: Optional[str], day: Optional[int], image_path: str) -> str:
    """
    Build a stage identifier without forcing D3/D5 -> ED mapping.
    Priority: stage_raw -> (day + stage_group) -> day -> stage_group -> filename-derived day -> unknown.
    """
    stage_raw_norm = str(stage_raw).upper() if stage_raw else None
    stage_group_norm = str(stage_group).upper() if stage_group else stage_group_from_raw(stage_raw_norm)
    day_val: Optional[int] = None
    if day is not None and not pd.isna(day):
        try:
            day_val = int(day)
        except Exception:
            day_val = None
    if day_val is None and stage_raw_norm:
        m = re.match(r"[Dd](\d+)", stage_raw_norm)
        if m:
            day_val = int(m.group(1))
    if day_val is None:
        inferred = infer_day_from_path(image_path)
        if inferred is not None:
            day_val = inferred

    if stage_raw_norm:
        return stage_raw_norm
    if day_val is not None and stage_group_norm:
        return f"D{day_val}_{stage_group_norm}"
    if day_val is not None:
        return f"D{day_val}"
    if stage_group_norm:
        return stage_group_norm
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
        use_roles: Optional[List[str]] = None,
        num_stage_positives: int = 1,
    ) -> None:
        self.root_dir = Path(root_dir) if root_dir else None
        self.root_map = {k: Path(v) for k, v in root_map.items()} if root_map else None
        self.use_domains = use_domains
        self.use_roles = use_roles
        self.num_stage_positives = num_stage_positives
        dfs = [pd.read_csv(p) for p in csv_paths]
        df = pd.concat(dfs, ignore_index=True)
        if self.use_domains is not None and "domain" in df.columns:
            df = df[df["domain"].isin(self.use_domains)].reset_index(drop=True)
        if self.use_roles is not None:
            if "role" not in df.columns:
                raise KeyError("use_roles provided but 'role' column missing from metadata.")
            df = df[df["role"].isin(self.use_roles)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No samples found after filtering by domain/role.")

        # Resolve full paths per-domain if root_map is provided; fallback to single root_dir.
        def _resolve_image_path(image_path: str, domain: Optional[str]) -> Path:
            def _lookup_base(domain_key: Optional[str]) -> Path:
                if self.root_map is None:
                    if self.root_dir is None:
                        raise ValueError("Either root_dir or root_map must be provided.")
                    return self.root_dir
                if domain_key in self.root_map:
                    return self.root_map[domain_key]  # type: ignore[index]
                if isinstance(domain_key, str):
                    dk = domain_key.lower()
                    alias_map = {
                        "hv_clinical": ["hospital"],
                        "hospital": ["hv_clinical"],
                        "roboflow_ssl": ["public"],
                        "public": ["roboflow_ssl"],
                        "hv_kaggle": ["kaggle"],
                        "kaggle": ["hv_kaggle"],
                    }
                    for alias in alias_map.get(dk, []):
                        if alias in self.root_map:
                            return self.root_map[alias]  # type: ignore[index]
                raise KeyError(f"domain '{domain_key}' not found in root_map keys={list(self.root_map.keys())}")

            if os.path.isabs(str(image_path)):
                return Path(image_path)
            if self.root_map is not None:
                base_path = _lookup_base(domain) / str(image_path)
            elif self.root_dir is not None:
                base_path = self.root_dir / str(image_path)
            else:
                raise ValueError("Either root_dir or root_map must be provided.")

            # Try case-insensitive fallback on the relative path only (e.g., ED1 -> ed1) for hospital.
            if not base_path.exists() and isinstance(domain, str) and domain.lower() in {"hospital", "hv_clinical"}:
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
        stage_ids: List[str] = []
        for _, row in df.iterrows():
            stage_raw = row["stage_raw"] if "stage_raw" in df.columns else None
            stage_group = row["stage_group"] if "stage_group" in df.columns else None
            day = row["day"] if "day" in df.columns else None
            stage_id = build_stage_id(stage_raw, stage_group, day, str(row["image_path"]))
            stage_ids.append(stage_id)
        self.stage_ids = stage_ids

        self.transform1 = transform1
        self.transform2 = transform2
        self.abs_paths: List[Path] = df["abs_path"].tolist()

        # Build stage -> indices map for potential use
        self.stage_to_indices: Dict[str, List[int]] = {}
        for idx, sid in enumerate(self.stage_ids):
            self.stage_to_indices.setdefault(sid, []).append(idx)

        # Quick stat: fraction of samples with at least 2 per stage
        two_plus = sum(len(v) for v in self.stage_to_indices.values() if len(v) >= 2)
        if len(self.stage_ids) > 0:
            pct_two_plus = 100.0 * two_plus / len(self.stage_ids)
            print(f"[IVFSSLDataset] {pct_two_plus:.1f}% samples have >=2 per stage.")

    def __len__(self) -> int:
        return len(self.df)

    def sample_stage_positives(self, idx: int, k: Optional[int] = None) -> List[int]:
        """Sample up to k other indices with the same stage as idx, deterministic per worker."""
        k = k if k is not None else self.num_stage_positives
        stage_id = self.stage_ids[idx]
        candidates = [j for j in self.stage_to_indices.get(stage_id, []) if j != idx]
        if not candidates or k <= 0:
            return []
        if len(candidates) <= k:
            return candidates
        info = get_worker_info()
        seed = (info.seed if info else torch.initial_seed()) + idx
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(candidates), generator=g)[:k].tolist()
        return [candidates[i] for i in perm]

    def get_view(self, idx: int, view: int = 1):
        """Load an image and apply the same SSL transform pipeline."""
        img_path = self.abs_paths[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        if view == 2:
            return self.transform2(img)
        return self.transform1(img)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.df.iloc[idx]
        img_path = row["abs_path"]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        img1 = self.transform1(img)
        img2 = self.transform2(img)
        stage_id = self.stage_ids[idx]
        pos_indices = self.sample_stage_positives(idx, self.num_stage_positives)
        return img1, img2, stage_id, pos_indices, idx

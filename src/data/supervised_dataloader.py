"""
DataLoader builder for supervised finetuning, including optional stratified split creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.supervised_dataset import IVFClassifDataset


def _make_transforms(img_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.05)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def _ensure_split(cfg_data: Dict) -> Path:
    """
    Ensure CSV has split column; if missing, create stratified split within the requested domain.
    If group id exists (patient_id / case_id / series_id), use group-aware split.
    """
    csv_path = Path(cfg_data["csv_path"])
    df = pd.read_csv(csv_path)
    if "split" in df.columns and df["split"].notna().any():
        return csv_path

    use_domain = cfg_data.get("use_domain")
    if use_domain:
        df = df[df["domain"] == use_domain].reset_index(drop=True)
    labels = df["unified_label"].astype(int)
    seed = cfg_data.get("split", {}).get("seed", 42)
    test_size = cfg_data.get("split", {}).get("test_size", 0.15)
    val_size = cfg_data.get("split", {}).get("val_size", 0.15)

    group_cols = [c for c in ["patient_id", "case_id", "series_id"] if c in df.columns]
    groups = df[group_cols[0]] if group_cols else None

    if groups is not None:
        # Group-aware split: first split train/test, then train/val on remaining
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(df, groups=groups))
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        labels_train = df_train["unified_label"].astype(int)
        groups_train = df_train[group_cols[0]]
        val_size_adj = val_size / (1 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adj, random_state=seed)
        train_idx2, val_idx = next(gss_val.split(df_train, groups=groups_train))
        df_train, df_val = df_train.iloc[train_idx2].reset_index(drop=True), df_train.iloc[val_idx].reset_index(drop=True)
    else:
        df_train, df_test = train_test_split(df, test_size=test_size, stratify=labels, random_state=seed)
        labels_train = df_train["unified_label"].astype(int)
        val_size_adj = val_size / (1 - test_size)
        df_train, df_val = train_test_split(
            df_train, test_size=val_size_adj, stratify=labels_train, random_state=seed
        )

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    out_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    out_path = csv_path.parent / f"{csv_path.stem}_with_split.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[split] Created split at {out_path}")
    return out_path


def create_supervised_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    csv_with_split = _ensure_split(data_cfg)
    img_size = data_cfg["img_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    use_domain = data_cfg.get("use_domain")
    root_map = data_cfg.get("root_map")
    root_dir = Path(data_cfg["root_dir"]) if data_cfg.get("root_dir") else None

    train_ds = IVFClassifDataset(
        csv_path=csv_with_split,
        split="train",
        root_dir=root_dir,
        root_map=root_map,
        use_domain=use_domain,
        transform=_make_transforms(img_size, train=True),
    )
    val_ds = IVFClassifDataset(
        csv_path=csv_with_split,
        split="val",
        root_dir=root_dir,
        root_map=root_map,
        use_domain=use_domain,
        transform=_make_transforms(img_size, train=False),
    )
    test_ds = IVFClassifDataset(
        csv_path=csv_with_split,
        split="test",
        root_dir=root_dir,
        root_map=root_map,
        use_domain=use_domain,
        transform=_make_transforms(img_size, train=False),
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

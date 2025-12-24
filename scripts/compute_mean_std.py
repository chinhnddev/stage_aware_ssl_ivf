"""
Compute mean/std from the PUBLIC train split only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Ensure local src/ is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.transforms import build_eval_transforms


def compute_mean_std(csv_path: Path, root: Path, split: str = "train") -> tuple[list[float], list[float], int]:
    df = pd.read_csv(csv_path)
    if "split" in df.columns and split:
        df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"No rows found for split='{split}' in {csv_path}")

    mean = torch.zeros(3)
    std = torch.zeros(3)
    count = 0
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing mean/std"):
        img_path = root / row["image_path"]
        if not img_path.is_file():
            missing += 1
            continue
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        tensor = build_eval_transforms(mean=[0, 0, 0], std=[1, 1, 1])(img)
        # flatten spatial dims
        pixels = tensor.view(3, -1)
        mean += pixels.mean(dim=1)
        std += pixels.std(dim=1)
        count += 1

    if count == 0:
        raise ValueError("No images processed; check paths/split.")

    mean /= count
    std /= count
    return mean.tolist(), std.tolist(), missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute mean/std from public train split.")
    parser.add_argument("--csv", type=Path, required=True, help="metadata_public.csv (train split filtered).")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root for public images.")
    parser.add_argument("--split", type=str, default="train", help="Split to use (default: train).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mean, std, missing = compute_mean_std(args.csv, args.root, split=args.split)
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    if missing:
        print(f"Skipped {missing} missing files.")


if __name__ == "__main__":
    main()

"""
Sanity-check dataloaders: print batch shapes and label stats.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

# Ensure local src/ is importable when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataloader import get_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check IVF dataloaders.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root (common root for image paths).")
    parser.add_argument("--public_csv", type=Path, help="metadata_public.csv path.")
    parser.add_argument("--hospital_csvs", nargs="*", type=Path, help="Hospital metadata CSVs.")
    parser.add_argument("--mean", nargs=3, type=float, required=True, help="Normalization mean.")
    parser.add_argument("--std", nargs=3, type=float, required=True, help="Normalization std.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = {"train": [], "val": [], "test": []}

    if args.public_csv:
        import pandas as pd

        df = pd.read_csv(args.public_csv)
        for split in ["train", "val", "test"]:
            subset_path = args.public_csv.parent / f"tmp_{split}.csv"
            df[df["split"] == split].to_csv(subset_path, index=False)
            csv_paths[split].append(subset_path)

    if args.hospital_csvs:
        for p in args.hospital_csvs:
            csv_paths["test"].append(p)  # hospital used for eval-only

    loaders = get_dataloaders(
        csv_paths=csv_paths,
        root=args.root,
        mean=args.mean,
        std=args.std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for split, loader in loaders.items():
        print(f"\n{split.upper()} loader:")
        print(f"  batches: {len(loader)}")
        label_counter = Counter()
        for i, (images, labels) in enumerate(loader):
            print(f"  batch {i}: images {tuple(images.shape)}, labels {labels.shape}")
            label_counter.update(labels.tolist())
            if i >= 1:
                break
        print("  label distribution (first batches):", dict(label_counter))


if __name__ == "__main__":
    main()

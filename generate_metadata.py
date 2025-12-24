"""
Metadata curation script for the IVF embryo image dataset exported from Roboflow.

Usage example:
    python curate_public.py --dataset_root "D:/Master/IVF_project/dataset_1" --out_csv "metadata_public.csv"

Outputs a CSV with columns:
    image_path, raw_label, unified_label, domain, split

Expected console summary (example only):
    Total images: 13285
    Images per unified_label:
      arrested: 1636
      blastocyst: 12433
      ...
    Images per split:
      train: 10866
      val: 1204
      test: 1215
    Split x unified_label table:
      split\\label, arrested, blastocyst, early, morula
      test, 150, 900, 90, 75
      ...
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


# Supported image extensions (case-insensitive).
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

# Map folder names to split values used in the CSV.
SPLIT_ALIASES: Dict[str, str] = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "test": "test",
}


def map_unified_label(raw_label: str) -> str:
    """Apply the unified label mapping rules."""
    if re.fullmatch(r"\d+-\d+-\d+", raw_label):
        return "blastocyst"
    if raw_label == "Early":
        return "early"
    if raw_label == "Morula":
        return "morula"
    if raw_label == "Arrested":
        return "arrested"
    return raw_label


def iter_images(root: Path) -> List[Tuple[str, str, str, str, str]]:
    """
    Collect metadata rows for all images under the known split folders.

    Returns a list of tuples: (image_path, raw_label, unified_label, domain, split).
    """
    rows: List[Tuple[str, str, str, str, str]] = []
    for split_dir_name, split_value in SPLIT_ALIASES.items():
        split_dir = root / split_dir_name
        if not split_dir.is_dir():
            continue

        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            raw_label = label_dir.name
            unified_label = map_unified_label(raw_label)

            # Deterministic traversal of files.
            for path in sorted(label_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                rel_path = path.relative_to(root).as_posix()  # forward slashes
                rows.append((rel_path, raw_label, unified_label, "public", split_value))

    # Final deterministic ordering by path.
    rows.sort(key=lambda r: r[0])
    return rows


def write_metadata(rows: Sequence[Tuple[str, str, str, str, str]], output_path: Path) -> None:
    """Write the metadata rows to a CSV file."""
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "raw_label", "unified_label", "domain", "split"])
        writer.writerows(rows)


def print_stats(rows: Sequence[Tuple[str, str, str, str, str]]) -> None:
    """Print dataset statistics derived from the metadata rows."""
    total = len(rows)
    label_counts = Counter(row[2] for row in rows)
    split_counts = Counter(row[4] for row in rows)

    print(f"Total images: {total}")
    print("Images per unified_label:")
    for label in sorted(label_counts):
        print(f"  {label}: {label_counts[label]}")

    print("Images per split:")
    for split in sorted(split_counts):
        print(f"  {split}: {split_counts[split]}")

    # Split x unified_label table.
    table: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, _, unified_label, _, split in rows:
        table[split][unified_label] += 1

    unified_labels = sorted({row[2] for row in rows})
    print("Split x unified_label table:")
    header = ["split\\label"] + unified_labels
    print(", ".join(header))
    for split in sorted(table):
        counts = [str(table[split].get(label, 0)) for label in unified_labels]
        print(", ".join([split] + counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate IVF embryo public metadata CSV.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing train/ val/ test/ folders.",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path(__file__).resolve().parent / "metadata_public.csv",
        help="Output CSV path (default: metadata_public.csv in project root).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    output_csv: Path = args.out_csv

    rows = iter_images(dataset_root)
    write_metadata(rows, output_csv)
    print_stats(rows)
    print(f"\nWrote metadata to: {output_csv}")


if __name__ == "__main__":
    main()

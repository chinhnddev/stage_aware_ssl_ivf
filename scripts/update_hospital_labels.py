"""
Update hospital metadata to include ground-truth labels from target files (ED1–ED4).

Target file format (per line):
    data/embryo/edX/alldata/<folder>/<filename> <label_id>

Output CSV schema (order):
    image_path,label_id,raw_label,domain,dataset_id,split

Usage example:
    python scripts/update_hospital_labels.py \
        --hospital_root "D:/data/hospital_raw" \
        --targets_dir "D:/data/targets" \
        --out_dir "D:/data/metadata"
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATASETS = ["ED1", "ED2", "ED3", "ED4"]


def find_target_file(dataset_id: str, targets_dir: Path) -> Optional[Path]:
    """Find target file in common layouts."""
    candidates = [
        targets_dir / f"{dataset_id.lower()}_target.txt",
        targets_dir / dataset_id.lower() / f"{dataset_id.lower()}_target.txt",
        targets_dir / dataset_id / f"{dataset_id.lower()}_target.txt",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def parse_target_line(line: str) -> Optional[Tuple[str, int]]:
    parts = line.strip().split()
    if len(parts) != 2:
        return None
    path_str, label_str = parts
    try:
        label_id = int(label_str)
    except ValueError:
        return None
    return path_str, label_id


def build_index(dataset_root: Path) -> Dict[str, Path]:
    """Map relative posix path (from dataset_root) to file Path for fast lookup."""
    index: Dict[str, Path] = {}
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_root).as_posix()
        index[rel] = path
    return index


def resolve_rel_path(raw_path: str, index: Dict[str, Path]) -> Optional[str]:
    """
    Resolve a raw path from target to a relative posix path in index.
    Strategy:
        - If "alldata" exists in raw_path, use subpath starting from alldata/.
          If missing in index, also try dropping the "alldata/" prefix (in case data was extracted without that level).
        - Else, try full raw_path.
    """
    parts = Path(raw_path).as_posix().split("/")
    if "alldata" in parts:
        idx = parts.index("alldata")
        candidate = "/".join(parts[idx:])
        if candidate in index:
            return candidate
        # try without the alldata prefix
        candidate2 = "/".join(parts[idx + 1 :])
        if candidate2 in index:
            return candidate2
    if raw_path in index:
        return raw_path
    return None


def curate_dataset(dataset_id: str, hospital_root: Path, targets_dir: Path) -> List[Tuple[str, int, str, str, str, str]]:
    ds_root = hospital_root / dataset_id
    target_file = find_target_file(dataset_id, targets_dir)
    if target_file is None:
        print(f"Warning: missing target file for {dataset_id}")
        return []

    index = build_index(ds_root)
    rows: List[Tuple[str, int, str, str, str, str]] = []
    missing = 0

    with target_file.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_target_line(line)
            if parsed is None:
                continue
            raw_path, label_id = parsed
            rel = resolve_rel_path(raw_path, index)
            if rel is None:
                missing += 1
                continue

            file_path = index[rel]
            raw_label = file_path.parent.name
            image_path = (ds_root / rel).relative_to(hospital_root).as_posix()
            rows.append((image_path, label_id, raw_label, "hospital", dataset_id, "test"))

    rows.sort(key=lambda r: r[0])
    print(f"{dataset_id}: rows={len(rows)} missing={missing}")
    label_counts = Counter(r[1] for r in rows)
    if label_counts:
        print("  label_id counts:")
        for lid in sorted(label_counts):
            print(f"    {lid}: {label_counts[lid]}")
    return rows


def write_csv(rows: List[Tuple[str, int, str, str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label_id", "raw_label", "domain", "dataset_id", "split"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update hospital metadata with ground-truth labels (ED1–ED4).")
    parser.add_argument("--hospital_root", type=Path, required=True, help="Root containing ED1/ED2/ED3/ED4 folders.")
    parser.add_argument("--targets_dir", type=Path, required=True, help="Directory containing ed*_target.txt files.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for metadata CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hospital_root: Path = args.hospital_root
    targets_dir: Path = args.targets_dir
    out_dir: Path = args.out_dir

    for ds in DATASETS:
        rows = curate_dataset(ds, hospital_root, targets_dir)
        out_csv = out_dir / f"metadata_hospital_{ds.lower()}.csv"
        write_csv(rows, out_csv)
        print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()

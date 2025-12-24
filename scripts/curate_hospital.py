"""
Curate metadata for Hung Vuong Hospital embryo image datasets (ED1–ED4).

Steps:
1) Extract ed1.zip–ed4.zip into a consistent layout under --out_root
   (data/hospital_raw/ED1, ED2, ED3, ED4 by default).
2) Scan extracted images, write metadata_hospital.csv, and print summary stats.

Usage:
    python scripts/curate_hospital.py \
        --zips_dir "D:/.../datasets_HungVuong_Hospital/Human embryo image datasets" \
        --out_root "D:/.../data/hospital_raw" \
        --out_csv "D:/.../metadata_hospital.csv"

Notes:
- Only processes ed1.zip–ed4.zip (embryo). Does not touch sperm or malaria zips.
- No image preprocessing/augmentation is performed.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DATASET_IDS = ["ED1", "ED2", "ED3", "ED4"]


def extract_zip(zip_path: Path, dest_dir: Path) -> int:
    """
    Extract a zip file into dest_dir.

    - Handles a single top-level folder inside the zip by stripping it.
    - Returns number of files extracted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Files only, deterministic order.
        file_members = sorted([m for m in zf.namelist() if not m.endswith("/")])
        if not file_members:
            return 0

        first_parts = {Path(m).parts[0] for m in file_members if Path(m).parts}
        strip_first = len(first_parts) == 1

        for member in file_members:
            src_path = Path(member)
            parts = src_path.parts
            if strip_first and len(parts) > 1:
                parts = parts[1:]
            if not parts:
                continue

            out_path = dest_dir.joinpath(*parts)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1

    return count


def copy_existing_dataset(source_dir: Path, dest_dir: Path) -> int:
    """
    Copy an already-extracted dataset into dest_dir.

    - If the source has a single top-level directory, strip it (e.g., alldata/).
    - Returns number of files copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Determine actual data root (strip single top-level dir).
    entries = [p for p in source_dir.iterdir()]
    data_root = source_dir
    if len(entries) == 1 and entries[0].is_dir():
        data_root = entries[0]

    files = sorted([p for p in data_root.rglob("*") if p.is_file()])
    count = 0
    for file_path in files:
        rel_parts = file_path.relative_to(data_root)
        out_path = dest_dir.joinpath(rel_parts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, out_path)
        count += 1
    return count


def map_unified_label(raw_label: str) -> str:
    """No mapping for now; keep raw_label."""
    return raw_label


def iter_images(dataset_root: Path) -> List[Tuple[str, str, str, str, str, str]]:
    """
    Scan extracted datasets and return metadata rows.

    Returns tuples:
        (image_path, raw_label, unified_label, domain, dataset_id, split)
    """
    rows: List[Tuple[str, str, str, str, str, str]] = []
    for dataset_id in DATASET_IDS:
        ds_dir = dataset_root / dataset_id
        if not ds_dir.is_dir():
            continue

        for path in sorted(ds_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            raw_label = path.parent.name
            unified_label = map_unified_label(raw_label)
            image_path = path.relative_to(dataset_root).as_posix()
            rows.append((image_path, raw_label, unified_label, "hospital", dataset_id, "test"))

    rows.sort(key=lambda r: r[0])
    return rows


def write_metadata(rows: Sequence[Tuple[str, str, str, str, str, str]], output_path: Path) -> None:
    """Write metadata to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "raw_label", "unified_label", "domain", "dataset_id", "split"])
        writer.writerows(rows)


def print_stats(rows: Sequence[Tuple[str, str, str, str, str, str]]) -> None:
    """Print dataset statistics."""
    total = len(rows)
    dataset_counts = Counter(r[4] for r in rows)
    label_counts = Counter(r[1] for r in rows)  # raw_label == unified_label

    print(f"Total images: {total}")
    print("Images per dataset_id:")
    for ds in sorted(dataset_counts):
        print(f"  {ds}: {dataset_counts[ds]}")

    print("Images per raw_label (unified_label):")
    for lbl in sorted(label_counts):
        print(f"  {lbl}: {label_counts[lbl]}")

    # 2D table: dataset_id x raw_label
    table: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, raw_label, _, _, dataset_id, _ in rows:
        table[dataset_id][raw_label] += 1

    all_labels = sorted({r[1] for r in rows})
    header = ["dataset_id\\label"] + all_labels
    print("Dataset_id x raw_label table:")
    print(", ".join(header))
    for ds in sorted(table):
        counts = [str(table[ds].get(lbl, 0)) for lbl in all_labels]
        print(", ".join([ds] + counts))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate Hung Vuong Hospital embryo datasets (ED1–ED4).")
    parser.add_argument(
        "--zips_dir",
        type=Path,
        required=True,
        help="Directory containing ed1.zip … ed4.zip (embryo datasets).",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "hospital_raw",
        help="Output root for extracted data (default: data/hospital_raw).",
    )
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "metadata_hospital.csv",
        help="Output metadata CSV path (default: project root/metadata_hospital.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zips_dir: Path = args.zips_dir
    out_root: Path = args.out_root
    out_csv: Path = args.out_csv

    # Extraction
    total_extracted = 0
    for ds in DATASET_IDS:
        zip_path = zips_dir / f"{ds.lower()}.zip"
        dest_dir = out_root / ds
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(f"Found existing extracted data in {dest_dir}, skipping extraction.")
            extracted = 0
        elif zip_path.is_file():
            extracted = extract_zip(zip_path, dest_dir)
            print(f"Extracted {extracted} files to {dest_dir}")
        else:
            # Fallback to already-extracted folder in zips_dir (ed1/ED1).
            fallback_dirs = [zips_dir / ds.lower(), zips_dir / ds]
            source = next((p for p in fallback_dirs if p.is_dir()), None)
            if source:
                extracted = copy_existing_dataset(source, dest_dir)
                print(f"Copied {extracted} files from {source} to {dest_dir}")
            else:
                extracted = 0
                print(f"Warning: missing zip and source folder for {ds} (looked for {zip_path}).")

        total_extracted += extracted

    # Metadata
    rows = iter_images(out_root)
    write_metadata(rows, out_csv)
    print_stats(rows)
    print(f"\nTotal files extracted (all zips): {total_extracted}")
    print(f"Wrote metadata to: {out_csv}")


if __name__ == "__main__":
    main()

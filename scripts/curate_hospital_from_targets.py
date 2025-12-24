"""
Curate hospital embryo datasets (ED1–ED4) using provided target files.

Requirements:
- Read targets from ed1_target.txt ... ed4_target.txt (ground truth).
- Resolve paths relative to hospital_root/EDx, starting from "alldata/" when present.
- Write per-dataset CSVs and a combined CSV for management.

CSV columns (order):
    image_path, label_id, raw_label, raw_label_name, domain, dataset_id, split

Usage example:
    python scripts/curate_hospital_from_targets.py ^
        --hospital_root "D:/Master/IVF_project/DATA_CURATION/data/hospital_raw" ^
        --targets_dir  "D:/Master/IVF_project/DATA_CURATION/datasets_HungVuong_Hospital/Human embryo image datasets" ^
        --out_dir      "D:/Master/IVF_project/DATA_CURATION/artifacts"
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATASETS = ["ED1", "ED2", "ED3", "ED4"]


def load_file_index(dataset_root: Path) -> Tuple[Dict[str, Path], Dict[str, List[str]]]:
    """
    Build:
    - exact_map: relative posix path -> absolute Path
    - name_map: basename -> sorted list of relative posix paths
    """
    exact_map: Dict[str, Path] = {}
    name_map: Dict[str, List[str]] = defaultdict(list)

    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel_posix = path.relative_to(dataset_root).as_posix()
        exact_map[rel_posix] = path
        name_map[path.name].append(rel_posix)

    for name in name_map:
        name_map[name].sort()

    return exact_map, name_map


def resolve_rel_path(raw_path: str, exact_map: Dict[str, Path], name_map: Dict[str, List[str]]) -> Optional[str]:
    """
    Resolve a target path to a relative posix path present in exact_map.
    Strategy:
        1) If "alldata" in raw_path parts, strip to alldata/... and try exact match.
        2) If direct match fails, try full raw_path tail after dataset dir.
        3) Fallback to basename lookup via name_map (pick first sorted for determinism).
    Returns the relative posix path, or None if not found.
    """
    parts = Path(raw_path).as_posix().split("/")
    rel_candidate: Optional[str] = None

    if "alldata" in parts:
        idx = parts.index("alldata")
        rel_candidate = "/".join(parts[idx:])
        if rel_candidate in exact_map:
            return rel_candidate

    # Try the full path as-is (without drive)
    if raw_path in exact_map:
        return raw_path

    # Fallback to basename search
    basename = Path(raw_path).name
    candidates = name_map.get(basename)
    if candidates:
        return candidates[0]  # deterministic pick

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


def curate_dataset(
    dataset_id: str,
    hospital_root: Path,
    targets_dir: Path,
    out_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Curate one dataset using its target file.
    Returns (metadata_df, missing_paths).
    """
    ds_root = hospital_root / dataset_id
    target_path = find_target_file(dataset_id, targets_dir)
    exact_map, name_map = load_file_index(ds_root)

    rows: List[Tuple[str, int, str, str, str, str, str]] = []
    missing: List[str] = []

    if target_path is None:
        print(f"Warning: target file not found for {dataset_id} in {targets_dir}")
        return pd.DataFrame(
            columns=["image_path", "label_id", "raw_label", "raw_label_name", "domain", "dataset_id", "split"]
        ), missing

    with target_path.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_target_line(line)
            if parsed is None:
                continue
            raw_path, label_id = parsed
            rel = resolve_rel_path(raw_path, exact_map, name_map)
            if rel is None:
                missing.append(raw_path)
                continue

            file_path = exact_map[rel]
            raw_label = file_path.parent.name if file_path.parent.name else ""
            rows.append(
                (
                    (ds_root / rel).relative_to(hospital_root).as_posix(),
                    label_id,
                    raw_label,
                    f"class_{label_id}",
                    "hospital",
                    dataset_id,
                    "test",
                )
            )

    rows.sort(key=lambda r: r[0])
    df = pd.DataFrame(
        rows,
        columns=["image_path", "label_id", "raw_label", "raw_label_name", "domain", "dataset_id", "split"],
    )

    if missing:
        missing_file = out_dir / f"missing_{dataset_id}.txt"
        missing_file.parent.mkdir(parents=True, exist_ok=True)
        missing_file.write_text("\n".join(missing), encoding="utf-8")
        print(f"{dataset_id}: missing files logged to {missing_file} (count={len(missing)})")
    else:
        print(f"{dataset_id}: no missing files.")

    return df, missing


def print_stats(dfs: Dict[str, pd.DataFrame]) -> None:
    """Print counts per dataset and dataset x label_id table."""
    print("\nPer-dataset counts:")
    for ds, df in dfs.items():
        label_counts = df["label_id"].value_counts().sort_index()
        print(f"{ds}: total={len(df)}")
        for lid, cnt in label_counts.items():
            print(f"  label_id {lid}: {cnt}")

    print("\nDataset_id x label_id table:")
    # Build table
    label_ids = sorted({int(lid) for df in dfs.values() for lid in df["label_id"].unique()})
    header = ["dataset_id"] + [str(lid) for lid in label_ids]
    print(", ".join(header))
    for ds in sorted(dfs.keys()):
        df = dfs[ds]
        counts = df["label_id"].value_counts().to_dict()
        row = [ds] + [str(counts.get(lid, 0)) for lid in label_ids]
        print(", ".join(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate hospital embryo datasets (ED1–ED4) using target files.")
    parser.add_argument(
        "--hospital_root",
        type=Path,
        required=True,
        help="Root directory containing ED1/ED2/ED3/ED4 folders (extracted).",
    )
    parser.add_argument(
        "--targets_dir",
        type=Path,
        required=True,
        help="Directory containing ed1_target.txt ... ed4_target.txt.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts",
        help="Output directory for metadata CSVs and missing logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hospital_root: Path = args.hospital_root
    targets_dir: Path = args.targets_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_ds_dfs: Dict[str, pd.DataFrame] = {}
    combined_rows: List[pd.DataFrame] = []
    all_missing: Dict[str, List[str]] = {}

    for ds in DATASETS:
        df, missing = curate_dataset(ds, hospital_root, targets_dir, out_dir)
        per_ds_dfs[ds] = df
        if not df.empty:
            combined_rows.append(df)
        all_missing[ds] = missing

        # Write per-dataset CSV
        out_csv = out_dir / f"metadata_hospital_{ds.lower()}.csv"
        df.to_csv(out_csv, index=False)
        print(f"Wrote {len(df)} rows to {out_csv}")

    # Combined CSV
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
    else:
        combined_df = pd.DataFrame(columns=["image_path", "label_id", "raw_label", "raw_label_name", "domain", "dataset_id", "split"])
    combined_df = combined_df.sort_values(by=["dataset_id", "image_path"]).reset_index(drop=True)
    combined_csv = out_dir / "metadata_hospital_all.csv"
    combined_df.to_csv(combined_csv, index=False)
    print(f"Wrote combined metadata to {combined_csv} (rows={len(combined_df)})")

    print_stats(per_ds_dfs)

    # Missing summary
    print("\nMissing file counts:")
    for ds in DATASETS:
        print(f"  {ds}: {len(all_missing.get(ds, []))}")


if __name__ == "__main__":
    main()

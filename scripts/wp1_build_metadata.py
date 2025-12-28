"""
Build unified WP1 metadata files for:
- SSL pretraining (roboflow_ssl)
- Supervised finetuning (hv_kaggle)
- Cross-domain evaluation (hv_clinical)
- Optional target-unlabeled alignment set (hv_clinical)

Outputs:
    metadata_ssl.csv
    metadata_supervised.csv
    metadata_crossdomain_test.csv
    metadata_target_unlabeled.csv (optional)
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {"val": "val", "valid": "val", "validation": "val", "train": "train", "test": "test"}
OUTPUT_COL_ORDER = [
    "image_path",
    "domain",
    "role",
    "stage_raw",
    "stage_group",
    "quality_label",
    "split",
    "dataset_id",
    "capture_system",
]


def infer_stage_raw_from_name(name: str) -> Optional[str]:
    """Infer D3/D5 or ED1-ED4 from filename prefix."""
    m = re.match(r"(D3|D5)", name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r"(ED[1-4])", name, re.IGNORECASE)
    if m:
        return m.group(1).upper()
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


def iter_image_rows(
    root: Path,
    domain: str,
    role: str,
    split_map: Optional[Dict[str, str]] = None,
    label_from_parent: bool = False,
) -> Tuple[List[Dict], List[str]]:
    """
    Generic iterator for folder-structured datasets (split/label/files.jpg).
    Returns (rows, missing_paths).
    """
    split_map = split_map or {}
    rows: List[Dict] = []
    missing: List[str] = []

    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        split_name = split_map.get(split_dir.name.lower(), split_dir.name.lower())
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            try:
                q_label = int(label_dir.name) if label_from_parent else None
            except ValueError:
                q_label = None
            for path in sorted(label_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rel_path = path.relative_to(root).as_posix()
                if not path.exists():
                    missing.append(rel_path)
                    continue
                stage_raw = infer_stage_raw_from_name(path.name)
                rows.append(
                    {
                        "image_path": rel_path,
                        "domain": domain,
                        "role": role,
                        "stage_raw": stage_raw,
                        "stage_group": stage_group_from_raw(stage_raw),
                        "quality_label": q_label,
                        "split": split_name,
                        "dataset_id": domain,
                    }
                )
    return rows, missing


def build_roboflow_ssl(root: Path) -> Tuple[pd.DataFrame, List[str]]:
    rows, missing = iter_image_rows(root, domain="roboflow_ssl", role="ssl", split_map=SPLIT_ALIASES)
    df = pd.DataFrame(rows)
    return df, missing


def build_kaggle_supervised(root: Path, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, List[str]]:
    rows, missing = iter_image_rows(
        root, domain="hv_kaggle", role="supervised", split_map=SPLIT_ALIASES, label_from_parent=True
    )
    if not rows:
        return pd.DataFrame(), missing
    df = pd.DataFrame(rows)
    # Ensure labels are 0/1 and stage_group populated
    if df["quality_label"].isna().any():
        raise ValueError("Found non-numeric label folders in Kaggle dataset; expected 0/1.")
    df["quality_label"] = df["quality_label"].astype(int)
    df["stage_group"] = df["stage_raw"].apply(stage_group_from_raw)

    if "val" not in df["split"].unique() and val_ratio > 0:
        train_df = df[df["split"] == "train"]
        if not train_df.empty:
            train_part, val_part = train_test_split(
                train_df,
                test_size=val_ratio,
                stratify=train_df["quality_label"],
                random_state=seed,
            )
            df.loc[train_part.index, "split"] = "train"
            df.loc[val_part.index, "split"] = "val"
        else:
            print("[kaggle] No train split found; skipping val creation.")
    return df, missing


def build_clinical_sets(
    metadata_csv: Path, target_unlabeled_ratio: float, seed: int
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    df = pd.read_csv(metadata_csv)
    if df.empty:
        raise ValueError(f"No rows in {metadata_csv}")
    # Normalize paths: insert alldata/ when missing for EDx datasets.
    def _normalize_path(p: str) -> str:
        parts = Path(p).parts
        if parts and parts[0].upper().startswith("ED") and (len(parts) < 2 or parts[1].lower() != "alldata"):
            parts = (parts[0], "alldata") + tuple(parts[1:])
        return Path(*parts).as_posix()

    df["image_path"] = df["image_path"].apply(_normalize_path)
    df["domain"] = "hv_clinical"
    df["role"] = "crossdomain_test"
    df["stage_raw"] = df["dataset_id"].astype(str).str.upper()
    df["stage_group"] = "ED"
    # Only keep binary labels as quality_label; others remain null to avoid forced mapping.
    df["quality_label"] = df["label_id"].where(df["label_id"].isin([0, 1]), pd.NA)
    if "split" not in df.columns or df["split"].isna().all():
        df["split"] = "holdout"
    # Optional target-unlabeled subset (no overlap with held-out test)
    target_df: Optional[pd.DataFrame] = None
    if target_unlabeled_ratio > 0:
        rng = random.Random(seed)
        sampled_indices: List[int] = []
        for _, group in df.groupby("dataset_id"):
            n = min(len(group), max(1, int(len(group) * target_unlabeled_ratio)))
            idx = group.sample(n=n, random_state=rng.randint(0, 1_000_000)).index.tolist()
            sampled_indices.extend(idx)
        target_df = df.loc[sampled_indices].copy()
        df = df.drop(index=sampled_indices).reset_index(drop=True)
        target_df = target_df.reset_index(drop=True)
        target_df["role"] = "target_unlabeled"
        target_df["quality_label"] = pd.NA
        target_df["split"] = "unlabeled"
    return df.reset_index(drop=True), target_df


def verify_paths(df: pd.DataFrame, root: Path) -> List[str]:
    missing: List[str] = []
    for p in df["image_path"].tolist():
        abs_path = root / p
        if not abs_path.exists():
            missing.append(p)
    return missing


def write_metadata(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in OUTPUT_COL_ORDER if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    df.to_csv(out_path, index=False, columns=cols)
    print(f"Wrote {len(df)} rows to {out_path}")


def summarize(
    all_df: pd.DataFrame, supervised_df: pd.DataFrame, cross_df: pd.DataFrame, roots: Dict[str, Path]
) -> None:
    print("\n[summary] counts by role, domain")
    print(all_df.groupby(["role", "domain"]).size().unstack(fill_value=0))

    print("\n[summary] counts by domain, stage_raw")
    print(all_df.groupby(["domain", "stage_raw"]).size().unstack(fill_value=0))

    print("\n[summary] supervised label distribution (by split)")
    if not supervised_df.empty:
        print(supervised_df.groupby(["split", "quality_label"]).size().unstack(fill_value=0))
    else:
        print("No supervised rows.")

    print("\n[summary] overlap check (absolute paths)")
    sup_paths = {(roots["hv_kaggle"] / p).resolve() for p in supervised_df["image_path"].tolist()}
    cross_paths = {(roots["hv_clinical"] / p).resolve() for p in cross_df["image_path"].tolist()}
    overlap = sup_paths.intersection(cross_paths)
    if overlap:
        print(f"WARNING: found {len(overlap)} overlapping absolute paths between supervised and crossdomain_test.")
    else:
        print("No overlap between supervised and crossdomain_test paths.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified WP1 metadata files.")
    parser.add_argument("--roboflow_root", type=Path, default=Path("data/embyro_public_dataset"))
    parser.add_argument("--kaggle_root", type=Path, default=Path("data/hv_hospital_day3_day5"))
    parser.add_argument(
        "--clinical_csv", type=Path, default=Path("data-curation/metadata_hospital_all.csv"), help="Source clinical CSV."
    )
    parser.add_argument(
        "--clinical_root",
        type=Path,
        default=Path("data/HungVuong_hospital_dataset/Human embryo image datasets"),
        help="Root for hv_clinical images (used for path verification).",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("data-curation"))
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Val split ratio for Kaggle train set if missing.")
    parser.add_argument("--target_unlabeled_ratio", type=float, default=0.0, help="Fraction of clinical samples for UDA.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ssl_df, missing_ssl = build_roboflow_ssl(args.roboflow_root)
    kaggle_df, missing_kaggle = build_kaggle_supervised(args.kaggle_root, args.val_ratio, args.seed)
    clinical_df, target_df = build_clinical_sets(args.clinical_csv, args.target_unlabeled_ratio, args.seed)

    all_missing = {
        "roboflow_ssl": verify_paths(ssl_df, args.roboflow_root) + missing_ssl,
        "hv_kaggle": verify_paths(kaggle_df, args.kaggle_root) + missing_kaggle,
        "hv_clinical": verify_paths(clinical_df, args.clinical_root),
    }
    if target_df is not None:
        all_missing["hv_clinical_unlabeled"] = verify_paths(target_df, args.clinical_root)
    for k, v in all_missing.items():
        if v:
            print(f"[missing] {k}: {len(v)}")
    ssl_out = args.out_dir / "metadata_ssl.csv"
    sup_out = args.out_dir / "metadata_supervised.csv"
    cross_out = args.out_dir / "metadata_crossdomain_test.csv"
    write_metadata(ssl_df, ssl_out)
    write_metadata(kaggle_df, sup_out)
    write_metadata(clinical_df, cross_out)
    if target_df is not None:
        target_out = args.out_dir / "metadata_target_unlabeled.csv"
        write_metadata(target_df, target_out)

    concat_parts = [df for df in [ssl_df, kaggle_df, clinical_df, target_df] if df is not None and not df.empty]
    all_df = pd.concat(concat_parts, ignore_index=True)
    summarize(all_df, kaggle_df, clinical_df, roots={"hv_kaggle": args.kaggle_root, "hv_clinical": args.clinical_root})


if __name__ == "__main__":
    main()

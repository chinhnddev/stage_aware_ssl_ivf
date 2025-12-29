"""
WP2 protocol validation utility.

Checks:
- supervised_csv domains == {"hv_kaggle"} only
- cross_csv domains == {"hv_clinical"} only
- label column exists and has no missing values
- splits are present (train/val/test for supervised; test/holdout for cross)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import pandas as pd


def _check_domains(df: pd.DataFrame, expected: Set[str], csv_path: Path, name: str) -> None:
    domains = set(df["domain"].unique())
    if domains != expected:
        raise ValueError(f"[{name}] domain mismatch in {csv_path}: found {domains}, expected {expected}")


def _check_labels(df: pd.DataFrame, label_col: str, csv_path: Path, name: str) -> None:
    if label_col not in df.columns:
        raise ValueError(f"[{name}] label_col '{label_col}' missing in {csv_path}")
    missing = df[label_col].isna().sum()
    if missing > 0:
        raise ValueError(f"[{name}] label_col '{label_col}' has {missing} missing values in {csv_path}")


def validate(supervised_csv: Path, cross_csv: Path, label_col: str) -> None:
    sup_df = pd.read_csv(supervised_csv)
    cross_df = pd.read_csv(cross_csv)

    _check_domains(sup_df, {"hv_kaggle"}, supervised_csv, "supervised")
    _check_domains(cross_df, {"hv_clinical"}, cross_csv, "crossdomain_test")

    _check_labels(sup_df, label_col, supervised_csv, "supervised")
    _check_labels(cross_df, label_col, cross_csv, "crossdomain_test")

    sup_splits = set(sup_df["split"].unique())
    # Allow train/val/test for in-domain evaluation; disallow other splits
    if not sup_splits.issubset({"train", "val", "test"}):
        raise ValueError(f"[supervised] splits must be subset of train/val/test in {supervised_csv}: found {sup_splits}")
    if "train" not in sup_splits or "val" not in sup_splits:
        raise ValueError(f"[supervised] missing train or val split in {supervised_csv}: found {sup_splits}")

    cross_splits = set(cross_df["split"].unique())
    if not cross_splits.issubset({"test"}):
        raise ValueError(f"[crossdomain_test] splits must be 'test' only in {cross_csv}: found {cross_splits}")

    print(f"[validate] supervised_csv={supervised_csv} rows={len(sup_df)} domains={set(sup_df['domain'])} splits={sup_splits}")
    print(f"[validate] cross_csv={cross_csv} rows={len(cross_df)} domains={set(cross_df['domain'])} splits={cross_splits}")
    print("[validate] WP2 protocol checks passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--supervised_csv", type=Path, required=True)
    ap.add_argument("--cross_csv", type=Path, required=True)
    ap.add_argument("--label_col", type=str, default="quality_label")
    args = ap.parse_args()
    validate(args.supervised_csv, args.cross_csv, args.label_col)

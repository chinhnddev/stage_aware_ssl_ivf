"""
Generate binary quality labels for the Hung Vuong clinical dataset based on folder rules:
- ED1, ED2, ED4 (5-category):
    folders 1,2 -> non-blastocyst (0)
    folders 3,4,5 -> blastocyst (1)
- ED3 (2-category):
    folder 1 -> blastocyst (1)
    folder 2 -> non-blastocyst (0)

Outputs:
- data-curation/metadata_hospital_all_with_quality.csv
- data-curation/metadata_crossdomain_test_quality.csv (only rows with quality_label present)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def map_quality(row: pd.Series):
    ds = str(row.get("dataset_id", "")).upper()
    raw = row.get("raw_label", row.get("label_id", None))
    try:
        raw_int = int(raw)
    except Exception:
        return pd.NA
    if ds == "ED3":
        if raw_int == 1:
            return 1
        if raw_int == 2:
            return 0
    if ds in {"ED1", "ED2", "ED4"}:
        if raw_int in {1, 2}:
            return 0
        if raw_int in {3, 4, 5}:
            return 1
    return pd.NA


def main(args):
    df = pd.read_csv(args.input_csv)
    df["quality_label"] = df.apply(map_quality, axis=1)

    out_all = args.input_csv.parent / "metadata_hospital_all_with_quality.csv"
    df.to_csv(out_all, index=False)
    print(f"Saved {out_all} rows={len(df)} labeled={df['quality_label'].notna().sum()}")

    df_labeled = df[df["quality_label"].notna()].reset_index(drop=True)
    out_test = args.input_csv.parent / "metadata_crossdomain_test_quality.csv"
    df_labeled.to_csv(out_test, index=False)
    print(f"Saved {out_test} rows={len(df_labeled)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary quality labels for clinical dataset.")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("data-curation/metadata_hospital_all.csv"),
        help="Input metadata_hospital_all.csv",
    )
    args = parser.parse_args()
    main(args)

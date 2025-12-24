"""
Utility: compare metrics.json across fine-tune runs.
Usage: python scripts/compare_metrics.py outputs/ft/run_fromscratch/metrics.json outputs/ft/run_ssl_vanilla/metrics.json outputs/ft/run1_ssl_stage/metrics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_metrics(path: Path):
    data = json.loads(path.read_text())
    test = data.get("test", {})
    return {
        "run": path.parent.name,
        "auroc": test.get("auroc", 0),
        "auprc": test.get("auprc", 0),
        "f1": test.get("f1", 0),
        "acc": test.get("acc", 0),
        "bal_acc": test.get("bal_acc", 0),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/compare_metrics.py <metrics.json> [<metrics.json> ...]")
        sys.exit(1)
    rows = [load_metrics(Path(p)) for p in sys.argv[1:]]
    header = ["Run", "AUROC", "AUPRC", "F1", "ACC", "Bal-ACC"]
    print("\t".join(header))
    for r in rows:
        print(
            f"{r['run']}\t{r['auroc']:.4f}\t{r['auprc']:.4f}\t{r['f1']:.4f}\t{r['acc']:.4f}\t{r['bal_acc']:.4f}"
        )


if __name__ == "__main__":
    main()

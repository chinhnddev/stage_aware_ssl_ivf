"""
Metrics utilities for binary classification.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn import metrics


def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict:
    y_true = y_true.astype(int)
    auroc = metrics.roc_auc_score(y_true, y_score)
    auprc = metrics.average_precision_score(y_true, y_score)
    y_pred = (y_score >= threshold).astype(int)
    f1 = metrics.f1_score(y_true, y_pred)
    acc = metrics.accuracy_score(y_true, y_pred)
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "acc": acc,
        "bal_acc": bal_acc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": threshold,
    }


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """Returns (best_threshold, best_f1) based on val set."""
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    idx = f1_scores.argmax()
    best_thresh = thresholds[idx] if idx < len(thresholds) else 0.5
    return float(best_thresh), float(f1_scores[idx])

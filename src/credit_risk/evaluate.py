"""Metrics for an imbalanced binary problem, threshold tuning and subgroup analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_fbeta_threshold(y_true, scores, beta: float) -> float:
    """Threshold maximising F-beta over the distinct candidate scores."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    candidates = np.unique(np.round(scores, 6))
    if len(candidates) <= 1:
        return 0.5
    best_t, best_f = 0.5, -1.0
    for t in candidates:
        f = fbeta_score(y_true, scores >= t, beta=beta, zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t


def classification_metrics(y_true, scores, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "average_precision": float(average_precision_score(y_true, scores)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, scores)) if both else float("nan"),
        "brier": float(brier_score_loss(y_true, scores)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if both else float("nan"),
        "accuracy": float(accuracy_score(y_true, pred)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def subgroup_metrics(
    groups: pd.Series, y_true, scores, threshold: float, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Per-group error rates with bootstrap 95% intervals for FNR and FPR."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups)
    rows = []
    for g in [*pd.unique(groups), "ALL"]:
        idx = np.arange(len(y_true)) if g == "ALL" else np.flatnonzero(groups == g)
        m = classification_metrics(y_true[idx], scores[idx], threshold)
        boots = {"false_negative_rate": [], "false_positive_rate": [], "error_rate": []}
        for _ in range(n_boot):
            b = rng.choice(idx, size=len(idx), replace=True)
            mb = classification_metrics(y_true[b], scores[b], threshold)
            boots["false_negative_rate"].append(mb["false_negative_rate"])
            boots["false_positive_rate"].append(mb["false_positive_rate"])
            boots["error_rate"].append(1 - mb["accuracy"])
        row = {
            "group": g,
            "n": m["n"],
            "positives": m["positives"],
            "base_rate": m["positives"] / m["n"] if m["n"] else float("nan"),
            "error_rate": 1 - m["accuracy"],
            "false_negative_rate": m["false_negative_rate"],
            "false_positive_rate": m["false_positive_rate"],
            "recall": m["recall"],
            "precision": m["precision"],
            "average_precision": m["average_precision"],
            "flagged_rate": (m["tp"] + m["fp"]) / m["n"] if m["n"] else float("nan"),
        }
        for k, vals in boots.items():
            arr = np.asarray(vals, dtype=float)
            arr = arr[~np.isnan(arr)]
            row[f"{k}_ci_low"] = float(np.percentile(arr, 2.5)) if len(arr) else float("nan")
            row[f"{k}_ci_high"] = float(np.percentile(arr, 97.5)) if len(arr) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)

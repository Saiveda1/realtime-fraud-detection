"""Evaluation metrics tuned for heavy class imbalance.

Under ~0.5% fraud, ROC-AUC is optimistic and accuracy is meaningless. The
headline metric here is **PR-AUC (average precision)** plus operating-point
metrics an on-call fraud team actually cares about: precision@k (alert budget),
recall/fraud-caught at that budget, and calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class OperatingPoint:
    k: int
    threshold: float
    precision: float
    recall: float
    n_fraud_caught: int
    n_fraud_total: int


@dataclass
class EvalResult:
    pr_auc: float
    roc_auc: float
    baseline_pr_auc: float
    lift_over_baseline: float
    op: OperatingPoint
    pr_curve: tuple[np.ndarray, np.ndarray]  # (recall, precision)
    calibration: tuple[np.ndarray, np.ndarray]  # (mean_pred, frac_pos)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> OperatingPoint:
    """Metrics if we alert on the top-``k`` highest-scoring events."""
    k = min(k, len(scores))
    order = np.argsort(-scores)
    top = order[:k]
    caught = int(y_true[top].sum())
    total_fraud = int(y_true.sum())
    thr = float(scores[order[k - 1]]) if k > 0 else 1.0
    return OperatingPoint(
        k=k,
        threshold=thr,
        precision=caught / k if k else 0.0,
        recall=caught / total_fraud if total_fraud else 0.0,
        n_fraud_caught=caught,
        n_fraud_total=total_fraud,
    )


def calibration_curve(y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10):
    """Reliability curve via quantile bins (robust to skewed score dists)."""
    edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(scores.min(), scores.max() + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges[1:-1]), 0, len(edges) - 2)
    mean_pred, frac_pos = [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(float(scores[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
    return np.asarray(mean_pred), np.asarray(frac_pos)


def evaluate(y_true: np.ndarray, scores: np.ndarray, alert_k: int) -> EvalResult:
    pr_auc = float(average_precision_score(y_true, scores))
    roc = float(roc_auc_score(y_true, scores))
    # Baseline = random/prevalence classifier -> PR-AUC equals the base rate.
    baseline = float(y_true.mean())
    precision, recall, _ = precision_recall_curve(y_true, scores)
    op = precision_at_k(y_true, scores, alert_k)
    calib = calibration_curve(y_true, scores)
    return EvalResult(
        pr_auc=pr_auc,
        roc_auc=roc,
        baseline_pr_auc=baseline,
        lift_over_baseline=pr_auc / baseline if baseline else float("inf"),
        op=op,
        pr_curve=(recall, precision),
        calibration=calib,
    )

"""Fraud model: HistGradientBoostingClassifier on the offline feature table.

Training respects a **time-based split** (train on the earlier fraction of the
stream, test on the later fraction) so evaluation never sees the future — the
same discipline the online feature engine enforces per event. Class imbalance is
handled with per-sample weights (fraud upweighted), which HGB consumes directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from . import FEATURE_NAMES, SEED


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    ts_split: float
    feature_names: tuple[str, ...]


def load_time_split(parquet_path: str | Path, test_frac: float = 0.25) -> Dataset:
    """Load the offline table and split by time (no shuffling, no leakage).

    Uses pyarrow column reads (float32) so memory stays modest even at 5M rows.
    """
    tbl = pq.read_table(parquet_path, columns=["ts", *FEATURE_NAMES, "is_fraud"])
    ts = tbl.column("ts").to_numpy()
    X = np.column_stack([tbl.column(n).to_numpy() for n in FEATURE_NAMES]).astype(np.float32)
    y = tbl.column("is_fraud").to_numpy().astype(np.int8)

    # Chronological split point.
    split_ts = float(np.quantile(ts, 1.0 - test_frac))
    train_mask = ts <= split_ts
    test_mask = ~train_mask
    return Dataset(
        X_train=X[train_mask], y_train=y[train_mask],
        X_test=X[test_mask], y_test=y[test_mask],
        ts_split=split_ts, feature_names=FEATURE_NAMES,
    )


def _sample_weights(y: np.ndarray) -> np.ndarray:
    """Balanced weights: total weight of each class equalized."""
    n = len(y)
    pos = max(1, int(y.sum()))
    neg = max(1, n - pos)
    w = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg)).astype(np.float64)
    return w


def train(ds: Dataset, seed: int = SEED) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.12,
        max_iter=300,
        max_leaf_nodes=63,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )
    clf.fit(ds.X_train, ds.y_train, sample_weight=_sample_weights(ds.y_train))
    return clf


def permutation_importance_fast(
    clf: HistGradientBoostingClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 3,
    max_rows: int = 60_000,
    seed: int = SEED,
) -> np.ndarray:
    """Permutation importance in average-precision units (subsampled for speed)."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    if len(y) > max_rows:
        idx = rng.choice(len(y), size=max_rows, replace=False)
        X, y = X[idx], y[idx]
    base = average_precision_score(y, clf.predict_proba(X)[:, 1])
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - average_precision_score(y, clf.predict_proba(Xp)[:, 1]))
        imp[j] = float(np.mean(drops))
    return imp

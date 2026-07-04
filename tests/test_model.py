"""The model must clear the imbalance baseline by a wide margin."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from fraud import FEATURE_NAMES
from fraud.features import OnlineFeatureStore
from fraud.generator import GeneratorConfig, TransactionStream
from fraud.metrics import evaluate
from fraud.model import _sample_weights


def _stream_to_arrays(n: int, cfg: GeneratorConfig):
    store = OnlineFeatureStore()
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    ts = np.empty(n, dtype=np.float64)
    for i, ev in enumerate(TransactionStream(cfg).stream(n)):
        X[i] = store.process(ev)
        y[i] = ev["is_fraud"]
        ts[i] = ev["ts"]
    return X, y, ts


@pytest.fixture(scope="module")
def trained():
    cfg = GeneratorConfig(n_users=8_000, seed=123)
    X, y, ts = _stream_to_arrays(160_000, cfg)
    split = np.quantile(ts, 0.75)
    tr, te = ts <= split, ts > split
    clf = HistGradientBoostingClassifier(
        learning_rate=0.15, max_iter=200, min_samples_leaf=100,
        early_stopping=True, random_state=123,
    )
    clf.fit(X[tr], y[tr], sample_weight=_sample_weights(y[tr]))
    scores = clf.predict_proba(X[te])[:, 1]
    return y[te], scores


def test_has_fraud_in_both_splits(trained):
    y_test, _ = trained
    assert y_test.sum() >= 20, "need enough positives to evaluate PR-AUC"


def test_pr_auc_beats_baseline(trained):
    y_test, scores = trained
    ev = evaluate(y_test, scores, alert_k=max(1, int(0.001 * len(y_test))))
    # Baseline PR-AUC under imbalance == prevalence. Demand a large lift.
    assert ev.pr_auc > 10 * ev.baseline_pr_auc
    assert ev.pr_auc > 0.30
    assert ev.roc_auc > 0.90


def test_alert_precision_is_high(trained):
    y_test, scores = trained
    ev = evaluate(y_test, scores, alert_k=max(1, int(0.001 * len(y_test))))
    # At a tight alert budget, precision should be far above prevalence.
    assert ev.op.precision > 0.5
    assert ev.op.n_fraud_caught >= 1


def test_scores_separate_classes(trained):
    y_test, scores = trained
    assert scores[y_test == 1].mean() > 5 * scores[y_test == 0].mean()

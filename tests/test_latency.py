"""Serving latency must be measured and within a real-time budget."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from fraud import FEATURE_NAMES
from fraud.generator import GeneratorConfig
from fraud.serving import benchmark_serving, sample_events, warm_store


@pytest.fixture(scope="module")
def served():
    cfg = GeneratorConfig(n_users=5_000, seed=99)
    store = warm_store(40_000, cfg)
    # Tiny model that mimics the real serving shape (20 features in, prob out).
    rng = np.random.default_rng(0)
    Xd = rng.normal(size=(2_000, len(FEATURE_NAMES))).astype(np.float32)
    yd = (Xd[:, 3] + Xd[:, 10] > 1.0).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(Xd, yd)
    events = sample_events(3_000, cfg, skip=40_000)
    return benchmark_serving(clf, store, events)


def test_latency_report_is_populated(served):
    assert served.n == 3_000
    for v in (served.feat_p50_ms, served.feat_p99_ms,
              served.score_p50_ms, served.score_p99_ms,
              served.e2e_p50_ms, served.e2e_p99_ms):
        assert v > 0 and np.isfinite(v)


def test_percentile_ordering(served):
    assert served.feat_p99_ms >= served.feat_p50_ms
    assert served.score_p99_ms >= served.score_p50_ms
    assert served.e2e_p99_ms >= served.e2e_p50_ms


def test_realtime_budget(served):
    # Single-event online scoring must stay well under a 25ms p99 budget.
    assert served.e2e_p99_ms < 25.0
    assert served.throughput_eps > 1_000

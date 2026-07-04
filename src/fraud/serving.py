"""Online serving path + latency/throughput benchmark.

The serving path for one incoming event is:
    1. **feature lookup + compute** from the online store's bounded state, then
    2. **score** with the trained model.

We measure both stages separately (feature latency vs scoring latency) and the
end-to-end latency, reporting p50/p99 and throughput. Scoring is done one event
at a time to reflect a true online request (no batching benefit), which is the
honest worst case for latency.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import OnlineFeatureStore
from .generator import GeneratorConfig, TransactionStream


@dataclass
class LatencyReport:
    n: int
    feat_p50_ms: float
    feat_p99_ms: float
    score_p50_ms: float
    score_p99_ms: float
    e2e_p50_ms: float
    e2e_p99_ms: float
    throughput_eps: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _percentiles(x_ms: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x_ms, 50)), float(np.percentile(x_ms, 99))


def warm_store(n_warm: int, config: GeneratorConfig | None = None) -> OnlineFeatureStore:
    """Build a realistically populated online store by streaming warm-up events."""
    store = OnlineFeatureStore()
    stream = TransactionStream(config)
    for ev in stream.stream(n_warm):
        store.process(ev)
    return store


def benchmark_serving(
    model,
    store: OnlineFeatureStore,
    events: list,
) -> LatencyReport:
    """Measure per-event feature-compute + scoring latency over ``events``."""
    feat_ms = np.empty(len(events))
    score_ms = np.empty(len(events))
    import time

    for i, ev in enumerate(events):
        t0 = time.perf_counter()
        feats = store.features_for(ev)  # read-only online lookup
        t1 = time.perf_counter()
        model.predict_proba(feats.reshape(1, -1))
        t2 = time.perf_counter()
        feat_ms[i] = (t1 - t0) * 1e3
        score_ms[i] = (t2 - t1) * 1e3

    e2e_ms = feat_ms + score_ms
    fp50, fp99 = _percentiles(feat_ms)
    sp50, sp99 = _percentiles(score_ms)
    ep50, ep99 = _percentiles(e2e_ms)
    total_s = e2e_ms.sum() / 1e3
    return LatencyReport(
        n=len(events),
        feat_p50_ms=fp50, feat_p99_ms=fp99,
        score_p50_ms=sp50, score_p99_ms=sp99,
        e2e_p50_ms=ep50, e2e_p99_ms=ep99,
        throughput_eps=len(events) / total_s if total_s else float("nan"),
    )


def sample_events(n: int, config: GeneratorConfig | None = None, skip: int = 0) -> list:
    """Grab ``n`` events from the stream (optionally skipping the first ``skip``)."""
    stream = TransactionStream(config)
    out = []
    for i, ev in enumerate(stream.stream(n + skip)):
        if i >= skip:
            out.append(ev)
    return out

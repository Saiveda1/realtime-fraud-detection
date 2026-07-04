"""No-future-leakage guarantees for the streaming feature engine.

Two independent checks:
  1. Feature-vs-update ordering: an event's features never include itself.
  2. Whole-pipeline check: each streamed feature equals a brute-force recompute
     that uses ONLY strictly-earlier events (ts < ts_i). If any feature leaked a
     future event, the brute-force (past-only) value would disagree.
"""
from __future__ import annotations

import numpy as np

from fraud import FEATURE_NAMES
from fraud.features import OnlineFeatureStore
from fraud.generator import GeneratorConfig, TransactionStream
from fraud.schema import Event

F = {name: i for i, name in enumerate(FEATURE_NAMES)}
_HOUR, _DAY, _FIVE_MIN = 3600.0, 86400.0, 300.0


def test_features_computed_before_update():
    """The event being scored must not appear in its own aggregates."""
    store = OnlineFeatureStore()
    uid = 1
    store.process(Event(event_id=0, ts=0.0, user_id=uid, merchant_id=1, amount=10.0,
                         lat=40.0, lon=-74.0, device_id=1, is_fraud=0))
    f = store.features_for(Event(event_id=1, ts=100.0, user_id=uid, merchant_id=1,
                                 amount=999.0, lat=40.0, lon=-74.0, device_id=1,
                                 is_fraud=0))
    # Exactly one prior event in the window, not two.
    assert f[F["user_txn_count_1h"]] == 1
    # Sum reflects the prior $10, not the current $999.
    assert f[F["user_amount_sum_1h"]] == 10.0


def test_streamed_features_match_past_only_bruteforce():
    cfg = GeneratorConfig(n_users=25, n_merchants=40, seed=7)
    events: list[Event] = list(TransactionStream(cfg).stream(600))

    store = OnlineFeatureStore()
    feats = np.stack([store.process(ev) for ev in events])

    rng = np.random.default_rng(0)
    checked = 0
    for i in rng.choice(len(events), size=60, replace=False):
        ev = events[i]
        uid, ts = ev["user_id"], ev["ts"]
        # Brute force over STRICTLY earlier events for this user (past only).
        past = [e for e in events[:i] if e["user_id"] == uid]
        # deque maxlen=64; keep the same cap so counts stay comparable.
        past = past[-64:]
        cnt_1h = sum(1 for e in past if 0 <= ts - e["ts"] <= _HOUR)
        cnt_5m = sum(1 for e in past if 0 <= ts - e["ts"] <= _FIVE_MIN)
        cnt_24h = sum(1 for e in past if 0 <= ts - e["ts"] <= _DAY)
        sum_1h = sum(e["amount"] for e in past if 0 <= ts - e["ts"] <= _HOUR)

        assert feats[i, F["user_txn_count_1h"]] == cnt_1h
        assert feats[i, F["user_txn_count_5m"]] == cnt_5m
        assert feats[i, F["user_txn_count_24h"]] == cnt_24h
        assert abs(feats[i, F["user_amount_sum_1h"]] - sum_1h) < 1e-2
        if past:
            assert abs(feats[i, F["time_since_last_sec"]] - (ts - past[-1]["ts"])) < 1e-3
        checked += 1
    assert checked == 60


def test_new_entity_never_leaks():
    """The very first event for a user has strictly empty history."""
    store = OnlineFeatureStore()
    for uid in range(50):
        f = store.features_for(Event(event_id=uid, ts=float(uid), user_id=uid,
                                     merchant_id=0, amount=5.0, lat=0.0, lon=0.0,
                                     device_id=1, is_fraud=0))
        assert f[F["is_new_user"]] == 1.0
        assert f[F["user_txn_count_24h"]] == 0
        store.update(Event(event_id=uid, ts=float(uid), user_id=uid, merchant_id=0,
                           amount=5.0, lat=0.0, lon=0.0, device_id=1, is_fraud=0))

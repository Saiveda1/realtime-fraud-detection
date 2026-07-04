"""Rolling-aggregate correctness on a hand-computed tiny fixture."""
from __future__ import annotations

import numpy as np

from fraud import FEATURE_NAMES
from fraud.features import OnlineFeatureStore
from fraud.schema import Event, haversine_km

F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _ev(eid, ts, uid, mid, amount, lat=40.0, lon=-74.0, device=1) -> Event:
    return Event(event_id=eid, ts=ts, user_id=uid, merchant_id=mid, amount=amount,
                 lat=lat, lon=lon, device_id=device, is_fraud=0)


def test_counts_sums_and_time_since_last():
    store = OnlineFeatureStore()
    uid = 7
    # Three prior txns at t=0, 60, 1800s for user 7, then evaluate at t=2000.
    store.process(_ev(0, 0.0, uid, 100, 10.0))
    store.process(_ev(1, 60.0, uid, 101, 20.0))
    store.process(_ev(2, 1800.0, uid, 100, 30.0))

    f = store.features_for(_ev(3, 2000.0, uid, 102, 50.0))

    # All 3 within the last hour (dt = 2000, 1940, 200 <= 3600).
    assert f[F["user_txn_count_1h"]] == 3
    # Within last 5 min (<=300s): only the t=1800 txn (dt=200).
    assert f[F["user_txn_count_5m"]] == 1
    # Sum over last hour = 10+20+30.
    assert f[F["user_amount_sum_1h"]] == 60.0
    # Mean of history = 20, distinct merchants = {100,101} -> 2.
    assert f[F["user_amount_mean_hist"]] == 20.0
    assert f[F["user_distinct_merchants_hist"]] == 2
    # time_since_last = 2000 - 1800 = 200.
    assert f[F["time_since_last_sec"]] == 200.0
    # ratio to historical max (30) = 50/30.
    assert abs(f[F["amount_to_hist_max_ratio"]] - 50.0 / 30.0) < 1e-4


def test_windowing_excludes_old_events():
    store = OnlineFeatureStore()
    uid = 3
    store.process(_ev(0, 0.0, uid, 1, 100.0))       # 25h before eval
    store.process(_ev(1, 90_000.0, uid, 2, 5.0))    # ~1h before eval? see below
    # Evaluate at t = 90_000 + 7200 (2h later): first txn is 27.5h old (outside 24h),
    # second is 2h old (inside 24h, outside 1h).
    f = store.features_for(_ev(2, 97_200.0, uid, 3, 5.0))
    assert f[F["user_txn_count_24h"]] == 1     # only the 90_000 txn within 24h
    assert f[F["user_txn_count_1h"]] == 0      # nothing within the last hour
    assert f[F["user_txn_count_5m"]] == 0


def test_new_user_flag_and_neutral_features():
    store = OnlineFeatureStore()
    f = store.features_for(_ev(0, 100.0, 999, 1, 42.0))
    assert f[F["is_new_user"]] == 1.0
    assert f[F["user_txn_count_1h"]] == 0
    assert f[F["time_since_last_sec"]] == 0.0
    # Amount itself is a valid current-event feature.
    assert f[F["amount"]] == 42.0


def test_geo_speed_flags_impossible_travel():
    store = OnlineFeatureStore()
    uid = 5
    # First txn in NYC.
    store.process(_ev(0, 0.0, uid, 1, 20.0, lat=40.71, lon=-74.01))
    # Second txn in LA 60s later -> impossible speed.
    f = store.features_for(_ev(1, 60.0, uid, 2, 20.0, lat=34.05, lon=-118.24))
    expected_km = haversine_km(40.71, -74.01, 34.05, -118.24)
    assert abs(f[F["geo_dist_from_last_km"]] - expected_km) < 1.0
    # ~3900 km in 60s -> speed must be enormous (>> any real travel).
    assert f[F["geo_speed_kmh"]] > 100_000


def test_device_change_flag():
    store = OnlineFeatureStore()
    uid = 8
    store.process(_ev(0, 0.0, uid, 1, 20.0, device=111))
    same = store.features_for(_ev(1, 10.0, uid, 2, 20.0, device=111))
    changed = store.features_for(_ev(2, 10.0, uid, 2, 20.0, device=222))
    assert same[F["device_changed"]] == 0.0
    assert changed[F["device_changed"]] == 1.0

"""Online streaming feature engine (a.k.a. the online feature store).

The engine keeps **bounded per-entity state** and computes a feature vector for
each event from *only the events that preceded it* — the update of an entity's
state happens strictly *after* its feature vector is computed. That ordering is
the whole leakage-safety argument: a feature for the event at time ``t`` can
never see an event at time ``>= t``.

Bounded memory
--------------
* Per user we keep a ring buffer (``deque(maxlen=K)``) of the last ``K`` events
  plus a few running scalars. Per merchant we keep a ring buffer of recent
  timestamps/amounts. Both are O(K) per entity.
* The number of tracked entities is capped with an LRU eviction policy
  (``max_entities``). Cold entities are evicted, so total state is bounded
  regardless of how many events stream through — this is what lets the same
  code run at 5M or 1B events.

The per-event cost is O(K) (a scan of the ring buffer), which at K=64 is a small
constant, so throughput is effectively linear in the number of events.
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from . import FEATURE_NAMES
from .schema import Event, haversine_km

_EPS = 1e-6
_HOUR = 3600.0
_DAY = 86400.0
_FIVE_MIN = 300.0


@dataclass
class _UserState:
    # ring buffer of (ts, amount, merchant_id) for windowed aggregates
    events: Deque[tuple[float, float, int]] = field(default_factory=lambda: deque(maxlen=64))
    last_ts: float = 0.0
    last_lat: float = 0.0
    last_lon: float = 0.0
    last_device: int = 0
    hist_max_amount: float = 0.0
    seen: bool = False


@dataclass
class _MerchantState:
    events: Deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=128))


class OnlineFeatureStore:
    """Bounded-state online feature store.

    Use :meth:`features_for` for a read-only lookup (the serving path), and
    :meth:`update` to fold an event into state. :meth:`process` does both in the
    leakage-safe order and is what the offline table build and the stream use.
    """

    def __init__(self, max_events_per_user: int = 64, max_entities: int = 250_000) -> None:
        self.max_events_per_user = max_events_per_user
        self.max_entities = max_entities
        self._users: "OrderedDict[int, _UserState]" = OrderedDict()
        self._merchants: "OrderedDict[int, _MerchantState]" = OrderedDict()

    # ------------------------------------------------------------------ #
    # state access with LRU eviction
    def _get_user(self, uid: int) -> _UserState:
        st = self._users.get(uid)
        if st is None:
            st = _UserState(events=deque(maxlen=self.max_events_per_user))
            self._users[uid] = st
            if len(self._users) > self.max_entities:
                self._users.popitem(last=False)  # evict least-recently-used
        else:
            self._users.move_to_end(uid)
        return st

    def _get_merchant(self, mid: int) -> _MerchantState:
        st = self._merchants.get(mid)
        if st is None:
            st = _MerchantState()
            self._merchants[mid] = st
            if len(self._merchants) > self.max_entities:
                self._merchants.popitem(last=False)
        else:
            self._merchants.move_to_end(mid)
        return st

    # ------------------------------------------------------------------ #
    def features_for(self, event: Event) -> np.ndarray:
        """Compute the feature vector for ``event`` from PAST state only.

        Read-only: does not mutate entity state. This is the exact code the
        online serving path runs.
        """
        uid = event["user_id"]
        mid = event["merchant_id"]
        ts = event["ts"]
        amount = event["amount"]

        u = self._users.get(uid)
        m = self._merchants.get(mid)

        amount_log = math.log1p(amount)
        hour = float(int(ts // _HOUR) % 24)

        if u is None or not u.seen:
            # Brand-new user: no history. Neutral/`is_new_user` flag carries signal.
            merchant_cnt_1h, merchant_mean_1h = self._merchant_window(m, ts)
            return np.array([
                amount, amount_log, hour,
                0.0, 0.0, 0.0,            # 5m/1h/24h counts
                0.0, 0.0,                 # 1h/24h sums
                0.0, 0.0,                 # mean/std hist
                0.0, 0.0,                 # zscore, ratio
                0.0,                      # time_since_last (0 == unknown)
                0.0,                      # distinct merchants
                0.0, 0.0,                 # geo dist, speed
                0.0, 1.0,                 # device_changed, is_new_user
                merchant_cnt_1h, merchant_mean_1h,
            ], dtype=np.float64)

        cnt_5m = cnt_1h = cnt_24h = 0
        sum_1h = sum_24h = 0.0
        n_hist = 0
        sum_all = 0.0
        sumsq_all = 0.0
        merchants: set[int] = set()
        for (e_ts, e_amt, e_mid) in u.events:
            dt = ts - e_ts
            if dt < 0:
                continue
            n_hist += 1
            sum_all += e_amt
            sumsq_all += e_amt * e_amt
            merchants.add(e_mid)
            if dt <= _FIVE_MIN:
                cnt_5m += 1
            if dt <= _HOUR:
                cnt_1h += 1
                sum_1h += e_amt
            if dt <= _DAY:
                cnt_24h += 1
                sum_24h += e_amt

        if n_hist:
            mean_hist = sum_all / n_hist
            var = max(0.0, sumsq_all / n_hist - mean_hist * mean_hist)
            std_hist = math.sqrt(var)
            zscore = (amount - mean_hist) / (std_hist + _EPS)
        else:
            mean_hist = std_hist = 0.0
            zscore = 0.0
        ratio = amount / (u.hist_max_amount + _EPS) if u.hist_max_amount > 0 else 0.0

        time_since_last = max(0.0, ts - u.last_ts) if u.last_ts > 0 else 0.0
        dist_km = haversine_km(u.last_lat, u.last_lon, event["lat"], event["lon"])
        speed = dist_km / ((time_since_last / _HOUR) + _EPS)
        device_changed = 1.0 if event["device_id"] != u.last_device else 0.0

        merchant_cnt_1h, merchant_mean_1h = self._merchant_window(m, ts)

        return np.array([
            amount, amount_log, hour,
            float(cnt_5m), float(cnt_1h), float(cnt_24h),
            sum_1h, sum_24h,
            mean_hist, std_hist,
            float(zscore), float(ratio),
            time_since_last, float(len(merchants)),
            dist_km, speed,
            device_changed, 0.0,
            merchant_cnt_1h, merchant_mean_1h,
        ], dtype=np.float64)

    @staticmethod
    def _merchant_window(m: "_MerchantState | None", ts: float) -> tuple[float, float]:
        if m is None:
            return 0.0, 0.0
        cnt = 0
        tot = 0.0
        for (e_ts, e_amt) in m.events:
            if 0 <= ts - e_ts <= _HOUR:
                cnt += 1
                tot += e_amt
        mean = tot / cnt if cnt else 0.0
        return float(cnt), float(mean)

    # ------------------------------------------------------------------ #
    def update(self, event: Event) -> None:
        """Fold ``event`` into entity state (call AFTER computing its features)."""
        u = self._get_user(event["user_id"])
        u.events.append((event["ts"], event["amount"], event["merchant_id"]))
        u.last_ts = event["ts"]
        u.last_lat = event["lat"]
        u.last_lon = event["lon"]
        u.last_device = event["device_id"]
        if event["amount"] > u.hist_max_amount:
            u.hist_max_amount = event["amount"]
        u.seen = True

        m = self._get_merchant(event["merchant_id"])
        m.events.append((event["ts"], event["amount"]))

    def process(self, event: Event) -> np.ndarray:
        """Leakage-safe: compute features from the past, THEN update state."""
        feats = self.features_for(event)
        self.update(event)
        return feats

    @property
    def n_tracked_users(self) -> int:
        return len(self._users)


assert len(FEATURE_NAMES) == 20, "FEATURE_NAMES must match the vector width"

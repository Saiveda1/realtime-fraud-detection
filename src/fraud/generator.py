"""Streaming transaction event generator.

Design goals
------------
* **Time-ordered stream.** Events are emitted in non-decreasing timestamp order
  (inter-arrival times drawn from an exponential distribution), which is exactly
  what a real streaming feature engine consumes — no global sort ever needed.
* **Bounded memory, scalable to 1B.** The only state kept is *per user*
  (home location, last location/time/device, spend profile), which is bounded by
  the number of users, never by the number of events. Emitting 1B events costs
  the same memory as emitting 1M. The public API is a generator: callers pull
  events one at a time and never materialize the full stream.
* **Realistic fraud.** Fraud is injected as multi-event *episodes* keyed to a
  victim user, reproducing four canonical attack shapes (velocity/card-testing,
  geo-impossible, amount anomaly, account takeover). Heavy class imbalance
  (~0.5% fraud) is the default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .schema import Event


@dataclass
class GeneratorConfig:
    n_users: int = 60_000
    n_merchants: int = 8_000
    fraud_rate: float = 0.005          # target fraction of fraud events (~0.5%)
    mean_interarrival_sec: float = 2.0  # global stream cadence
    start_ts: float = 1_700_000_000.0   # arbitrary fixed epoch for reproducibility
    seed: int = 42


@dataclass
class _UserProfile:
    """Bounded per-user state — this is the generator's entire memory footprint."""

    home_lat: float
    home_lon: float
    device_id: int
    log_amount_mu: float   # lognormal spend profile
    log_amount_sigma: float
    last_ts: float = 0.0
    last_lat: float = 0.0
    last_lon: float = 0.0
    last_device: int = 0


@dataclass
class _PendingEpisode:
    """A queued fraud burst that will be emitted on the victim's next events."""

    user_id: int
    pattern: str
    remaining: int
    base_lat: float
    base_lon: float
    device_id: int
    amount_scale: float = 1.0
    step_sec: float = 5.0
    field_pad: int = field(default=0, repr=False)


class TransactionStream:
    """A rewindable, memory-bounded generator of labeled transaction events."""

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.cfg = config or GeneratorConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._users: list[_UserProfile] = self._init_users()

    def _init_users(self) -> list[_UserProfile]:
        rng = self._rng
        n = self.cfg.n_users
        # Users clustered around a handful of metro centers (continental US-ish box).
        centers = np.array(
            [[40.71, -74.01], [34.05, -118.24], [41.88, -87.63],
             [29.76, -95.37], [47.61, -122.33], [25.76, -80.19]]
        )
        pick = rng.integers(0, len(centers), size=n)
        lat = centers[pick, 0] + rng.normal(0, 0.4, n)
        lon = centers[pick, 1] + rng.normal(0, 0.4, n)
        devices = rng.integers(1, 2_000_000_000, size=n)
        mu = rng.normal(3.2, 0.5, n)         # exp(3.2) ~ $25 median spend
        sigma = rng.uniform(0.35, 0.8, n)
        return [
            _UserProfile(
                home_lat=float(lat[i]), home_lon=float(lon[i]),
                device_id=int(devices[i]), log_amount_mu=float(mu[i]),
                log_amount_sigma=float(sigma[i]),
                last_lat=float(lat[i]), last_lon=float(lon[i]),
                last_device=int(devices[i]),
            )
            for i in range(n)
        ]

    # ------------------------------------------------------------------ #
    def stream(self, n_events: int) -> Iterator[Event]:
        """Yield ``n_events`` labeled events in timestamp order.

        Memory is O(n_users), independent of ``n_events``. Fraud episodes are
        emitted as *consecutive bursts* on the victim user (a real velocity /
        takeover attack is back-to-back charges), interleaved into the global
        timeline via short intra-burst inter-arrivals.
        """
        cfg = self.cfg
        rng = self._rng
        ts = cfg.start_ts
        # Each trigger emits a whole episode (~mean_episode_len events), so scale
        # the per-event trigger probability down to hit the target fraud_rate.
        mean_episode_len = 3.9
        trigger_p = cfg.fraud_rate / mean_episode_len
        n_users = cfg.n_users
        exp = rng.exponential
        randint = rng.integers
        emitted = 0

        while emitted < n_events:
            ts += exp(cfg.mean_interarrival_sec)
            uid = int(randint(0, n_users))
            user = self._users[uid]

            if rng.random() < trigger_p:
                # Emit the full fraud burst back-to-back.
                ep = self._make_episode(uid, user, ts, rng)
                for _ in range(ep.remaining):
                    if emitted >= n_events:
                        break
                    yield self._emit_fraud_beat(emitted, ts, uid, user, ep)
                    emitted += 1
                    ts += ep.step_sec
                continue

            yield self._emit_legit(emitted, ts, uid, user, rng)
            emitted += 1

    # ------------------------------------------------------------------ #
    def _emit_legit(self, event_id, ts, uid, user, rng) -> Event:
        amount = float(np.exp(rng.normal(user.log_amount_mu, user.log_amount_sigma)))
        # Small jitter around home; occasional legitimate travel.
        if rng.random() < 0.03:
            lat = user.home_lat + rng.normal(0, 3.0)
            lon = user.home_lon + rng.normal(0, 3.0)
        else:
            lat = user.home_lat + rng.normal(0, 0.05)
            lon = user.home_lon + rng.normal(0, 0.05)
        merchant = int(rng.integers(0, self.cfg.n_merchants))
        device = user.device_id
        self._update_user(user, ts, lat, lon, device)
        return Event(
            event_id=event_id, ts=float(ts), user_id=uid, merchant_id=merchant,
            amount=round(amount, 2), lat=float(lat), lon=float(lon),
            device_id=device, is_fraud=0,
        )

    def _make_episode(self, uid, user, ts, rng) -> _PendingEpisode:
        pattern = rng.choice(
            ["velocity_attack", "geo_impossible", "amount_anomaly", "account_takeover"],
            p=[0.35, 0.25, 0.25, 0.15],
        )
        if pattern == "velocity_attack":
            return _PendingEpisode(
                user_id=uid, pattern=pattern, remaining=int(rng.integers(4, 9)),
                base_lat=user.last_lat, base_lon=user.last_lon,
                device_id=user.device_id, amount_scale=0.15, step_sec=3.0,
            )
        if pattern == "geo_impossible":
            # Jump thousands of km away within seconds.
            return _PendingEpisode(
                user_id=uid, pattern=pattern, remaining=int(rng.integers(2, 5)),
                base_lat=user.last_lat + float(rng.choice([-1, 1])) * rng.uniform(25, 45),
                base_lon=user.last_lon + float(rng.choice([-1, 1])) * rng.uniform(40, 70),
                device_id=user.device_id, amount_scale=1.0, step_sec=30.0,
            )
        if pattern == "amount_anomaly":
            return _PendingEpisode(
                user_id=uid, pattern=pattern, remaining=int(rng.integers(1, 3)),
                base_lat=user.last_lat, base_lon=user.last_lon,
                device_id=user.device_id,
                amount_scale=float(rng.uniform(15, 60)), step_sec=60.0,
            )
        # account_takeover: brand-new device + geo shift + burst
        return _PendingEpisode(
            user_id=uid, pattern=pattern, remaining=int(rng.integers(3, 7)),
            base_lat=user.last_lat + float(rng.choice([-1, 1])) * rng.uniform(5, 20),
            base_lon=user.last_lon + float(rng.choice([-1, 1])) * rng.uniform(8, 30),
            device_id=int(self._rng.integers(1, 2_000_000_000)),
            amount_scale=float(rng.uniform(3, 12)), step_sec=8.0,
        )

    def _emit_fraud_beat(self, event_id, ts, uid, user, ep: _PendingEpisode) -> Event:
        rng = self._rng
        base_amt = float(np.exp(rng.normal(user.log_amount_mu, user.log_amount_sigma)))
        amount = max(0.5, base_amt * ep.amount_scale)
        lat = ep.base_lat + rng.normal(0, 0.02)
        lon = ep.base_lon + rng.normal(0, 0.02)
        merchant = int(rng.integers(0, self.cfg.n_merchants))
        device = ep.device_id
        self._update_user(user, ts, lat, lon, device)
        return Event(
            event_id=event_id, ts=float(ts), user_id=uid, merchant_id=merchant,
            amount=round(amount, 2), lat=float(lat), lon=float(lon),
            device_id=device, is_fraud=1,
        )

    @staticmethod
    def _update_user(user: _UserProfile, ts, lat, lon, device) -> None:
        user.last_ts = float(ts)
        user.last_lat = float(lat)
        user.last_lon = float(lon)
        user.last_device = int(device)

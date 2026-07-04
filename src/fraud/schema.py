"""Transaction event schema and helpers.

Events are plain dicts on the hot path (cheap to create, no per-event object
overhead when streaming millions), but the schema is documented here as a typed
``TypedDict`` so the fields and dtypes are unambiguous.
"""
from __future__ import annotations

import math
from typing import TypedDict


class Event(TypedDict):
    """A single card transaction event as it arrives on the stream."""

    event_id: int
    ts: float          # unix seconds, monotonically non-decreasing on the stream
    user_id: int
    merchant_id: int
    amount: float      # transaction amount in USD
    lat: float         # transaction latitude (degrees)
    lon: float         # transaction longitude (degrees)
    device_id: int     # device fingerprint hash
    is_fraud: int      # ground-truth label (0/1); NOT visible to the feature engine


# Fraud pattern taxonomy — used both by the generator (to inject) and the
# README (to explain what the model learns to catch).
FRAUD_PATTERNS: tuple[str, ...] = (
    "legit",
    "velocity_attack",     # rapid card-testing burst of small charges
    "geo_impossible",      # charge from an implausibly distant location
    "amount_anomaly",      # charge far larger than the user's norm
    "account_takeover",    # new device + geo jump + burst
)


_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points in kilometers."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))

"""Real-Time Fraud Detection: streaming feature engineering + online serving.

A repo-ready portfolio project demonstrating:
  * a streaming transaction event generator (scalable to 1B events),
  * an online feature store with bounded per-entity state,
  * a HistGradientBoosting fraud model trained on a leakage-free time split,
  * an online serving path with measured p50/p99 latency and throughput.
"""
from __future__ import annotations

__version__ = "1.0.0"

SEED = 42

# Feature schema — the canonical order the model + serving path both rely on.
FEATURE_NAMES: tuple[str, ...] = (
    "amount",
    "amount_log",
    "hour_of_day",
    "user_txn_count_5m",
    "user_txn_count_1h",
    "user_txn_count_24h",
    "user_amount_sum_1h",
    "user_amount_sum_24h",
    "user_amount_mean_hist",
    "user_amount_std_hist",
    "amount_zscore",
    "amount_to_hist_max_ratio",
    "time_since_last_sec",
    "user_distinct_merchants_hist",
    "geo_dist_from_last_km",
    "geo_speed_kmh",
    "device_changed",
    "is_new_user",
    "merchant_txn_count_1h",
    "merchant_amount_mean_1h",
)

__all__ = ["FEATURE_NAMES", "SEED", "__version__"]

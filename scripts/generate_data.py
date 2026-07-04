#!/usr/bin/env python
"""Stand-alone data generator: stream events -> offline feature Parquet table.

Bounded memory regardless of ``--rows`` (streams + flushes in row groups), so
this scales from a smoke test to 1B rows without OOM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud import SEED  # noqa: E402
from fraud.generator import GeneratorConfig  # noqa: E402
from fraud.pipeline import stream_to_parquet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "features.parquet")
    ap.add_argument("--fraud-rate", type=float, default=0.005)
    ap.add_argument("--n-users", type=int, default=60_000)
    ap.add_argument("--chunk-rows", type=int, default=250_000)
    args = ap.parse_args()

    cfg = GeneratorConfig(n_users=args.n_users, fraud_rate=args.fraud_rate, seed=SEED)
    print(f"Streaming {args.rows:,} events -> {args.out}")
    stats = stream_to_parquet(
        args.out, args.rows, cfg,
        chunk_rows=args.chunk_rows, progress_every=1_000_000,
    )
    print(f"Done: {stats.events_per_sec:,.0f} ev/s | "
          f"fraud {stats.n_fraud:,} ({stats.fraud_rate*100:.3f}%) | "
          f"peak RSS {stats.rss_mb:.0f} MB | users tracked {stats.n_tracked_users:,}")


if __name__ == "__main__":
    main()

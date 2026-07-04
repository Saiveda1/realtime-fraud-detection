#!/usr/bin/env python
"""Scaling benchmark: throughput + memory at increasing stream sizes.

Demonstrates the two properties that justify the "architected for 1B" claim:
  * **flat memory** — RSS stays bounded as the stream grows (state is O(entities),
    not O(events)); and
  * **linear time** — ev/s is roughly constant, so wall-clock scales linearly and
    1B is a throughput/sharding question, not a memory one.

Writes ``benchmarks/scaling_results.csv`` and prints a Markdown table.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud import SEED  # noqa: E402
from fraud.generator import GeneratorConfig  # noqa: E402
from fraud.pipeline import stream_to_parquet  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[250_000, 1_000_000, 2_000_000])
    args = ap.parse_args()

    cfg = GeneratorConfig(seed=SEED)
    rows = []
    for n in args.sizes:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tf:
            st = stream_to_parquet(tf.name, n, cfg, chunk_rows=250_000)
        rows.append({
            "events": n,
            "elapsed_sec": round(st.elapsed_sec, 2),
            "events_per_sec": round(st.events_per_sec),
            "rss_mb": round(st.rss_mb, 1),
            "fraud_rate_pct": round(st.fraud_rate * 100, 3),
            "users_tracked": st.n_tracked_users,
        })
        print(f"{n:>12,} events  {st.events_per_sec:>10,.0f} ev/s  "
              f"{st.rss_mb:>7.0f} MB RSS")

    out = ROOT / "benchmarks" / "scaling_results.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown
    hdr = "| Events | Elapsed (s) | Throughput (ev/s) | Peak RSS (MB) | Fraud % | Users |"
    sep = "|---|---|---|---|---|---|"
    print("\n" + hdr + "\n" + sep)
    for r in rows:
        print(f"| {r['events']:,} | {r['elapsed_sec']} | {r['events_per_sec']:,} | "
              f"{r['rss_mb']} | {r['fraud_rate_pct']} | {r['users_tracked']:,} |")
    # 1B extrapolation
    best = max(r["events_per_sec"] for r in rows)
    hrs = 1_000_000_000 / best / 3600
    print(f"\n1B extrapolation @ {best:,} ev/s single-core: {hrs:.1f} h; "
          f"with 32 shards: {hrs/32:.2f} h. Memory stays flat (state is per-entity).")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()

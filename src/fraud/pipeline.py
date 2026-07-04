"""Streaming pipeline: generate -> online-featurize -> write offline table.

This is the single streaming pass that proves the scale claim. Events are pulled
one at a time from the generator, run through the bounded-state
:class:`OnlineFeatureStore`, and flushed to Parquet in fixed-size row groups.
At no point is the full event stream or the full feature matrix held in memory,
so the same call streams 5M or 1B events in constant space (bounded by the
feature store's entity cap + one row-group buffer).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import FEATURE_NAMES
from .features import OnlineFeatureStore
from .generator import GeneratorConfig, TransactionStream


@dataclass
class StreamStats:
    n_events: int
    n_fraud: int
    fraud_rate: float
    elapsed_sec: float
    events_per_sec: float
    n_tracked_users: int
    rss_mb: float
    chunk_rows: int


def _rss_mb() -> float:
    """Resident set size in MB (best-effort, stdlib only)."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return kb / 1024.0 if kb > 1_000_000 else kb / 1024.0
    except Exception:  # pragma: no cover
        return float("nan")


def _arrow_schema() -> pa.schema:
    fields = [pa.field("event_id", pa.int64()), pa.field("ts", pa.float64())]
    fields += [pa.field(name, pa.float32()) for name in FEATURE_NAMES]
    fields += [pa.field("is_fraud", pa.int8())]
    return pa.schema(fields)


def stream_to_parquet(
    out_path: str | Path,
    n_events: int,
    config: GeneratorConfig | None = None,
    chunk_rows: int = 250_000,
    max_entities: int = 250_000,
    progress_every: int = 0,
) -> StreamStats:
    """Stream ``n_events`` through the online feature store into a Parquet table.

    Returns real throughput/memory stats measured during the pass.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = _arrow_schema()
    store = OnlineFeatureStore(max_entities=max_entities)
    stream = TransactionStream(config)

    n_feat = len(FEATURE_NAMES)
    # Pre-allocated column buffers for one row group (bounded memory).
    eid_buf = np.empty(chunk_rows, dtype=np.int64)
    ts_buf = np.empty(chunk_rows, dtype=np.float64)
    feat_buf = np.empty((chunk_rows, n_feat), dtype=np.float32)
    lbl_buf = np.empty(chunk_rows, dtype=np.int8)

    writer = pq.ParquetWriter(out_path, schema, compression="zstd")
    n_fraud = 0
    fill = 0
    t0 = time.perf_counter()

    def _flush(count: int) -> None:
        arrays = [
            pa.array(eid_buf[:count]),
            pa.array(ts_buf[:count]),
        ]
        arrays += [pa.array(feat_buf[:count, j]) for j in range(n_feat)]
        arrays += [pa.array(lbl_buf[:count])]
        writer.write_table(pa.Table.from_arrays(arrays, schema=schema))

    try:
        for ev in stream.stream(n_events):
            feats = store.process(ev)  # leakage-safe: features from past, then update
            eid_buf[fill] = ev["event_id"]
            ts_buf[fill] = ev["ts"]
            feat_buf[fill] = feats
            lbl_buf[fill] = ev["is_fraud"]
            n_fraud += ev["is_fraud"]
            fill += 1
            if fill == chunk_rows:
                _flush(fill)
                fill = 0
                if progress_every and (ev["event_id"] + 1) % progress_every == 0:
                    rate = (ev["event_id"] + 1) / (time.perf_counter() - t0)
                    print(f"  streamed {ev['event_id']+1:,} events  "
                          f"({rate:,.0f} ev/s, {n_fraud:,} fraud)")
        if fill:
            _flush(fill)
    finally:
        writer.close()

    elapsed = time.perf_counter() - t0
    return StreamStats(
        n_events=n_events,
        n_fraud=int(n_fraud),
        fraud_rate=n_fraud / n_events if n_events else 0.0,
        elapsed_sec=elapsed,
        events_per_sec=n_events / elapsed if elapsed else float("nan"),
        n_tracked_users=store.n_tracked_users,
        rss_mb=_rss_mb(),
        chunk_rows=chunk_rows,
    )


def stats_dict(stats: StreamStats) -> dict:
    return asdict(stats)

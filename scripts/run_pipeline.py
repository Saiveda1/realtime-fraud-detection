#!/usr/bin/env python
"""End-to-end run: stream -> train -> evaluate -> serve, persist artifacts.

Outputs
-------
* ``data/features.parquet`` — the offline feature table (built by the stream).
* ``data/model.pkl``        — the trained HistGradientBoosting model.
* ``data/results.json``     — all real numbers (scale, metrics, latency) that the
                              README and the screenshots read from.
* ``data/eval_arrays.npz``  — held-out scores/labels + curves for plotting.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud import FEATURE_NAMES, SEED  # noqa: E402
from fraud.generator import GeneratorConfig  # noqa: E402
from fraud.metrics import evaluate  # noqa: E402
from fraud.model import (  # noqa: E402
    load_time_split,
    permutation_importance_fast,
    train,
)
from fraud.pipeline import stats_dict, stream_to_parquet  # noqa: E402
from fraud.serving import benchmark_serving, sample_events, warm_store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=int, default=5_000_000)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--alert-budget", type=float, default=0.001,
                    help="fraction of the test stream we can afford to alert on")
    ap.add_argument("--serve-n", type=int, default=20_000)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    np.random.seed(SEED)
    data = args.data_dir
    data.mkdir(parents=True, exist_ok=True)
    features_path = data / "features.parquet"
    cfg = GeneratorConfig(seed=SEED)

    # ---- 1. Streaming pass -------------------------------------------------
    print(f"[1/4] Streaming {args.events:,} events through the online feature store...")
    stats = stream_to_parquet(
        features_path, args.events, cfg,
        chunk_rows=250_000, progress_every=1_000_000,
    )
    print(f"      {stats.events_per_sec:,.0f} ev/s | fraud {stats.n_fraud:,} "
          f"({stats.fraud_rate*100:.3f}%) | RSS {stats.rss_mb:.0f} MB | "
          f"{stats.n_tracked_users:,} users tracked")

    # ---- 2. Train on a leakage-free time split -----------------------------
    print("[2/4] Loading offline table + training HistGradientBoosting (time split)...")
    ds = load_time_split(features_path, test_frac=args.test_frac)
    t0 = time.perf_counter()
    model = train(ds, seed=SEED)
    train_sec = time.perf_counter() - t0
    print(f"      train rows {len(ds.y_train):,} | test rows {len(ds.y_test):,} | "
          f"fit {train_sec:.1f}s")

    with open(data / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)

    # ---- 3. Evaluate -------------------------------------------------------
    print("[3/4] Evaluating on the held-out (future) split...")
    scores = model.predict_proba(ds.X_test)[:, 1].astype(np.float32)
    alert_k = max(1, int(args.alert_budget * len(ds.y_test)))
    ev = evaluate(ds.y_test, scores, alert_k=alert_k)
    imp = permutation_importance_fast(model, ds.X_test, ds.y_test, seed=SEED)
    print(f"      PR-AUC {ev.pr_auc:.4f} (baseline {ev.baseline_pr_auc:.4f}, "
          f"{ev.lift_over_baseline:.0f}x) | ROC-AUC {ev.roc_auc:.4f}")
    print(f"      alert@{alert_k}: precision {ev.op.precision:.3f}, "
          f"recall {ev.op.recall:.3f} ({ev.op.n_fraud_caught}/{ev.op.n_fraud_total})")

    # ---- 4. Serving latency ------------------------------------------------
    print(f"[4/4] Measuring online serving latency ({args.serve_n:,} events)...")
    warm = warm_store(400_000, cfg)
    serve_events = sample_events(args.serve_n, cfg, skip=400_000)
    lat = benchmark_serving(model, warm, serve_events)
    print(f"      feat p50/p99 {lat.feat_p50_ms:.3f}/{lat.feat_p99_ms:.3f} ms | "
          f"score p50/p99 {lat.score_p50_ms:.3f}/{lat.score_p99_ms:.3f} ms | "
          f"e2e p99 {lat.e2e_p99_ms:.3f} ms | {lat.throughput_eps:,.0f} ev/s")

    # ---- persist artifacts -------------------------------------------------
    rec, prec = ev.pr_curve
    mp, fp = ev.calibration
    np.savez_compressed(
        data / "eval_arrays.npz",
        y_test=ds.y_test, scores=scores,
        pr_recall=rec, pr_precision=prec,
        cal_mean_pred=mp, cal_frac_pos=fp,
        importances=imp, feature_names=np.array(FEATURE_NAMES),
    )

    results = {
        "scale": stats_dict(stats),
        "train": {
            "train_rows": int(len(ds.y_train)),
            "test_rows": int(len(ds.y_test)),
            "train_fraud": int(ds.y_train.sum()),
            "test_fraud": int(ds.y_test.sum()),
            "fit_sec": train_sec,
            "ts_split": ds.ts_split,
        },
        "metrics": {
            "pr_auc": ev.pr_auc,
            "roc_auc": ev.roc_auc,
            "baseline_pr_auc": ev.baseline_pr_auc,
            "lift_over_baseline": ev.lift_over_baseline,
            "alert_k": ev.op.k,
            "alert_precision": ev.op.precision,
            "alert_recall": ev.op.recall,
            "alert_threshold": ev.op.threshold,
            "fraud_caught": ev.op.n_fraud_caught,
            "fraud_total": ev.op.n_fraud_total,
        },
        "latency": lat.as_dict(),
        "config": {
            "events": args.events, "test_frac": args.test_frac,
            "alert_budget": args.alert_budget, "n_users": cfg.n_users,
            "n_merchants": cfg.n_merchants, "seed": SEED,
        },
    }
    with open(data / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved artifacts to {data}/  (results.json, model.pkl, eval_arrays.npz)")


if __name__ == "__main__":
    main()

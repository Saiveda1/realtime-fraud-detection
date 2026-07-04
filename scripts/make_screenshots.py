#!/usr/bin/env python
"""Render the four portfolio screenshots from real pipeline artifacts.

Reads ``data/results.json`` + ``data/eval_arrays.npz`` (produced by
``run_pipeline.py``) and writes PNGs into ``assets/``:
  1. pr_curve.png        — Precision-Recall vs baseline
  2. score_dist.png      — score distribution, fraud vs legit (log y)
  3. feature_importance.png — top permutation importances
  4. kpi_dashboard.png   — throughput / latency / PR-AUC / alert precision KPIs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

from fraud.viztheme import (  # noqa: E402
    ACCENT, BAD, GOOD, GRID, MUTED, PALETTE, PANEL, TEXT, WARN,
    apply_theme, kpi, save_panel,
)

DATA = ROOT / "data"
ASSETS = ROOT / "assets"


def _load():
    with open(DATA / "results.json") as fh:
        res = json.load(fh)
    arr = np.load(DATA / "eval_arrays.npz", allow_pickle=True)
    return res, arr


def plot_pr_curve(res, arr):
    rec, prec = arr["pr_recall"], arr["pr_precision"]
    baseline = res["metrics"]["baseline_pr_auc"]
    pr_auc = res["metrics"]["pr_auc"]
    op = res["metrics"]

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.plot(rec, prec, color=ACCENT, lw=2.4,
            label=f"HistGBDT  (PR-AUC = {pr_auc:.3f})")
    ax.axhline(baseline, color=BAD, ls="--", lw=1.6,
               label=f"Prevalence baseline ({baseline:.4f})")
    # Mark the operating point.
    ax.scatter([op["alert_recall"]], [op["alert_precision"]], color=WARN, s=90,
               zorder=5, edgecolor="white", linewidth=0.6,
               label=(f"Alert budget: P={op['alert_precision']:.2f}, "
                      f"R={op['alert_recall']:.2f}"))
    ax.set_xlabel("Recall (fraud caught)")
    ax.set_ylabel("Precision (alert purity)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    lift = res["metrics"]["lift_over_baseline"]
    save_panel(fig, str(ASSETS / "pr_curve.png"),
               suptitle=f"Precision-Recall vs Baseline  —  {lift:.0f}x lift under 0.5% fraud")


def plot_score_dist(res, arr):
    y, scores = arr["y_test"], arr["scores"]
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    bins = np.linspace(0, 1, 60)
    ax.hist(scores[y == 0], bins=bins, color=PALETTE[0], alpha=0.75,
            label=f"Legit  (n={int((y == 0).sum()):,})")
    ax.hist(scores[y == 1], bins=bins, color=BAD, alpha=0.85,
            label=f"Fraud  (n={int((y == 1).sum()):,})")
    ax.set_yscale("log")
    ax.set_xlabel("Fraud score  P(fraud)")
    ax.set_ylabel("Count (log scale)")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", fontsize=9)
    save_panel(fig, str(ASSETS / "score_dist.png"),
               suptitle="Score Separation: Fraud vs Legit (held-out future split)")


def plot_feature_importance(res, arr):
    names = list(arr["feature_names"])
    imp = arr["importances"]
    order = np.argsort(imp)[::-1][:12][::-1]
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ypos = np.arange(len(order))
    ax.barh(ypos, imp[order], color=ACCENT, height=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("Permutation importance  (drop in PR-AUC when shuffled)")
    ax.grid(axis="y", alpha=0)
    save_panel(fig, str(ASSETS / "feature_importance.png"),
               suptitle="Top Streaming Features Driving Fraud Detection")


def plot_kpi_dashboard(res, arr):
    m, lat, scale = res["metrics"], res["latency"], res["scale"]
    y, scores = arr["y_test"], arr["scores"]

    fig = plt.figure(figsize=(13.5, 7.6))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.85, 1.15, 1.15],
                  hspace=0.5, wspace=0.28)

    def tile(r, c, label, value, sub, color=ACCENT):
        kpi(fig.add_subplot(gs[r, c]), label, value, sub, color)

    caught_pct = 100 * m["fraud_caught"] / max(1, m["fraud_total"])
    tile(0, 0, "Events streamed", f"{scale['n_events']/1e6:.1f}M",
         f"{scale['events_per_sec']:,.0f} ev/s", GOOD)
    tile(0, 1, "Serving p99", f"{lat['e2e_p99_ms']:.2f} ms",
         f"p50 {lat['e2e_p50_ms']:.2f} ms | {lat['throughput_eps']:,.0f} ev/s", ACCENT)
    tile(0, 2, "PR-AUC", f"{m['pr_auc']:.3f}",
         f"{m['lift_over_baseline']:.0f}x baseline | ROC {m['roc_auc']:.3f}", WARN)
    tile(0, 3, "Alert precision", f"{m['alert_precision']*100:.1f}%",
         f"caught {caught_pct:.0f}% of fraud @ budget", GOOD)

    # Row 2 left: latency breakdown bars
    ax1 = fig.add_subplot(gs[1, :2])
    stages = ["feat p50", "feat p99", "score p50", "score p99", "e2e p99"]
    vals = [lat["feat_p50_ms"], lat["feat_p99_ms"], lat["score_p50_ms"],
            lat["score_p99_ms"], lat["e2e_p99_ms"]]
    colors = [PALETTE[6], PALETTE[6], PALETTE[0], PALETTE[0], WARN]
    ax1.bar(stages, vals, color=colors)
    ax1.set_ylabel("latency (ms)")
    ax1.set_title("Online serving latency breakdown", fontsize=11)
    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT)
    ax1.tick_params(axis="x", labelsize=8)

    # Row 2 right: precision@k as alert budget grows
    ax2 = fig.add_subplot(gs[1, 2:])
    order = np.argsort(-scores)
    ks = np.unique(np.linspace(1, min(len(scores), 20000), 40).astype(int))
    ysorted = y[order]
    csum = np.cumsum(ysorted)
    prec_at_k = csum[ks - 1] / ks
    budget_pct = 100 * ks / len(scores)
    ax2.plot(budget_pct, prec_at_k, color=ACCENT, lw=2.2)
    ax2.set_xlabel("alert budget (% of stream flagged)")
    ax2.set_ylabel("precision@k")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Alert precision vs budget", fontsize=11)

    # Row 3 left: cumulative fraud caught vs stream reviewed (gain curve)
    ax3 = fig.add_subplot(gs[2, :2])
    total_fraud = max(1, int(y.sum()))
    frac_reviewed = np.arange(1, len(scores) + 1) / len(scores)
    frac_caught = np.cumsum(ysorted) / total_fraud
    ax3.plot(frac_reviewed * 100, frac_caught * 100, color=GOOD, lw=2.2, label="model")
    ax3.plot([0, 100], [0, 100], color=MUTED, ls=":", lw=1.2, label="random")
    ax3.set_xlabel("% of stream reviewed (highest score first)")
    ax3.set_ylabel("% fraud caught")
    ax3.set_title("Fraud caught vs review effort", fontsize=11)
    ax3.legend(loc="lower right", fontsize=8)

    # Row 3 right: calibration
    ax4 = fig.add_subplot(gs[2, 2:])
    mp, fp = arr["cal_mean_pred"], arr["cal_frac_pos"]
    ax4.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.2, label="perfect")
    ax4.plot(mp, fp, "o-", color=PALETTE[4], lw=1.8, ms=5, label="model")
    ax4.set_xlabel("mean predicted P(fraud)")
    ax4.set_ylabel("observed fraud rate")
    ax4.set_title("Calibration (quantile bins)", fontsize=11)
    ax4.legend(loc="upper left", fontsize=8)

    save_panel(fig, str(ASSETS / "kpi_dashboard.png"),
               suptitle="Real-Time Fraud Detection — Production KPI Dashboard")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    apply_theme()
    res, arr = _load()
    plot_pr_curve(res, arr)
    plot_score_dist(res, arr)
    plot_feature_importance(res, arr)
    plot_kpi_dashboard(res, arr)
    print(f"Wrote 4 PNGs to {ASSETS}/")


if __name__ == "__main__":
    main()

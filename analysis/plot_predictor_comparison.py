"""Three-predictor comparison figures (P1).

Reads analysis/predictor_comparison/summary.json (structure:
    {label: {T_steps, dim, horizons: [
        {horizon_steps, eval_step,
         scalar: {weighted_R2, corr, wRMSE},
         firstorder: {weighted_R2, corr, wRMSE},
         replay: {weighted_R2, corr, wRMSE},
         effective_coords}]}}
for label in k32,k64,k128,k256)
and writes to figures/predictor_comparison/:
  fig1_R2_vs_stage.pdf/png     R^2 at h=20 vs K, 3 predictor curves
  fig2_R2_vs_horizon.pdf/png   R^2(h), 3 predictor curves, 4 panels
  fig3_corr_vs_stage.pdf/png   Corr at h=20 vs K, 3 predictor curves

Honest reporting: descriptive titles, no causal claims. The replay is the exact
reference (R^2~1); the scalar and first-order predictors are compared against it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KIMG = {"k32": 32, "k64": 64, "k128": 128, "k256": 256}
LABELS = ["k32", "k64", "k128", "k256"]
PREDS = ["scalar", "firstorder", "replay"]
PRED_LABEL = {"scalar": "scalar (cross-K)", "firstorder": "first-order scale-lag",
              "replay": "finite-history replay"}
COLORS = {"scalar": "#8e44ad", "firstorder": "#2980b9", "replay": "#27ae60"}


def _h20(data, L):
    return next(h for h in data[L]["horizons"] if h["horizon_steps"] == 20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path,
                    default=Path("analysis/predictor_comparison/summary.json"))
    ap.add_argument("--out", type=Path, default=Path("figures/predictor_comparison"))
    a = ap.parse_args(argv)

    data = json.loads(a.summary.read_text())
    a.out.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: R^2 at h=20 vs K, 3 predictor curves ----
    ks = [KIMG[L] for L in LABELS]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for p in PREDS:
        r2 = [_h20(data, L)[p]["weighted_R2"] for L in LABELS]
        ax.plot(ks, r2, "o-", color=COLORS[p], label=PRED_LABEL[p])
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("training stage $K$ (kimg, log)")
    ax.set_ylabel("Weighted $R^2$ at $h{=}20$")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("Predictor explanatory power vs training stage (h=20, fresh start)")
    fig.tight_layout()
    fig.savefig(a.out / "fig1_R2_vs_stage.pdf")
    fig.savefig(a.out / "fig1_R2_vs_stage.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: R^2(h), 3 predictor curves, 4 panels ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, L in zip(axes, LABELS):
        for p in PREDS:
            hs = [h["horizon_steps"] for h in data[L]["horizons"]]
            r2 = [h[p]["weighted_R2"] for h in data[L]["horizons"]]
            ax.plot(hs, r2, "o-", color=COLORS[p], label=PRED_LABEL[p], ms=3, lw=1.2)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(f"K={KIMG[L]}", fontsize=9)
        ax.set_xlabel("horizon $h$ (steps)")
        if ax is axes[0]:
            ax.set_ylabel("Weighted $R^2$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle("Predictor explanatory power vs horizon (fresh start)")
    fig.tight_layout()
    fig.savefig(a.out / "fig2_R2_vs_horizon.pdf")
    fig.savefig(a.out / "fig2_R2_vs_horizon.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: Corr at h=20 vs K, 3 predictor curves ----
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for p in PREDS:
        cr = [_h20(data, L)[p]["corr"] for L in LABELS]
        ax.plot(ks, cr, "o-", color=COLORS[p], label=PRED_LABEL[p])
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("training stage $K$ (kimg, log)")
    ax.set_ylabel("Corr at $h{=}20$")
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax.legend(fontsize=8)
    ax.set_title("Predictor correlation with actual $h$ vs training stage (h=20, fresh start)")
    fig.tight_layout()
    fig.savefig(a.out / "fig3_corr_vs_stage.pdf")
    fig.savefig(a.out / "fig3_corr_vs_stage.png", dpi=200)
    plt.close(fig)

    # ---- console summary ----
    print("K     predictor      R2(h20)   Corr(h20)  wRMSE(h20)")
    for L in LABELS:
        for p in PREDS:
            h = _h20(data, L)[p]
            print(f"{KIMG[L]:<5} {p:<14} {h['weighted_R2']:.4f}  {h['corr']:.4f}  "
                  f"{h['wRMSE']:.4f}")
        print()


if __name__ == "__main__":
    main()

"""Same-information-budget predictor comparison figures (P1, revised).

Reads analysis/predictor_comparison/summary.json (structure:
    {label: {T_steps, dim, a_bar, regimes: {
        "fresh"|"real": {horizons: [
            {horizon_steps, eval_step,
             global_scalar: {weighted_R2, corr, wRMSE},
             local_continuous: {weighted_R2, corr, wRMSE},
             discrete_replay: {weighted_R2, corr, wRMSE},
             Var_w_h_actual, h_actual_mean, effective_coords}]}}}})
and writes to figures/predictor_comparison/:
  fig1_R2_vs_stage.pdf/png    R^2 at h=20 vs K, 3 predictors, 2 regimes
  fig2_R2_vs_horizon.pdf/png  R^2(h), 3 predictors, 2 regimes x 4 K
  fig3_Var_w_vs_stage.pdf/png weighted target variance vs K, 2 regimes
  fig4_corr_vs_stage.pdf/png  Corr at h=20 vs K, 3 predictors, 2 regimes

Honest reporting: descriptive titles, no causal claims. All predictors use only
G^1 + a* (same information budget); no oracle comparison.
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
PREDS = ["global_scalar", "local_continuous", "discrete_replay"]
PRED_LABEL = {"global_scalar": "global scalar",
              "local_continuous": "local/continuous",
              "discrete_replay": "discrete replay"}
COLORS = {"global_scalar": "#8e44ad", "local_continuous": "#2980b9",
          "discrete_replay": "#27ae60"}
REGIMES = ["fresh", "real"]
REGIME_LABEL = {"fresh": "fresh start (zero history)", "real": "real accumulated history"}


def _h20(data, L, regime):
    return next(h for h in data[L]["regimes"][regime]["horizons"]
                if h["horizon_steps"] == 20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path,
                    default=Path("analysis/predictor_comparison/summary.json"))
    ap.add_argument("--out", type=Path, default=Path("figures/predictor_comparison"))
    a = ap.parse_args(argv)

    data = json.loads(a.summary.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    ks = [KIMG[L] for L in LABELS]

    # ---- Figure 1: R^2 at h=20 vs K, 3 predictors, 2 regimes ----
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        for p in PREDS:
            r2 = [_h20(data, L, regime)[p]["weighted_R2"] for L in LABELS]
            ax.plot(ks, r2, "o-", color=COLORS[p], label=PRED_LABEL[p])
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("training stage $K$ (kimg, log)")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(REGIME_LABEL[regime], fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Weighted $R^2$ at $h{=}20$")
        ax.legend(fontsize=8)
    fig.suptitle("Predictor explanatory power vs training stage (h=20, same information budget)")
    fig.tight_layout()
    fig.savefig(a.out / "fig1_R2_vs_stage.pdf")
    fig.savefig(a.out / "fig1_R2_vs_stage.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: R^2(h), 3 predictors, 2 regimes x 4 K ----
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharey=True, sharex=True)
    for ri, regime in enumerate(REGIMES):
        for ci, L in enumerate(LABELS):
            ax = axes[ri, ci]
            for p in PREDS:
                hs = [h["horizon_steps"] for h in data[L]["regimes"][regime]["horizons"]]
                r2 = [h[p]["weighted_R2"] for h in data[L]["regimes"][regime]["horizons"]]
                ax.plot(hs, r2, "o-", color=COLORS[p], label=PRED_LABEL[p], ms=3, lw=1.2)
            ax.axhline(0, color="k", lw=0.6, alpha=0.4)
            ax.set_title(f"K={KIMG[L]} ({REGIME_LABEL[regime]})", fontsize=8)
            ax.set_xlabel("horizon $h$")
            ax.grid(alpha=0.3)
            if ci == 0:
                ax.set_ylabel("Weighted $R^2$")
            if ri == 0 and ci == 0:
                ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle("Predictor explanatory power vs horizon (same information budget)")
    fig.tight_layout()
    fig.savefig(a.out / "fig2_R2_vs_horizon.pdf")
    fig.savefig(a.out / "fig2_R2_vs_horizon.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: Var_w(h_actual) at h=20 vs K, 2 regimes ----
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for regime in REGIMES:
        v = [_h20(data, L, regime)["Var_w_h_actual"] for L in LABELS]
        ax.plot(ks, v, "o-", label=REGIME_LABEL[regime])
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("training stage $K$ (kimg, log)")
    ax.set_ylabel(r"$\mathrm{Var}_w(h_{\mathrm{actual}})$ at $h{=}20$")
    ax.legend(fontsize=8)
    ax.set_title("Target variance of the update ratio (contextualizes $R^2$)")
    fig.tight_layout()
    fig.savefig(a.out / "fig3_Var_w_vs_stage.pdf")
    fig.savefig(a.out / "fig3_Var_w_vs_stage.png", dpi=200)
    plt.close(fig)

    # ---- Figure 4: Corr at h=20 vs K, 3 predictors, 2 regimes ----
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        for p in PREDS:
            cr = [_h20(data, L, regime)[p]["corr"] for L in LABELS]
            ax.plot(ks, cr, "o-", color=COLORS[p], label=PRED_LABEL[p])
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("training stage $K$ (kimg, log)")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(REGIME_LABEL[regime], fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Corr at $h{=}20$")
        ax.legend(fontsize=8)
    fig.suptitle("Predictor correlation with actual $h$ vs training stage (h=20)")
    fig.tight_layout()
    fig.savefig(a.out / "fig4_corr_vs_stage.pdf")
    fig.savefig(a.out / "fig4_corr_vs_stage.png", dpi=200)
    plt.close(fig)

    # ---- console summary ----
    print("K     regime  predictor       R2(h20)   Corr(h20)  wRMSE(h20)  Var_w(h20)")
    for L in LABELS:
        for regime in REGIMES:
            for p in PREDS:
                h = _h20(data, L, regime)[p]
                v = _h20(data, L, regime)["Var_w_h_actual"]
                print(f"{KIMG[L]:<5} {regime:<6} {p:<15} {h['weighted_R2']:.4f}  "
                      f"{h['corr']:.4f}  {h['wRMSE']:.2e}  {v:.2e}")
        print()


if __name__ == "__main__":
    main()

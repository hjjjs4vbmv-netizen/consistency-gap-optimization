"""Cross-K figures for the moment-memory mechanism experiment.

Reads the crossk_horizon_sweep summary.json (structure:
    {label: {T_steps, dim, a_star_mean, a_star_std,
             verify_u1_final, verify_ug_final, verify_u1_full_torch, verify_ug_full_torch,
             horizons: [{horizon_steps, eval_step, weighted_R2, corr, wRMSE, R_opt,
                         cosine, a_star_mean, a_star_std, h_pred_mean, h_actual_mean,
                         effective_coords}]}}
for label in k32,k64,k128,k256)
and writes to figures/cross_k_scalar_history/:
  fig1_R2_vs_training_stage.pdf/png   R2 at h=20 vs K (log-K axis)
  fig2_R2_vs_horizon.pdf/png          R2(h), 4 curves (one per K)
  fig3_scatter_h20.pdf/png            h_pred vs h_actual at h=20, 4 panels

Honest reporting: title/labels are descriptive ("explanatory power vs ..."), NOT
"degrades"/"causes". The R² values come straight from the JSON — no fitting to a
desired conclusion.
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
COLORS = {"k32": "#8e44ad", "k64": "#2980b9", "k128": "#c0392b", "k256": "#27ae60"}


def _h20(horizons):
    return next(h for h in horizons if h["horizon_steps"] == 20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path,
                    default=Path("analysis/crossk_scalar_history/summary.json"))
    ap.add_argument("--out", type=Path, default=Path("figures/cross_k_scalar_history"))
    a = ap.parse_args(argv)

    data = json.loads(a.summary.read_text())
    a.out.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: R2 (and Corr, R_opt) at h=20 vs training stage K ----
    ks = [KIMG[L] for L in LABELS]
    r2_20 = np.array([_h20(data[L]["horizons"])["weighted_R2"] for L in LABELS])
    cr_20 = np.array([_h20(data[L]["horizons"])["corr"] for L in LABELS])
    ro_20 = np.array([_h20(data[L]["horizons"])["R_opt"] for L in LABELS])
    astar = np.array([data[L]["a_star_mean"] for L in LABELS])

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("training stage $K$ (kimg, log)")
    ax.plot(ks, r2_20, "o-", color="#c0392b", label="Weighted $R^2$ at $h{=}20$")
    ax.plot(ks, cr_20, "s-", color="#2c3e50", label="Corr at $h{=}20$")
    ax.plot(ks, ro_20, "^-", color="#7f8c8d", label="$R_{opt}$ at $h{=}20$")
    ax.set_ylabel("metric value")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Scalar-history explanatory power across training stages (h=20)")
    fig.tight_layout()
    fig.savefig(a.out / "fig1_R2_vs_training_stage.pdf")
    fig.savefig(a.out / "fig1_R2_vs_training_stage.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: R2(h), 4 K curves ----
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for L in LABELS:
        hs = [h["horizon_steps"] for h in data[L]["horizons"]]
        r2 = [h["weighted_R2"] for h in data[L]["horizons"]]
        ax.plot(hs, r2, "o-", color=COLORS[L], label=f"K={KIMG[L]}")
    ax.set_xlabel("prediction horizon $h$ (steps)")
    ax.set_ylabel("Weighted $R^2$")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({h["horizon_steps"] for L in LABELS for h in data[L]["horizons"]}))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="training stage", fontsize=8)
    ax.set_title("Scalar-history explanatory power vs prediction horizon")
    fig.tight_layout()
    fig.savefig(a.out / "fig2_R2_vs_horizon.pdf")
    fig.savefig(a.out / "fig2_R2_vs_horizon.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3 (optional): coordinate-level scatter h_pred vs h_actual at h=20 ----
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
    for ax, L in zip(axes, LABELS):
        row = _h20(data[L]["horizons"])
        raw = Path(a.summary).parent / L / "raw_predictions"
        hp = np.load(raw / "h_pred_scalar_h20.npy")
        ha = np.load(raw / "h_actual_h20.npy")
        w = np.load(raw / "weights_h20.npy")
        n = hp.shape[0]
        # weighted subsample for a legible scatter
        wsum = w.sum()
        idx = rng.choice(n, size=min(30000, n), replace=False,
                         p=(w / wsum))
        ax.scatter(ha[idx], hp[idx], s=1, alpha=0.25, color=COLORS[L])
        vmax = np.percentile(np.concatenate([ha, hp]), 99.5)
        lim = [0, vmax]
        ax.plot(lim, lim, "k--", lw=0.7, alpha=0.5)
        ax.set_title(f"K={KIMG[L]}  R$^2$={row['weighted_R2']:.3f}  "
                     f"Corr={row['corr']:.3f}", fontsize=9)
        ax.set_xlabel("$h^{actual}$ (h=20)")
        if ax is axes[0]:
            ax.set_ylabel("$h^{scalar}$ (h=20)")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.grid(alpha=0.3)
    fig.suptitle("Scalar-history predictor: coordinate-level $h^{scalar}$ vs $h^{actual}$ at h=20")
    fig.tight_layout()
    fig.savefig(a.out / "fig3_scatter_h20.pdf")
    fig.savefig(a.out / "fig3_scatter_h20.png", dpi=200)
    plt.close(fig)

    # ---- console summary ----
    print("K     h=20 R2    Corr   R_opt  a*(mean)")
    for L in LABELS:
        h = _h20(data[L]["horizons"])
        print(f"{KIMG[L]:<5} {h['weighted_R2']:.4f} {h['corr']:.4f} {h['R_opt']:.4f} "
              f"{data[L]['a_star_mean']:.4f}")

    # also write a compact table for the summary.csv
    rows = []
    for L in LABELS:
        for h in data[L]["horizons"]:
            rows.append({
                "K_kimg": KIMG[L], "h": h["horizon_steps"],
                "weighted_R2": h["weighted_R2"], "corr": h["corr"],
                "wRMSE": h["wRMSE"], "R_opt": h["R_opt"], "cosine": h["cosine"],
                "a_star_mean": h["a_star_mean"], "a_star_std": h["a_star_std"],
            })
    import csv
    with (a.out.parent / "summary_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out.parent / "summary_table.csv")


if __name__ == "__main__":
    main()

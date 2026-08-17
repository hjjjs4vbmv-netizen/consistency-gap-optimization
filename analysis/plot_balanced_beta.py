"""Balanced-β figures: does balancing β1=β2 reduce the gap-induced R_opt?

Reads analysis/balanced_beta/summary.json (structure:
    {label: {T_steps, dim, configs: {name: {beta1, beta2, horizons: [
        {horizon_steps, eval_step, R_opt, Disp_h, corr_u1_ug,
         h_actual_mean, h_actual_std, effective_coords}]}}}}
for label in k32,k64,k128,k256; configs: standard, balanced_0.9, balanced_0.99,
balanced_0.999)
and writes to figures/balanced_beta/:
  fig1_Ropt_vs_beta.pdf/png     R_opt at h=20 vs β config, 4 panels (one per K)
  fig2_Ropt_vs_horizon.pdf/png  R_opt(h), one line per β config, 4 panels
  fig3_disp_corr_h20.pdf/png    Disp(h) and Corr(u1,ug) at h=20 vs β config

Honest reporting: descriptive titles, no causal claims. The core prediction
(R_opt^{β1=β2} < R_opt^{0.9,0.999}) is reported as observed, not asserted.
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
CONFIGS = ["standard", "balanced_0.9", "balanced_0.99", "balanced_0.999"]
CFG_LABEL = {"standard": r"$\beta_1{=}0.9,\ \beta_2{=}0.999$",
             "balanced_0.9": r"$\beta_1{=}\beta_2{=}0.9$",
             "balanced_0.99": r"$\beta_1{=}\beta_2{=}0.99$",
             "balanced_0.999": r"$\beta_1{=}\beta_2{=}0.999$"}
COLORS = {"standard": "#2c3e50", "balanced_0.9": "#8e44ad",
          "balanced_0.99": "#2980b9", "balanced_0.999": "#c0392b"}


def _h20(cfg):
    return next(h for h in cfg["horizons"] if h["horizon_steps"] == 20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path,
                    default=Path("analysis/balanced_beta/summary.json"))
    ap.add_argument("--out", type=Path, default=Path("figures/balanced_beta"))
    a = ap.parse_args(argv)

    data = json.loads(a.summary.read_text())
    a.out.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: R_opt at h=20 vs β config, 4 panels ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, L in zip(axes, LABELS):
        r20 = {n: _h20(data[L]["configs"][n])["R_opt"] for n in CONFIGS}
        std = r20["standard"]
        xs = np.arange(len(CONFIGS))
        bars = ax.bar(xs, [r20[n] for n in CONFIGS],
                      color=[COLORS[n] for n in CONFIGS], width=0.6)
        # mark the standard config
        bars[0].set_edgecolor("k"); bars[0].set_linewidth(1.5)
        for x, n in zip(xs, CONFIGS):
            ax.text(x, r20[n] + 0.002, f"{r20[n]:.3f}", ha="center", fontsize=8)
        ax.set_xticks(xs)
        ax.set_xticklabels([r"std", r"0.9", r"0.99", r"0.999"], fontsize=8)
        ax.set_title(f"K={KIMG[L]}  (std={std:.3f})", fontsize=9)
        ax.set_xlabel(r"$\beta_1{=}\beta_2$ config")
        if ax is axes[0]:
            ax.set_ylabel(r"$R_{opt}$ at $h{=}20$")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(r"Gap-induced $R_{opt}$ at h=20 under balanced vs standard $\beta$")
    fig.tight_layout()
    fig.savefig(a.out / "fig1_Ropt_vs_beta.pdf")
    fig.savefig(a.out / "fig1_Ropt_vs_beta.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: R_opt(h), one line per β config, 4 panels ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, L in zip(axes, LABELS):
        for n in CONFIGS:
            hs = [h["horizon_steps"] for h in data[L]["configs"][n]["horizons"]]
            ro = [h["R_opt"] for h in data[L]["configs"][n]["horizons"]]
            ax.plot(hs, ro, "o-", color=COLORS[n], label=CFG_LABEL[n], ms=3, lw=1.2)
        ax.set_title(f"K={KIMG[L]}", fontsize=9)
        ax.set_xlabel("horizon $h$ (steps)")
        if ax is axes[0]:
            ax.set_ylabel(r"$R_{opt}$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle(r"Gap-induced $R_{opt}$ vs horizon: balanced vs standard $\beta$")
    fig.tight_layout()
    fig.savefig(a.out / "fig2_Ropt_vs_horizon.pdf")
    fig.savefig(a.out / "fig2_Ropt_vs_horizon.png", dpi=200)
    plt.close(fig)

    # ---- Figure 3: Disp(h) and Corr(u1,ug) at h=20 vs β config ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, ylab in [(axes[0], "Disp_h", r"Disp$(h)$ at $h{=}20$"),
                             (axes[1], "corr_u1_ug", r"Corr$(u_1,u_g)$ at $h{=}20$")]:
        for L in LABELS:
            vals = [_h20(data[L]["configs"][n])[metric] for n in CONFIGS]
            ax.plot(CONFIGS, vals, "o-", label=f"K={KIMG[L]}", ms=4)
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels([r"std", r"0.9", r"0.99", r"0.999"])
        ax.set_xlabel(r"$\beta_1{=}\beta_2$ config")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(r"Update-ratio dispersion and update correlation at h=20")
    fig.tight_layout()
    fig.savefig(a.out / "fig3_disp_corr_h20.pdf")
    fig.savefig(a.out / "fig3_disp_corr_h20.png", dpi=200)
    plt.close(fig)

    # ---- console summary ----
    print("K     config        R_opt(h20)  Disp(h20)  Corr(h20)  h_mean")
    for L in LABELS:
        for n in CONFIGS:
            h = _h20(data[L]["configs"][n])
            print(f"{KIMG[L]:<5} {n:<14} {h['R_opt']:.4f}  {h['Disp_h']:.4f}  "
                  f"{h['corr_u1_ug']:.4f}  {h['h_actual_mean']:.4f}")
        print()

    # prediction check (honest): balanced < standard at h=20?
    print("Prediction check: R_opt^{beta1=beta2} < R_opt^{0.9,0.999} at h=20")
    for L in LABELS:
        std = _h20(data[L]["configs"]["standard"])["R_opt"]
        for n in CONFIGS[1:]:
            v = _h20(data[L]["configs"][n])["R_opt"]
            print(f"  {L}: {n} {'HOLDS' if v < std else 'FAILS'} "
                  f"({v:.4f} vs {std:.4f})")


if __name__ == "__main__":
    main()

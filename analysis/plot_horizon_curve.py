"""Unified horizon figure: scalar-history explanatory power vs trajectory horizon.

Reads the horizon_sweep.json table (R², Corr, R_opt, cosine at h=1,2,5,10,20)
and plots:
    - Weighted R²(t)          (left axis, primary)
    - R_opt(t)  (residual norm, reference-normalized)   (left axis)
    - cosine(t) (matching quality of the actual paired updates) (right axis)

x-axis = trajectory horizon t (1..20, log-scaled). Produces
`figures/horizon_R2_residual_cosine.pdf` + .png.

STATEMENT: scalar matching accurately explains the local (short-horizon) update,
but its explanatory power decays over the trajectory horizon as the non-scalar
residual accumulates. This is a mechanism diagnostic on the optimizer update
only — it does NOT extend to FID / generation-quality causality.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path,
                    default=Path("analysis/real_history/k256/horizon_sweep.json"))
    ap.add_argument("--out", type=Path, default=Path("figures/horizon_R2_residual_cosine"))
    a = ap.parse_args(argv)

    rows = json.loads(a.sweep.read_text())["horizons"]
    rows.sort(key=lambda r: r["horizon_steps"])
    h = np.array([r["horizon_steps"] for r in rows])
    r2 = np.array([r["weighted_R2"] for r in rows])
    r_opt = np.array([r["R_opt"] for r in rows])
    cos = np.array([r["cosine"] for r in rows])

    fig, ax1 = plt.subplots(figsize=(6.4, 4.4))
    # log-scaled horizon axis: 1,2,5,10,20
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(h)
    ax1.set_xticklabels([str(int(x)) for x in h])
    ax1.set_xlabel("trajectory horizon $t$ (steps, log)")

    ax1.plot(h, r2, "o-", color="#c0392b", label="Weighted $R^2(\\hat{h}^{scalar}, h^{actual})$")
    ax1.plot(h, r_opt, "s-", color="#2c3e50", label="$R_{opt}$ (normalized residual norm)")
    ax1.set_ylabel("Weighted $R^2$ / residual $R_{opt}$")
    ax1.set_ylim(0.0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(h, cos, "d-", color="#27ae60", label="$\\cos(u_1, u_g)$ (update matching)")
    ax2.set_ylabel("$\\cos(u_1,u_g)$", color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right", fontsize=8)

    ax1.set_title("Scalar-history predictor: explanatory power vs horizon")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"))
    fig.savefig(a.out.with_suffix(".png"), dpi=200)
    print(f"wrote {a.out}.pdf / .png")
    print("horizons:", ", ".join(f"h={int(x)}" for x in h))
    print("R2    :", " ".join(f"{v:.4f}" for v in r2))
    print("R_opt :", " ".join(f"{v:.4f}" for v in r_opt))
    print("cosine:", " ".join(f"{v:.4f}" for v in cos))


if __name__ == "__main__":
    main()

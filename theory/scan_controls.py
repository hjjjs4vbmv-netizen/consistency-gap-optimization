"""LR-matched and gradient-matched controls for the toy model (review P0-2).

Tests whether the g-dependent U-shape survives controls that remove the
effective-learning-rate interpretation:

  1. eta_scale sweep {0.05,0.1,0.25,0.5,1.0}  (eta = eta_scale/lambda_max(H_1))
  2. LR-matched:    eta_g = eta_1 * lambda_max(H_1)/lambda_max(H_g)
                    => eta_g * lambda_max(H_g) == eta_1*lambda_max(H_1) = const
                    so the *fastest* direction rate is identical across g;
                    any remaining g-dependence is curvature-shape / slow-dir.
  3. gradient-RMS matched: choose beta0 so initial grad RMS is equal across g.

Decision rule: if the U-shape disappears under (2), the original U was a
learning-rate artifact. If it persists, there is gap-specific geometry.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toy_core import (sample_t, base_gap_sigmoid, hessian_symmetric,
                      gd_final_error, sgd_expectation_exact, noise_cov_from_rms)


def build_base(seed=0, n=200000):
    t = sample_t(n, rng=np.random.default_rng(seed))
    d0 = base_gap_sigmoid(t)
    return t, d0


def run(sigma_d=0.5, out="."):
    os.makedirs(out, exist_ok=True)
    t, d0 = build_base()
    gs = np.arange(0.5, 2.01, 0.05)
    Ks = [50, 200, 1000]
    noises = [0.0, 0.01, 0.05]
    eta_scales = [0.05, 0.1, 0.25, 0.5, 1.0]

    H1 = hessian_symmetric(sigma_d, t, 1.0, d0)
    lam1_max = np.linalg.eigvalsh(H1)[-1]

    rows = []
    # ---- 1. eta_scale sweep (fixed-H1 normalization) ----
    for es in eta_scales:
        eta1 = es / lam1_max
        for g in gs:
            H = hessian_symmetric(sigma_d, t, g, d0)
            beta0 = np.ones(2) * 1e-2
            for K in Ks:
                e = gd_final_error(H, beta0, eta1, K)[0]
                rows.append(dict(control="fixed_lr", eta_scale=es, g=round(float(g),4),
                                 K=K, noise=0.0, error=float(e)))
                for nz in noises:
                    if nz == 0:
                        continue
                    nu = noise_cov_from_rms(H, nz)
                    rows.append(dict(control="fixed_lr", eta_scale=es, g=round(float(g),4),
                                     K=K, noise=nz, error=float(sgd_expectation_exact(H, beta0, eta1, K, nu))))

    # ---- 2. LR-matched: eta_g * lambda_max(H_g) = const = eta1*lam1_max ----
    for es in eta_scales:
        target = es  # eta_g * lam_max(H_g) = es  (i.e. eta1*lam1_max = es)
        for g in gs:
            H = hessian_symmetric(sigma_d, t, g, d0)
            lam_max = np.linalg.eigvalsh(H)[-1]
            eta_g = target / lam_max          # makes eta_g*lam_max == es for all g
            beta0 = np.ones(2) * 1e-2
            for K in Ks:
                e = gd_final_error(H, beta0, eta_g, K)[0]
                rows.append(dict(control="lr_matched", eta_scale=es, g=round(float(g),4),
                                 K=K, noise=0.0, error=float(e)))
                for nz in noises:
                    if nz == 0:
                        continue
                    nu = noise_cov_from_rms(H, nz)
                    rows.append(dict(control="lr_matched", eta_scale=es, g=round(float(g),4),
                                     K=K, noise=nz, error=float(sgd_expectation_exact(H, beta0, eta_g, K, nu))))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "toy_lr_controls.csv"), index=False)

    # ---- diagnostics: report optimal g* per (control, eta_scale, K, noise) ----
    print("=== optimal g* by control (eta_scale=1.0) ===")
    sub = df[df.eta_scale == 1.0]
    for (ctrl, K, nz), grp in sub.groupby(["control", "K", "noise"]):
        gstar = grp.loc[grp.error.idxmin(), "g"]
        emin = grp.error.min()
        emax = grp.error.max()
        spread = (emax - emin) / max(emin, 1e-30)
        print(f"  {ctrl:10s} K={K:4d} noise={nz}: g*={gstar:.2f} err_min={emin:.3e} spread={spread:.2f}")

    # ---- decision: does U-shape survive lr_matched? ----
    print("\n=== DECISION (does U survive LR-matching?) ===")
    for K in Ks:
        for nz in noises:
            fixed = df[(df.control=="fixed_lr")&(df.eta_scale==1.0)&(df.K==K)&(df.noise==nz)]
            matched = df[(df.control=="lr_matched")&(df.eta_scale==1.0)&(df.K==K)&(df.noise==nz)]
            f_spread = (fixed.error.max()-fixed.error.min())/max(fixed.error.min(),1e-30)
            m_spread = (matched.error.max()-matched.error.min())/max(matched.error.min(),1e-30)
            verdict = "PERSISTS (gap-specific)" if m_spread > 0.10 else "COLLAPSES (LR artifact)"
            print(f"  K={K} noise={nz}: fixed_spread={f_spread:.2f} matched_spread={m_spread:.4f} -> {verdict}")


if __name__ == "__main__":
    run(out=".")

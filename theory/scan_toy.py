"""Scan g x K x noise for the linear-Gaussian toy model.

Outputs:
  theory/toy_hessian.csv       - per-g Hessian spectrum, condition number, stability
  theory/toy_finite_budget.csv - per (g, K, noise) final error

Usage: python scan_toy.py  [--out PREFIX] [--sigma-d 0.5] [--gmin 0.5 --gmax 2.0 --gstep 0.05]
"""
import argparse
import os
import numpy as np
import pandas as pd
from toy_core import (sample_t, base_gap_sigmoid, hessian_symmetric,
                      hessian_power_terms, gd_spectral_radius, gd_final_error,
                      sgd_expectation_exact, stop_gradient_operator,
                      asym_spectral_report, noise_cov_from_rms,
                      v_g)


def build_base(sigma_d, n_t=200000, seed=0, q=256.0):
    t = sample_t(n_t, rng=np.random.default_rng(seed))
    delta0 = base_gap_sigmoid(t, q=q)
    return t, delta0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".", help="output dir")
    ap.add_argument("--sigma-d", type=float, default=0.5)
    ap.add_argument("--gmin", type=float, default=0.5)
    ap.add_argument("--gmax", type=float, default=2.0)
    ap.add_argument("--gstep", type=float, default=0.05)
    ap.add_argument("--Ks", default="50,200,1000")
    ap.add_argument("--noises", default="0,0.01,0.05")
    ap.add_argument("--eta-scale", type=float, default=1.0,
                    help="eta = eta_scale / lambda_max(H at g=1)")
    ap.add_argument("--n-t", type=int, default=200000)
    args = ap.parse_args()

    sigma_d = args.sigma_d
    t, delta0 = build_base(sigma_d, n_t=args.n_t)
    gs = np.arange(args.gmin, args.gmax + 1e-9, args.gstep)
    Ks = [int(x) for x in args.Ks.split(",")]
    noises = [float(x) for x in args.noises.split(",")]

    # Hessian of the reference g=1 (normalization anchor for eta)
    H1 = hessian_symmetric(sigma_d, t, 1.0, delta0)
    lam1 = np.linalg.eigvalsh(H1)
    lam1_max = lam1[-1]
    eta = args.eta_scale / lam1_max
    beta0 = np.ones(2) * 1e-2  # small initial error

    rows_hess = []
    rows_budget = []
    for g in gs:
        H = hessian_symmetric(sigma_d, t, g, delta0)
        lam = np.linalg.eigvalsh(H)
        cond = lam[-1] / max(lam[0], 1e-15)
        rho, _ = gd_spectral_radius(H, eta)
        stable = bool(0 < eta * lam[-1] < 2)
        # stability margin: distance of eta*lambda_max to the instability boundary 2
        stab_margin = 2.0 - eta * lam[-1]

        # stop-gradient asymmetric operator spectral report
        A = stop_gradient_operator(sigma_d, t, g, delta0)
        asym_spec, asym_norm = asym_spectral_report(A)

        rows_hess.append(dict(
            g=round(float(g), 6),
            lambda_min=float(lam[0]), lambda_max=float(lam[-1]),
            cond=float(cond), eta=float(eta),
            eta_lambda_max=float(eta * lam[-1]),
            spectral_radius=float(rho),
            stable=stable,
            stability_margin=float(stab_margin),
            asym_min=float(asym_spec[0]), asym_max=float(asym_spec[-1]),
            asym_norm2=float(asym_norm),
        ))

        # finite-budget final error for each (K, noise)
        for K in Ks:
            err0 = gd_final_error(H, beta0, eta, K)[0]
            rows_budget.append(dict(g=round(float(g), 6), K=K,
                                    noise=0.0, error=float(err0)))
            for noise in noises:
                if noise == 0:
                    continue
                nu = noise_cov_from_rms(H, noise)
                e = sgd_expectation_exact(H, beta0, eta, K, nu)
                rows_budget.append(dict(g=round(float(g), 6), K=K,
                                        noise=noise, error=float(e)))

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(rows_hess).to_csv(os.path.join(args.out, "toy_hessian.csv"),
                                   index=False)
    pd.DataFrame(rows_budget).to_csv(os.path.join(args.out, "toy_finite_budget.csv"),
                                     index=False)

    # Power terms H2 H3 H4 (report: H_g = g^2 H2 + g^3 H3 + g^4 H4)
    H2, H3, H4 = hessian_power_terms(sigma_d, t, gs, delta0)
    power_info = pd.DataFrame({
        "coef": ["H2", "H3", "H4"],
        "a00": [H2[0,0], H3[0,0], H4[0,0]],
        "a01": [H2[0,1], H3[0,1], H4[0,1]],
        "a11": [H2[1,1], H3[1,1], H4[1,1]],
    })
    power_info.to_csv(os.path.join(args.out, "toy_hessian_power_terms.csv"),
                      index=False)

    print(f"sigma_d={sigma_d}, eta={eta:.6f} (= {args.eta_scale}/lam1_max), lam1_max={lam1_max:.4f}")
    print(f"H_g power terms (g^2,g^3,g^4):")
    print(power_info.to_string(index=False))
    print(f"\nhessian rows: {len(rows_hess)}, budget rows: {len(rows_budget)}")
    print("saved:", os.path.join(args.out, "toy_hessian.csv"),
          os.path.join(args.out, "toy_finite_budget.csv"))


if __name__ == "__main__":
    main()

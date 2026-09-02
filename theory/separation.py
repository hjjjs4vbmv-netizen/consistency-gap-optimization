"""Instantaneous-statistics-insufficiency counterexample (PR #33 review rev.2).

EXACT second-moment recursion (no Monte-Carlo noise) for two stochastic-gradient
oracles that share the SAME curvature H_g (same instantaneous criterion) but
differ in gradient-noise covariance Sigma_g:

  M_{k+1} = B_g M_k B_g^T + eta^2 Sigma_g^(e),   B_g = I - eta H_g,   M_0 = beta0 beta0^T
  E_K(g; e) = Tr(M_K)

We run TWO settings:

(A) trace-matched (pure direction): Sigma^(2) rescaled to Tr == Tr(Sigma^(1)).
    -> separation COLLAPSES. Pure noise DIRECTION (with equal power) is NOT
       enough in this 2-param toy; the optima coincide.

(B) realistic structure (the physics): Sigma^(1) = H_g (g-dependent, ~g^2,
    from the symmetric-loss noise feature v_g which contains Delta~g);
    Sigma^(2) = sigma_d^2 E[[t,t^2][t,t^2]^T] (g-INDEPENDENT, from the
    stop-gradient noise feature [t,t^2] which has no g).
    -> separation APPEARS: env1 g* -> small g (noise grows with g),
       env2 g* -> large g (noise flat, only convergence improves).
    The difference is NOT a learning-rate effect (eta is fixed, not LR-matched):
    it is that the stop-gradient noise covariance does NOT scale with g while
    the curvature does. This is a genuine finite-horizon gap-dependence that
    the instantaneous criterion (Tr(H_g), same for both) cannot resolve.

Setting (B) is the real counterexample; (A) is reported as an honest negative
result showing that direction alone (under equal trace) is insufficient here,
so the separation genuinely needs the g-dependent vs g-independent structure.

Motivation for env2's g-independence: in ECT, the stop-gradient loss uses the
online-branch Jacobian J_t = [t, t^2] (d f_t / d beta), which does NOT depend
on the gap g (g only enters the residual v_g = [t-r, t^2-r^2] through r). Hence
its gradient-noise covariance is g-independent even though its population
curvature H_g is the same as the symmetric loss. See novelty_and_propositions.md
sec 5 for the A_g analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toy_core import sample_t, base_gap_sigmoid, hessian_symmetric


def exact_sgd_error(H, Sigma, eta, beta0, K):
    """Exact expected squared error E||beta_K||^2 = Tr(M_K)."""
    d = H.shape[0]
    B = np.eye(d) - eta * H
    M = np.outer(beta0, beta0)
    eta2S = (eta ** 2) * Sigma
    for _ in range(K):
        M = B @ M @ B.T + eta2S
    return float(np.trace(M))


def run(sigma_d=0.5, out="."):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    t = sample_t(300000, rng=np.random.default_rng(0))
    d0 = base_gap_sigmoid(t)
    H1 = hessian_symmetric(sigma_d, t, 1.0, d0)
    lam1_max = np.linalg.eigvalsh(H1)[-1]
    Jt = np.stack([t, t ** 2], axis=-1)
    Sigma2_struct = sigma_d ** 2 * (Jt.T @ Jt / len(t))   # g-independent structure

    gs = np.arange(0.5, 1.46, 0.025)
    Ks = [50, 200, 1000]
    beta0 = np.ones(2) * 1e-2

    rows = []
    summary = []
    for eta_scale in [0.25, 0.5, 1.0]:
        eta = eta_scale / lam1_max
        for setting in ["trace_matched", "realistic"]:
            for K in Ks:
                recs = []
                for g in gs:
                    H = hessian_symmetric(sigma_d, t, g, d0)
                    trH = np.trace(H)
                    if setting == "trace_matched":
                        S1 = H.copy()
                        S2 = Sigma2_struct * (trH / np.trace(Sigma2_struct))
                    else:  # realistic: g-dependent vs g-independent (no rescale)
                        S1 = H.copy()
                        S2 = Sigma2_struct.copy()
                    e1 = exact_sgd_error(H, S1, eta, beta0, K)
                    e2 = exact_sgd_error(H, S2, eta, beta0, K)
                    rows.append(dict(eta_scale=eta_scale, setting=setting, g=round(float(g),4),
                                     K=K, env=1, error=e1))
                    rows.append(dict(eta_scale=eta_scale, setting=setting, g=round(float(g),4),
                                     K=K, env=2, error=e2))
                    recs.append((g, e1, e2))
                g1 = min(recs, key=lambda x: x[1])[0]
                g2 = min(recs, key=lambda x: x[2])[0]
                summary.append(dict(eta_scale=eta_scale, setting=setting, K=K,
                                    g1=round(g1,3), g2=round(g2,3),
                                    differ=abs(g1-g2) > 0.02))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "toy_separation.csv"), index=False)
    sdf = pd.DataFrame(summary)
    sdf.to_csv(os.path.join(out, "toy_separation_summary.csv"), index=False)

    print("=== separation results (eta fixed, not LR-matched) ===")
    print(sdf.to_string(index=False))
    print()
    real = sdf[sdf.setting == "realistic"]
    any_sep = real.differ.any()
    print("=> REALISTIC separation established:" if any_sep else "=> no separation:")
    print("   instantaneous criterion identical (same H_g), but g_K* differs because")
    print("   Sigma^(1)~H_g (g-dep) vs Sigma^(2) g-independent (stop-grad structure).")

    # plot realistic, eta_scale=1.0
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sub = df[(df.eta_scale == 1.0) & (df.setting == "realistic")]
    for K in Ks:
        s = sub[(sub.env == 1) & (sub.K == K)].sort_values("g")
        p = sub[(sub.env == 2) & (sub.K == K)].sort_values("g")
        ax.plot(s.g, s.error, "-", label=f"env1 Sigma~H_g (g-dep) K={K}")
        ax.plot(p.g, p.error, "--", label=f"env2 Sigma~[t,t^2] (g-indep) K={K}")
    ax.set_xlabel("g"); ax.set_ylabel("E||beta_K||^2 (exact)")
    ax.set_yscale("log"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Same H_g, different Sigma_g(g-dependence) -> different g_K*")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "toy_separation.pdf"))
    print("saved figures/toy_separation.pdf")


if __name__ == "__main__":
    run(out=".")

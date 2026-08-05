"""True stop-gradient ECT toy: exact finite-horizon second-moment dynamics.

Model: f_beta(x_t, t) = z (1 + beta1 t + beta2 t^2),   x_t = m(t) z,  z~N(0,sigma_d^2).
Stop-gradient loss: L = 1/2 ( f_t - sg[f_r] )^2,   r = t - Delta,  Delta = min(g delta0, t - tmin).
Residual:  f_t - f_r = z v_g(t)^T beta,   v_g = [Delta, t^2 - r^2]^T
Online Jacobian:  J_t = [t, t^2]^T          (d f_t / d beta = z J_t)

Per-sample gradient:  grad = z^2 (v_g^T beta) J_t
Random update matrix:  Q_g(z,t) = I - eta z^2 J_t v_g(t)^T
Mean update operator:  A_g = E[z^2 J_t v_g^T] = sigma_d^2 E_t[J_t v_g^T]      (asymmetric, ~ g)
Forward-loss curvature: H_g = sigma_d^2 E_t[v_g v_g^T]                       (~ g^2)

EXACT second-moment recursion (exact for resampled (z,t) SGD, no MC):
    M_{k+1} = M_k - eta (A_g M_k + M_k A_g^T) + 3 eta^2 sigma_d^4 E_t[(v_g^T M_k v_g) J_t J_t^T]
    E_K(g) = Tr(M_K)

The noise term couples to M_k through the residual variance (v_g^T M v_g); it is
NOT a constant covariance Sigma_g. This is the honest difference from the
abstract-stochastic-oracle recursion used in the PR #33 separation counterexample.

We implement the linear action of T_g = E[Q_g (x) Q_g] on the 3-dim symmetric
basis of M, verify it against Monte-Carlo SGD, and scan g x K under three eta
modes (fixed, LR-matched-by-H, LR-matched-by-A).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toy_core import sample_t, base_gap_sigmoid, hessian_symmetric


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def build_vg_Jt(t, delta0, g, t_min=1e-3):
    """Return v_g (n,2), J_t (n,2) arrays."""
    Delta = np.minimum(g * delta0, t - t_min)
    r = t - Delta
    vg = np.stack([Delta, t ** 2 - r ** 2], axis=-1)
    Jt = np.stack([t, t ** 2], axis=-1)
    return vg, Jt


def build_A(sigma_d, t, delta0, g):
    """Mean stop-gradient update operator A_g = sigma_d^2 E_t[J_t v_g^T] (2x2)."""
    vg, Jt = build_vg_Jt(t, delta0, g)
    return sigma_d ** 2 * (Jt.T @ vg / len(t)), vg, Jt


def build_T_matrix(sigma_d, t, delta0, g, eta):
    """3x3 matrix of T_g = E[Q_g (x) Q_g] on the symmetric basis of M.

    Basis: E0 = [[1,0],[0,0]], E1 = [[0,1],[1,0]], E2 = [[0,0],[0,1]].
    Columns are the symmetric-3-vector of T_g(E_b).
    """
    A, vg, Jt = build_A(sigma_d, t, delta0, g)
    bases = [np.array([[1., 0.], [0., 0.]]),
             np.array([[0., 1.], [1., 0.]]),
             np.array([[0., 0.], [0., 1.]])]
    # noise coefficient tensor: E_t[ (v^T M v) J J^T ] is linear in M with
    # coefficient 3 eta^2 sigma_d^4 * E_t[ v_a v_b J_i J_j ].
    coef = (3 * eta ** 2 * sigma_d ** 4 / len(t)) * np.einsum(
        "na,nb,ni,nj->abij", vg, vg, Jt, Jt)          # (2,2,2,2)
    Tcols = []
    for Eb in bases:
        # term1: M ; term2: -eta(A M + M A^T) ; term3: noise
        TM = Eb - eta * (A @ Eb + Eb @ A.T)
        TM += np.einsum("abij,ab->ij", coef, Eb)       # sum over a,b of coef[...,a,b]*Eb[a,b]
        Tcols.append([TM[0, 0], TM[0, 1], TM[1, 1]])
    return np.array(Tcols).T, A, vg, Jt


def exact_E_K(sigma_d, t, delta0, g, eta, beta0, Ks):
    """Exact expected squared error E_K(g) = Tr(M_K) for each K in Ks.

    FIXED (review PR #34, issue 3): each K is computed independently from the
    initial m0 (T^K m0), NOT by cumulatively iterating across the Ks list
    (which gave T^70 for the second entry when Ks=[20,50,...]). We iterate
    over sorted Ks and advance only the difference to stay efficient.
    """
    T, _, _, _ = build_T_matrix(sigma_d, t, delta0, g, eta)
    m0 = np.array([beta0[0] ** 2, beta0[0] * beta0[1], beta0[1] ** 2])
    rho = max(abs(np.linalg.eigvals(T)))
    out = {}
    m = m0.copy()
    prev = 0
    for K in sorted(Ks):
        if K > prev:
            for _ in range(K - prev):
                m = T @ m
                if abs(m).max() > 1e30:
                    m = np.full(3, np.inf)
                    break
        out[K] = float(m[0] + m[2])
        prev = K
    return out, rho


def mc_sgd(sigma_d, t, delta0, g, eta, beta0, K, n_traj=2000, seed=0):
    """Monte-Carlo SGD E[||beta_K||^2] with resampled (z, t) each step."""
    rng = np.random.default_rng(seed)
    vg, Jt = build_vg_Jt(t, delta0, g)
    n = len(t)
    sq = 0.0
    for _ in range(n_traj):
        beta = beta0.copy()
        for _ in range(K):
            idx = rng.integers(0, n)
            z = rng.normal(0, sigma_d)
            grad = (z ** 2) * (vg[idx] @ beta) * Jt[idx]
            beta = beta - eta * grad
        sq += beta @ beta
    return sq / n_traj


# ---------------------------------------------------------------------------
# eta modes
# ---------------------------------------------------------------------------

def _grid():
    return [round(float(g), 4) for g in np.arange(0.5, 1.51, 0.05)]


def eta_fixed(sigma_d, t, delta0, eta1):
    # fixed eta, normalized to the TRUE operator A_1 (stable regime):
    # eta1 = eta_scale / rho(A_1)
    return dict((g, eta1) for g in _grid())


def eta_lr_match_H(sigma_d, t, delta0, eta1, H1_lmax):
    # eta_g s.t. eta_g * lambda_max(H_g) == eta1 * lambda_max(H_1)
    out = {}
    for g in _grid():
        H = hessian_symmetric(sigma_d, t, g, delta0)
        lmax = np.linalg.eigvalsh(H)[-1]
        out[g] = eta1 * H1_lmax / lmax
    return out


def eta_lr_match_A(sigma_d, t, delta0, eta1, A1_rho=None):
    """TRUE learning-rate matching for the stop-gradient dynamics (review PR #34
    issue 4). The mean update is governed by A_g (not H_g):
        E[beta_{k+1}] = (I - eta_g A_g) E[beta_k].
    With ||A_g|| ~ g, matching to A_1 uses the Frobenius-optimal scalar
        a(g) = <A_g, A_1>_F / ||A_1||_F^2,   eta_g = eta_1 / a(g).
    Under A-matching the curve is expected to be almost flat (spread ~1e-6),
    which is the cleanest statement of 'gap ~ optimizer-step rescaling'.
    """
    A1, _, _ = build_A(sigma_d, t, delta0, 1.0)
    denom = float(np.sum(A1 * A1))  # ||A_1||_F^2
    out = {}
    for g in _grid():
        A, _, _ = build_A(sigma_d, t, delta0, g)
        a = float(np.sum(A * A1)) / denom
        out[g] = eta1 / a if a > 1e-15 else eta1
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(sigma_d=0.5, out="."):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    t = sample_t(200000, rng=np.random.default_rng(0))
    d0 = base_gap_sigmoid(t)
    gs = _grid()
    Ks = [20, 50, 100, 200]
    beta0 = np.ones(2) * 1e-2
    _ = t  # keep name

    H1 = hessian_symmetric(sigma_d, t, 1.0, d0)
    H1_lmax = np.linalg.eigvalsh(H1)[-1]
    A1, _, _ = build_A(sigma_d, t, d0, 1.0)
    rhoA1 = max(abs(np.linalg.eigvals(A1)))
    # fixed eta normalized to the TRUE operator A_1 (stable regime, heavy tail):
    #   eta1 = eta_scale / rho(A_1). We use a SMALL eta_scale so the second
    #   moment is finite at short K under heavy-tail noise.
    eta1 = 0.005 / rhoA1

    # three modes: fixed, H-matched (wrong matching, shown for contrast), and
    # the TRUE A-matched control (mean update governed by A_g, not H_g).
    modes = {
        "fixed": eta_fixed(sigma_d, t, d0, eta1),
        "lr_match_H": eta_lr_match_H(sigma_d, t, d0, eta1, H1_lmax),
        "lr_match_A": eta_lr_match_A(sigma_d, t, d0, eta1, None),
    }

    # g-scaling diagnostics of A_g vs H_g, and T_g spectral radius (stability)
    scaling = []
    rows = []
    for g in gs:
        A, _, _ = build_A(sigma_d, t, d0, g)
        H = hessian_symmetric(sigma_d, t, g, d0)
        rhoA = max(abs(np.linalg.eigvals(np.eye(2) - eta1 * A)))   # with fixed eta
        rhoH = max(abs(np.linalg.eigvals(np.eye(2) - eta1 * H)))
        # spectral radius of the exact second-moment operator T_g (finite K)
        try:
            T, _, _, _ = build_T_matrix(sigma_d, t, d0, g, eta1)
            rhoT = max(abs(np.linalg.eigvals(T)))
        except Exception:
            rhoT = float("nan")
        scaling.append(dict(g=round(float(g), 4),
                            nA=float(np.linalg.norm(A)), nH=float(np.linalg.norm(H)),
                            trA=float(np.trace(A)), trH=float(np.trace(H)),
                            rho_mean_A_fixed=float(rhoA), rho_mean_H_fixed=float(rhoH),
                            rho_second_moment_T=float(rhoT)))
        for mode, etas in modes.items():
            eta = etas[g]
            E, _ = exact_E_K(sigma_d, t, d0, g, eta, beta0, Ks)
            for K in Ks:
                rows.append(dict(g=round(float(g), 4), mode=mode, K=K, eta=round(float(eta), 8),
                                 E=float(E[K])))
    sdf = pd.DataFrame(scaling)
    df = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "true_sg_horizon.csv"), index=False)
    sdf.to_csv(os.path.join(out, "true_sg_operators.csv"), index=False)

    # ---- g-star per (mode, K) ----
    print("\n=== g_K* per (mode, K) ===")
    crossover = {}
    for mode in modes:
        line = []
        prev = None
        for K in Ks:
            sub = df[(df["mode"] == mode) & (df.K == K)].sort_values("g")
            sub = sub[np.isfinite(sub.E)]
            if len(sub) == 0:
                line.append((K, float("nan"), True)); continue
            gs_star = sub.loc[sub.E.idxmin(), "g"]
            diverged_all = not np.any(np.isfinite(sub.E))
            line.append((K, gs_star, diverged_all))
            if prev is not None and abs(gs_star - prev) > 0.01:
                crossover[mode] = True
            prev = gs_star
        print(f"  {mode}: " + ", ".join(f"K={K}:g*={gs:g}" + ("(div)" if dv else "") for K, gs, dv in line))
    print(f"\n  crossover (g* varies with K): {crossover}")

    # ---- flatness under LR-match (honest LR-scaling test) ----
    print("\n=== flatness of error vs g under each mode (K=200) ===")
    for mode in modes:
        sub = df[(df["mode"] == mode) & (df.K == 200)]
        sub = sub[np.isfinite(sub.E)]
        spread = (sub.E.max() - sub.E.min()) / max(sub.E.min(), 1e-30)
        print(f"  {mode}: spread={spread:.4f} (g range {sub.g.min()}-{sub.g.max()})")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(Ks)))
    for ax, mode in zip(axes, modes):
        for c, K in zip(colors, Ks):
            sub = df[(df["mode"] == mode) & (df.K == K)].sort_values("g")
            sub = sub[np.isfinite(sub.E)]
            ax.plot(sub.g, sub.E, "-o", ms=2, color=c, label=f"K={K}")
            gstar = sub.loc[sub.E.idxmin(), "g"]
            ax.plot(gstar, sub.E.min(), "x", color="red", ms=5)
        ax.set_xlabel("g"); ax.set_ylabel("E_K(g) = Tr(M_K)")
        ax.set_yscale("log"); ax.set_title(mode); ax.grid(alpha=0.3)
        if mode == "fixed":
            ax.legend(fontsize=7)
    fig.suptitle("True stop-gradient toy: exact finite-horizon error vs g (red x = g_K*)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "true_sg_error_vs_g_budget.pdf"))
    print("\nsaved figures/true_sg_error_vs_g_budget.pdf, theory/true_sg_horizon.csv, theory/true_sg_operators.csv")


if __name__ == "__main__":
    run(out=".")

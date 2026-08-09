"""Predict the optimizer-update distortion h from the gap-scale history δ_j.

Implements the PR #45 moment-memory chain on real paired gradient histories:

    δ_j  →  A^(1)_{t,i}, A^(2)_{t,i}, B^(2)_{t,i}  →  ĥ_{t,i}
    ĥ_{t,i}  vs  h^actual_{t,i}

The gradient history comes from a paired (same-batch) sweep: for each virtual
step j we record the reference gradient G_{j,i} and the candidate gradient
G^g_{j,i}, and recover the per-step scalar scale δ_j as the coordinate-aggregate
best fit of G^g to G:

    δ_j ≈ <G^g_j, G_j> / ||G_j||^2 - 1

Then the moment-memory terms are the history-weighted gauges:

    A^(1)_{t,i} = ( Σ_j p_j δ_j G_{j,i} ) / ( Σ_j p_j G_{j,i} ),  p_j ∝ β1^{t-j}
    A^(2)_{t,i} = ( Σ_j q_j δ_j G²_{j,i} ) / ( Σ_j q_j G²_{j,i} ), q_j ∝ β2^{t-j}
    B^(2)_{t,i} = ( Σ_j q_j δ_j² G²_{j,i} ) / ( Σ_j q_j G²_{j,i} )

    ĥ_{t,i} = (1 + A^(1)) / sqrt(1 + 2 A^(2) + B^(2))

Outputs per evaluation state t:
    - weighted RMSE(ĥ, h^update)
    - correlation Corr(ĥ, h^update)
    - Disp(ĥ) (weighted std) vs R_opt (update residual)

The chain is self-contained: given the paired gradient history it predicts the
optimizer distortion from the scale history alone (no access to the optimizer
moments), which is the #45 theorem's content.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

BETA1, BETA2 = 0.9, 0.999


def recover_delta_j(grad_g: np.ndarray, grad_1: np.ndarray) -> float:
    """Per-step scalar scale: G^g ≈ (1+δ) G  =>  δ = <Gg,G>/||G||² - 1."""
    denom = float(np.sum(grad_1 * grad_1))
    if denom <= 0:
        return 0.0
    return float(np.sum(grad_g * grad_1) / denom) - 1.0


def moment_memory_terms(grad_hist_1: list[np.ndarray], grad_hist_g: list[np.ndarray],
                        t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (A1, A2, B2, delta_hist) at step t from gradient histories.

    grad_hist_1[j] / grad_hist_g[j] are (d,) arrays for steps j = 0..t.
    """
    d = grad_hist_1[0].shape[0]
    A1 = np.zeros(d); A2 = np.zeros(d); B2 = np.zeros(d)
    num1 = np.zeros(d); den1 = np.zeros(d)
    num2 = np.zeros(d); den2 = np.zeros(d)
    numB = np.zeros(d)
    delta_hist = []
    for j in range(t + 1):
        G = grad_hist_1[j]; Gg = grad_hist_g[j]
        dj = recover_delta_j(Gg, G)
        delta_hist.append(dj)
        p = BETA1 ** (t - j)
        q = BETA2 ** (t - j)
        num1 += p * dj * G; den1 += p * G
        num2 += q * dj * G * G; den2 += q * G * G
        numB += q * dj * dj * G * G
    with np.errstate(divide="ignore", invalid="ignore"):
        A1 = np.where(np.abs(den1) > 1e-30, num1 / np.where(np.abs(den1) > 1e-30, den1, 1.0), 0.0)
        A2 = np.where(np.abs(den2) > 1e-30, num2 / np.where(np.abs(den2) > 1e-30, den2, 1.0), 0.0)
        B2 = np.where(np.abs(den2) > 1e-30, numB / np.where(np.abs(den2) > 1e-30, den2, 1.0), 0.0)
    return A1, A2, B2, np.array(delta_hist)


def predict_h(A1: np.ndarray, A2: np.ndarray, B2: np.ndarray) -> np.ndarray:
    """ĥ = (1+A1) / sqrt(1 + 2 A2 + B2)."""
    radicand = np.maximum(1.0 + 2.0 * A2 + B2, 1e-30)
    return (1.0 + A1) / np.sqrt(radicand)


def actual_update_h(u1: np.ndarray, ug: np.ndarray) -> np.ndarray:
    """h^actual = ug / u1 on the support (else 1)."""
    h = np.ones_like(u1)
    sup = np.abs(u1) > 1e-30
    h[sup] = ug[sup] / u1[sup]
    return h


def weighted_rmse(h_pred: np.ndarray, h_act: np.ndarray, w: np.ndarray) -> float:
    sup = w > 0
    if not sup.any():
        return math.nan
    return math.sqrt(float(np.sum(w[sup] * (h_pred[sup] - h_act[sup]) ** 2) / np.sum(w[sup])))


def corr(h_pred: np.ndarray, h_act: np.ndarray, w: np.ndarray) -> float:
    sup = w > 0
    if sup.sum() < 2:
        return math.nan
    wp = w[sup]
    x, y = h_pred[sup], h_act[sup]
    xm = np.sum(wp * x) / np.sum(wp); ym = np.sum(wp * y) / np.sum(wp)
    cov = np.sum(wp * (x - xm) * (y - ym))
    vx = np.sum(wp * (x - xm) ** 2); vy = np.sum(wp * (y - ym) ** 2)
    if vx <= 0 or vy <= 0:
        return math.nan
    return float(cov / math.sqrt(vx * vy))


def dispersion(h_pred: np.ndarray, w: np.ndarray) -> float:
    sup = w > 0
    if not sup.any():
        return math.nan
    wp = w[sup]
    m = np.sum(wp * h_pred[sup]) / np.sum(wp)
    return math.sqrt(float(np.sum(wp * (h_pred[sup] - m) ** 2) / np.sum(wp)))


def main(args=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grad-history-1", type=Path, help="npy of stacked reference gradients (T,d)")
    ap.add_argument("--grad-history-g", type=Path, help="npy of stacked candidate gradients (T,d)")
    ap.add_argument("--u1", type=Path, help="npy of final reference update (d,)")
    ap.add_argument("--ug", type=Path, help="npy of final candidate update (d,)")
    ap.add_argument("--eval-step", type=int, default=-1, help="which step to evaluate (default last)")
    ap.add_argument("--out", type=Path, default=Path("analysis/moment_memory_prediction.json"))
    a = ap.parse_args(args)

    G1 = np.load(a.grad_history_1)      # (T, d)
    Gg = np.load(a.grad_history_g)      # (T, d)
    u1 = np.load(a.u1)                  # (d,)
    ug = np.load(a.ug)                  # (d,)
    T = G1.shape[0]
    t = T - 1 if a.eval_step < 0 else min(a.eval_step, T - 1)

    grad_hist_1 = [G1[j] for j in range(t + 1)]
    grad_hist_g = [Gg[j] for j in range(t + 1)]
    A1, A2, B2, delta_hist = moment_memory_terms(grad_hist_1, grad_hist_g, t)
    h_pred = predict_h(A1, A2, B2)
    h_act = actual_update_h(u1, ug)
    w = u1 ** 2

    rmse = weighted_rmse(h_pred, h_act, w)
    r = corr(h_pred, h_act, w)
    disp = dispersion(h_pred, w)
    # R_opt (update residual, reference-normalized)
    s_opt = float(np.sum(ug * u1) / max(np.sum(u1 * u1), 1e-30))
    R_opt = float(np.linalg.norm(ug - s_opt * u1) / max(np.linalg.norm(u1), 1e-30))

    result = {
        "T_steps": T,
        "eval_step": t,
        "delta_mean": float(np.mean(delta_hist)),
        "delta_std": float(np.std(delta_hist)),
        "h_pred": h_pred.tolist(),
        "h_actual": h_act.tolist(),
        "weights": w.tolist(),
        "weighted_RMSE_h_pred_vs_actual": rmse,
        "corr_h_pred_vs_actual": r,
        "Disp_h_pred": disp,
        "R_opt": R_opt,
        "Disp_vs_R_opt": (disp / R_opt) if R_opt > 1e-12 else math.nan,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))

    print("=== moment-memory prediction chain ===")
    print(f"steps: {T}, eval at t={t}")
    print(f"δ_j: mean={result['delta_mean']:.4f}, std={result['delta_std']:.4f}")
    print(f"weighted RMSE(ĥ, h^actual) = {rmse:.4e}")
    print(f"Corr(ĥ, h^actual)         = {r:.4f}")
    print(f"Disp(ĥ)                   = {disp:.4f}")
    print(f"R_opt                     = {R_opt:.4f}")
    print(f"Disp(ĥ)/R_opt             = {result['Disp_vs_R_opt']:.4f}  (≈1 if dispersion explains residual)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

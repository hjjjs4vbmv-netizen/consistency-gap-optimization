"""Verify the corrected stop-gradient operator A_g by finite differences.

Checks (on the linear-Gaussian toy):
  1. population loss E[L] = 1/2 beta^T H_g beta  (curvature == symmetric H_g)
  2. population gradient == H_g beta  (== sigma_d^2 A_g ... wait: grad = sigma_d^2 A_g beta? )
  3. per-sample gradient == z^2 (v_g^T beta) [t, t^2]  -> empirical mean == sigma_d^2 A_g beta
  4. symmetric part of A_g vs H_g (they need NOT be equal)
"""
import numpy as np
from toy_core import (sample_t, base_gap_sigmoid, hessian_symmetric,
                      stop_gradient_operator, v_g)


def loss_and_grad_sample(z, t, r, beta):
    """Per-sample stop-gradient loss and gradient for ONE (z, t) pair.

    f_t = z (1 + b1 t + b2 t^2),  f_r = z (1 + b1 r + b2 r^2)
    residual R = f_t - f_r = z [ b1 (t-r) + b2 (t^2 - r^2) ] = z v_g^T beta
    L = 0.5 R^2
    grad_beta L = R * d f_t / d beta = R * z [t, t^2]^T = z^2 (v_g^T beta) [t, t^2]^T
    """
    vg = np.array([t - r, t ** 2 - r ** 2])
    R = z * (vg @ beta)
    L = 0.5 * R ** 2
    Jt = np.array([t, t ** 2])
    grad = (z ** 2) * (vg @ beta) * Jt
    return L, grad


def main():
    rng = np.random.default_rng(0)
    sigma_d = 0.5
    t = sample_t(200000, rng=rng)
    delta0 = base_gap_sigmoid(t)
    g = 1.1
    Delta = np.minimum(g * delta0, t - 1e-3)
    r = t - Delta

    H = hessian_symmetric(sigma_d, t, g, delta0)
    A = stop_gradient_operator(sigma_d, t, g, delta0)
    print("H_g (symmetric Hessian) =\n", H)
    print("A_g (stop-gradient op) =\n", A)
    print("sym(A_g) =\n", 0.5 * (A + A.T))
    print("||A_g - sym(A_g)|| (antisym) =", np.linalg.norm(A - A.T) / 2)

    beta = np.array([0.03, -0.02])

    # --- Check 1: population loss curvature == H_g ---
    # Monte-Carlo population loss
    z = rng.normal(0, sigma_d, size=len(t))
    # vectorized per-sample loss
    vg_all = np.stack([Delta, t ** 2 - r ** 2], axis=-1)   # (n,2)
    R_all = z * (vg_all @ beta)                            # (n,)
    L_mc = 0.5 * np.mean(R_all ** 2)
    L_pred = 0.5 * beta @ H @ beta
    print(f"\n[1] population loss:  MC={L_mc:.6e}  pred 0.5 b^T H_g b={L_pred:.6e}  "
          f"rel_err={abs(L_mc-L_pred)/abs(L_pred):.2e}")

    # --- Check 2: population gradient == H_g beta (mean of per-sample grad) ---
    # per-sample grad = z^2 (v_g^T beta) [t, t^2]
    Jt_all = np.stack([t, t ** 2], axis=-1)                # (n,2)
    grad_all = (z ** 2)[:, None] * (vg_all @ beta)[:, None] * Jt_all   # (n,2)
    grad_mc = grad_all.mean(axis=0)
    grad_pred_H = H @ beta
    print(f"[2] pop gradient:  MC={grad_mc}  H_g@beta={grad_pred_H}  "
          f"rel_err={np.linalg.norm(grad_mc-grad_pred_H)/np.linalg.norm(grad_pred_H):.2e}")

    # --- Check 3: mean per-sample grad == A_g beta  (A_g already includes sigma_d^2) ---
    grad_pred_A = A @ beta
    print(f"[3] pop gradient:  MC={grad_mc}  A_g@beta={grad_pred_A}  "
          f"rel_err={np.linalg.norm(grad_mc-grad_pred_A)/max(np.linalg.norm(grad_pred_A),1e-15):.2e}")
    print("   (A_g already folds in sigma_d^2; MC uses z^2 so it also folds it in)")

    # --- Check 4: finite-difference gradient on the population loss ---
    # fd grad of E[L] should equal H_g beta (since pop loss = 0.5 b^T H_g b)
    eps = 1e-5
    def poploss(bb):
        Rr = z * (vg_all @ bb)
        return 0.5 * np.mean(Rr ** 2)
    fd = np.zeros(2)
    fd[0] = (poploss(beta + [eps, 0]) - poploss(beta - [eps, 0])) / (2 * eps)
    fd[1] = (poploss(beta + [0, eps]) - poploss(beta - [0, eps])) / (2 * eps)
    print(f"[4] finite-diff pop grad = {fd}  H_g@beta={grad_pred_H}  "
          f"rel_err={np.linalg.norm(fd-grad_pred_H)/np.linalg.norm(grad_pred_H):.2e}")


if __name__ == "__main__":
    main()

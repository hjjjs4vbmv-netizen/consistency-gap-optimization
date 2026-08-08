"""Numeric checks for the first-order moment-memory theorem (Role C, 2026-08-08).

Checks with the real ``torch.optim.RAdam``:

  T1. Constant δ_j ≡ δ  ->  h - 1 = O(δ²)   (quantitative null, Cor 1).
  T2. Time-varying δ_j  ->  h - 1 ≈ A^(1) - A^(2) + O(δ²), where
        A^(1) = (Σ p_j δ_j G_j)/(Σ p_j G_j),  p_j ∝ β1^{t-j}
        A^(2) = (Σ q_j δ_j G_j²)/(Σ q_j G_j²), q_j ∝ β2^{t-j}
      (moment-predicted vs actual h match — validating the memory mechanism).
  T3. Coordinate heterogeneity  ->  h coordinate-varying  ->  R_opt > 0.

Setup: G^g_{j,i} = (1+δ_j) G_{j,i}, paired RAdam arms, same step index.
We build δ_j history, then compare the moment formula with the actual update.
"""
import numpy as np
import torch
from torch.optim import RAdam

TINY = 1e-30
BETA1, BETA2 = 0.9, 0.999


def run_arms(dim, delta_schedule, lr=1e-3, seed=0):
    """Paired RAdam; G_g = (1+δ_j) G_1. Returns actual h, u1, and δ/G history."""
    torch.manual_seed(seed)
    p1 = torch.nn.Parameter(torch.zeros(dim))
    pg = torch.nn.Parameter(torch.zeros(dim))
    o1 = RAdam([p1], lr=lr)
    og = RAdam([pg], lr=lr)
    rng = np.random.default_rng(seed)
    n = len(delta_schedule)

    G_hist = []      # reference gradients per step
    h_actual = []    # actual update ratio per step
    for k in range(n):
        g1 = torch.from_numpy(rng.standard_normal(dim)).float()
        G_hist.append(g1.detach().clone())
        old1, oldg = p1.detach().clone(), pg.detach().clone()
        o1.zero_grad(); p1.grad = g1.clone()
        og.zero_grad(); pg.grad = ((1 + delta_schedule[k]) * g1).clone()
        o1.step(); og.step()
        u1 = p1.detach() - old1
        ug = pg.detach() - oldg
        sup = torch.abs(u1) > TINY
        h = torch.ones_like(u1)
        h[sup] = ug[sup] / u1[sup]
        h_actual.append(h.detach().clone())
    return torch.stack(h_actual), G_hist, np.asarray(delta_schedule)


def moment_predicted_h(t, G_hist, delta_sched, dim):
    """h_t - 1 ≈ A^(1) - A^(2) from the moment-memory formula."""
    j = torch.arange(t + 1)
    p = BETA1 ** (t - j)              # first-moment weights
    q = BETA2 ** (t - j)              # second-moment weights
    A1 = torch.zeros(dim); A2 = torch.zeros(dim)
    num1 = torch.zeros(dim); den1 = torch.zeros(dim)
    num2 = torch.zeros(dim); den2 = torch.zeros(dim)
    for jj in range(t + 1):
        G = G_hist[jj]
        d = delta_sched[jj]
        num1 += p[jj] * d * G; den1 += p[jj] * G
        num2 += q[jj] * d * G ** 2; den2 += q[jj] * G ** 2
    A1 = num1 / (den1 + TINY)
    A2 = num2 / (den2 + TINY)
    return 1.0 + A1 - A2, A1, A2


def main():
    dim = 64
    n = 60

    # ---- T1: constant δ = 0.3 -> h - 1 = O(δ²) ----
    d_const = np.full(n, 0.3)
    hc, Gc, _ = run_arms(dim, d_const, seed=1)
    hc_dev = (hc - 1.0).abs()
    print("=== T1: constant δ = 0.3 (quantitative null) ===")
    print(f"  max |h-1| over coords, step 40: {hc_dev[40].max().item():.4e}")
    print(f"  δ² = {0.09:.2f} (should bound h-1: O(δ²))")
    # corollary says h-1 = O(δ²); check it's far below δ=0.3
    assert hc_dev[40].max().item() < 0.09, "constant δ should give second-order h-1"

    # ---- T2: time-varying δ_j (two blocks) -> h-1 ≈ A1-A2 ----
    d_alt = np.where(np.arange(n) % 20 < 10, 0.3, -0.2).astype(float)
    ha, Ga, da = run_arms(dim, d_alt, seed=2)
    print("\n=== T2: time-varying δ_j (blocks 0.3 / -0.2) ===")
    for t in [19, 21, 40, 59]:
        h_pred, A1, A2 = moment_predicted_h(t, Ga, da, dim)
        h_act = ha[t]
        sup = torch.abs(h_act) > 0  # all coords active (gaussian)
        err = float(torch.norm(h_act[sup] - h_pred[sup]) / (torch.norm(h_act[sup]) + TINY))
        # A1 vs A2 difference (the mechanism)
        a1m = float(A1.mean()); a2m = float(A2.mean())
        print(f"  t={t:3d}: A1-A2 mean={a1m-a2m:+.4f}  "
              f"h_pred-1 mean={float((h_pred-1).mean()):+.4f}  "
              f"h_act-1 mean={float((h_act-1).mean()):+.4f}  "
              f"match rel-err={err:.2e}")
        assert err < 0.3, "first-order formula should approximate actual h"

    # ---- T3: coordinate heterogeneity -> R_opt > 0 ----
    # use two coordinates with very different gradient scales so temporal
    # composition differs -> h coordinate-varying
    print("\n=== T3: coordinate heterogeneity -> R_opt > 0 ===")
    torch.manual_seed(3)
    # custom: coordinate 0 has big gradient magnitude, coordinate 1 small
    p1 = torch.nn.Parameter(torch.zeros(2))
    pg = torch.nn.Parameter(torch.zeros(2))
    o1 = RAdam([p1], lr=1e-3)
    og = RAdam([pg], lr=1e-3)
    rng = np.random.default_rng(3)
    for k in range(60):
        d = 0.3 if (k % 20 < 10) else -0.2
        # INDEPENDENT random series per coordinate -> different temporal
        # gradient compositions (not proportional).
        g = torch.tensor([rng.standard_normal(), 0.05 * rng.standard_normal()])
        old1, oldg = p1.detach().clone(), pg.detach().clone()
        o1.zero_grad(); p1.grad = g.clone()
        og.zero_grad(); pg.grad = ((1 + d) * g).clone()
        o1.step(); og.step()
        u1 = p1.detach() - old1
        ug = pg.detach() - oldg
        if k == 59:
            sup = torch.abs(u1) > TINY
            h = ug[sup] / u1[sup]
            w = u1[sup].square()
            s_star = float((w * h).sum() / (w.sum() + TINY))
            R = float(torch.norm(ug - s_star * u1) / (torch.norm(u1) + TINY))
            print(f"  h per coord (t=59): {h.detach().numpy()}")
            print(f"  s* = {s_star:.4f}, R_opt = {R:.4f}")
            # mechanism demonstration: coordinate-varying h => R_opt > 0
            assert R > 1e-3, "coordinate heterogeneity should give R_opt > 0"
    print("\nALL MOMENT-MEMORY CHECKS PASSED")


if __name__ == "__main__":
    main()

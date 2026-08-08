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

    # ---- T1: constant δ = 0.3 ----
    # self-review fix: report BOTH phases. Unrectified (steps 0-4): h-1 = δ
    # (first order, P-R1). Rectified (step >= 5): h-1 ~ eps-level (null).
    d_const = np.full(n, 0.3)
    hc, Gc, _ = run_arms(dim, d_const, seed=1)
    hc_dev = (hc - 1.0).abs()
    print("=== T1: constant δ = 0.3 (phase-qualified null) ===")
    print(f"  max |h-1| step 0 (unrectified): {hc_dev[0].max().item():.4f}  (expect ~δ=0.3)")
    print(f"  max |h-1| step 4 (unrectified): {hc_dev[4].max().item():.4f}")
    print(f"  max |h-1| step 6 (rectified):   {hc_dev[6].max().item():.4e}  (null)")
    print(f"  max |h-1| step 40 (rectified):  {hc_dev[40].max().item():.4e}")
    # unrectified phase: h-1 ~ δ (first order)
    assert abs(hc_dev[0].max().item() - 0.3) < 0.05, "unrectified h-1 = δ"
    # rectified phase: h-1 much smaller than δ (null)
    assert hc_dev[6].max().item() < 0.01, "rectified constant-scale null"

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

    # ---- T3: coordinate heterogeneity -> R_opt > 0 (high-dim, self-review) ----
    # 64 coordinates with INDEPENDENT per-coordinate gradient series -> widely
    # different temporal compositions -> h coordinate-varying -> R_opt > 0.
    print("\n=== T3: coordinate heterogeneity -> R_opt > 0 (64-dim) ===")
    dim3 = 64
    p1 = torch.nn.Parameter(torch.zeros(dim3))
    pg = torch.nn.Parameter(torch.zeros(dim3))
    o1 = RAdam([p1], lr=1e-3)
    og = RAdam([pg], lr=1e-3)
    rng = np.random.default_rng(3)
    h_std_hist = []
    for k in range(60):
        d = 0.3 if (k % 20 < 10) else -0.2
        # each coordinate its own random walk-ish series (slow vs fast mix)
        g = torch.from_numpy(rng.standard_normal(dim3)).float()
        g[dim3 // 2:] *= 0.1                       # second half small-scale
        old1, oldg = p1.detach().clone(), pg.detach().clone()
        o1.zero_grad(); p1.grad = g.clone()
        og.zero_grad(); pg.grad = ((1 + d) * g).clone()
        o1.step(); og.step()
        u1 = p1.detach() - old1
        ug = pg.detach() - oldg
        if k >= 5:   # rectified phase only
            sup = torch.abs(u1) > TINY
            h = ug[sup] / u1[sup]
            w = u1[sup].square()
            s_star = float((w * h).sum() / (w.sum() + TINY))
            R = float(torch.norm(ug - s_star * u1) / (torch.norm(u1) + TINY))
            h_std_hist.append((h.std().item(), R))
            if k == 59:
                print(f"  h std (t=59): {h.std().item():.4f}  "
                      f"s* = {s_star:.4f}  R_opt = {R:.4f}")
                # mechanism demonstration: coordinate-varying h => R_opt > 0
                assert R > 0.01, "high-dim heterogeneity should give clear R_opt > 0"
    # R_opt should be substantially above 0 across the rectified phase
    maxR = max(r for _, r in h_std_hist)
    print(f"  max R_opt over rectified phase: {maxR:.4f}  (well above 0)")
    assert maxR > 0.02, "time-varying heterogeneous history should create R_opt > 0"
    # ---- T4: R_opt vs R_grad (leader's key comparison) ----
    # R_grad = raw-gradient directional residual (known small); R_opt = the
    # optimizer-update residual. The theorem's content is R_opt - R_grad > 0
    # under a nontrivial δ history, NOT H = R_opt (an identity).
    print("\n=== T4: R_opt(K) - R_grad(K) under time-varying δ ===")
    torch.manual_seed(4)
    dim4 = 64
    p1 = torch.nn.Parameter(torch.zeros(dim4))
    pg = torch.nn.Parameter(torch.zeros(dim4))
    o1 = RAdam([p1], lr=1e-3)
    og = RAdam([pg], lr=1e-3)
    rng = np.random.default_rng(4)
    for k in range(60):
        d = 0.3 if (k % 20 < 10) else -0.2
        g = torch.from_numpy(rng.standard_normal(dim4)).float()
        old1, oldg = p1.detach().clone(), pg.detach().clone()
        o1.zero_grad(); p1.grad = g.clone()
        og.zero_grad(); pg.grad = ((1 + d) * g).clone()
        o1.step(); og.step()
        u1 = p1.detach() - old1
        ug = pg.detach() - oldg
        if k == 59:
            # R_grad: raw-gradient residual (instantaneous scalar relation exact)
            grad1 = g.clone(); gradg = ((1 + d) * g).clone()
            s_grad = float(torch.dot(gradg, grad1) / torch.dot(grad1, grad1))
            R_grad = float(torch.norm(gradg - s_grad * grad1) / torch.norm(grad1))
            # R_opt: update residual
            s_opt = float(torch.dot(ug, u1) / torch.dot(u1, u1))
            R_opt = float(torch.norm(ug - s_opt * u1) / torch.norm(u1))
            print(f"  R_grad = {R_grad:.2e} (instantaneous, ~0 by construction)")
            print(f"  R_opt  = {R_opt:.4f} (optimizer update)")
            print(f"  R_opt - R_grad = {R_opt - R_grad:.4f}")
            assert R_opt > R_grad + 0.01, \
                "optimizer memory should raise residual above the gradient residual"
    # ---- T5: clean memory attribution (round-2 self-review) ----
    # Same CURRENT gradient, different δ history -> any R_opt difference is
    # PURELY memory-induced (excludes "last-step gradient differs" confound).
    print("\n=== T5: clean memory attribution (same current grad, diff history) ===")
    def run_hist(dim, delta_sched, seed):
        torch.manual_seed(seed)
        pa = torch.nn.Parameter(torch.zeros(dim)); pb = torch.nn.Parameter(torch.zeros(dim))
        oa = RAdam([pa], lr=1e-3); ob = RAdam([pb], lr=1e-3)
        rng = np.random.default_rng(seed)
        last_g = None
        for k, d in enumerate(delta_sched):
            g = torch.from_numpy(rng.standard_normal(dim)).float()
            last_g = g
            olda, oldb = pa.detach().clone(), pb.detach().clone()
            oa.zero_grad(); pa.grad = g.clone()
            ob.zero_grad(); pb.grad = ((1 + d) * g).clone()
            oa.step(); ob.step()
        return pa.detach() - olda, pb.detach() - oldb, last_g

    dim5 = 64; n5 = 60
    sched_alt = np.where(np.arange(n5) % 20 < 10, 0.3, -0.2)
    sched_const = np.full(n5, 0.3)
    u1a, uga, ga = run_hist(dim5, sched_alt, 4)
    u1b, ugb, gb = run_hist(dim5, sched_const, 4)
    def ropt(u1, ug):
        s = float(torch.dot(ug, u1) / torch.dot(u1, u1))
        return float(torch.norm(ug - s * u1) / torch.norm(u1))
    Ra, Rb = ropt(u1a, uga), ropt(u1b, ugb)
    print(f"  last-step gradients identical: {torch.allclose(ga, gb)}")
    print(f"  time-varying δ: R_opt = {Ra:.4f}")
    print(f"  constant   δ: R_opt = {Rb:.4f}")
    print(f"  difference (pure memory) = {abs(Ra - Rb):.4f}")
    assert torch.allclose(ga, gb)
    assert Ra > 0.05 and Rb < 1e-3, "memory is the sole cause of R_opt in T5"
    print("\nALL MOMENT-MEMORY CHECKS PASSED")


if __name__ == "__main__":
    main()

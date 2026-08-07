"""Numeric verification of the coordinate-wise history gauge theorem (P-R3 rev.4).

Checks with the real ``torch.optim.RAdam`` implementation:

T1. On rectified steps, the moment-defined coordinate history gauge predicts the
    actual single-step update ratio up to the implementation's small nonzero eps.
T2. Constant scalar history is the rectified null: update scale s* ~= 1 and the
    non-scalar optimizer residual is negligible.
T3. For a time-varying scalar history, a synthetic multi-coordinate example can
    produce nonzero optimizer residual despite zero instantaneous gradient
    directional residual.
T4. The direct projection residual equals the analytic weighted-dispersion
    expression, and the normalized history statistic H equals R_opt.

The theorem itself is stated in ``theory/radam_gap_equivalence.md`` with an
explicit effective-support condition; the time-varying example here is an
existence/mechanism check, not a universal implication.
"""

import numpy as np
import torch
from torch.optim import RAdam


TINY = 1e-30


def run_trace(dim, a_schedule, lr=1e-3, seed=0):
    """Run paired RAdam arms with G_g,k = a_k G_1,k and record updates/gauges."""
    torch.manual_seed(seed)
    p1 = torch.nn.Parameter(torch.zeros(dim))
    pg = torch.nn.Parameter(torch.zeros(dim))
    o1 = RAdam([p1], lr=lr)
    og = RAdam([pg], lr=lr)
    rng = np.random.default_rng(seed)

    h_all = []
    residual_all = []
    s_star_all = []
    u1_all = []
    ug_all = []

    for k, a_k in enumerate(a_schedule):
        g1 = torch.from_numpy(rng.standard_normal(dim)).float()
        old1 = p1.detach().clone()
        oldg = pg.detach().clone()

        o1.zero_grad()
        p1.grad = g1.clone()
        og.zero_grad()
        pg.grad = (float(a_k) * g1).clone()
        o1.step()
        og.step()

        u1 = p1.detach() - old1
        ug = pg.detach() - oldg
        support = torch.abs(u1) > TINY
        h = torch.ones_like(u1)
        h[support] = ug[support] / u1[support]

        w = u1.square()
        s_star = float(torch.dot(ug, u1) / (torch.dot(u1, u1) + TINY))
        residual = float(torch.norm(ug - s_star * u1) / (torch.norm(u1) + TINY))

        h_all.append(h.detach().clone())
        residual_all.append(residual)
        s_star_all.append(s_star)
        u1_all.append(u1.detach().clone())
        ug_all.append(ug.detach().clone())

    return {
        "h": torch.stack(h_all),
        "residual": np.asarray(residual_all),
        "s_star": np.asarray(s_star_all),
        "u1": torch.stack(u1_all),
        "ug": torch.stack(ug_all),
    }


def check_identity_from_moments():
    """T1: compare the moment-defined h with the actual rectified RAdam update."""
    torch.manual_seed(0)
    dim = 64
    lr = 1e-3
    a = 1.3
    p1 = torch.nn.Parameter(torch.zeros(dim))
    pg = torch.nn.Parameter(torch.zeros(dim))
    o1 = RAdam([p1], lr=lr)
    og = RAdam([pg], lr=lr)
    rng = np.random.default_rng(0)

    rel_errs = []
    for _ in range(24):
        g1 = torch.from_numpy(rng.standard_normal(dim)).float()
        old1 = p1.detach().clone()
        oldg = pg.detach().clone()

        o1.zero_grad()
        p1.grad = g1.clone()
        og.zero_grad()
        pg.grad = (a * g1).clone()
        o1.step()
        og.step()

        u1 = p1.detach() - old1
        ug = pg.detach() - oldg

        st1 = o1.state[p1]
        stg = og.state[pg]
        step = int(st1["step"].item() if torch.is_tensor(st1["step"]) else st1["step"])

        # PyTorch RAdam uses the unrectified branch for the earliest steps.
        # Only test the rectified theorem once that branch has been left.
        if step <= 5:
            continue

        m1, v1 = st1["exp_avg"], st1["exp_avg_sq"]
        mg, vg = stg["exp_avg"], stg["exp_avg_sq"]
        bc1_m = 1.0 - 0.9 ** step
        bc1_v = 1.0 - 0.999 ** step
        mh1, vh1 = m1 / bc1_m, v1 / bc1_v
        mhg, vhg = mg / bc1_m, vg / bc1_v

        support = torch.abs(u1) > TINY
        h = torch.ones_like(u1)
        h[support] = (
            (mhg[support] / (mh1[support] + TINY))
            * torch.sqrt(vh1[support] / (vhg[support] + TINY))
        )

        # The theorem assumes eps=0; real RAdam uses eps=1e-8, so this is a
        # near-identity rather than bit-exact equality.
        pred = h * u1
        rel = float(torch.norm((ug - pred)[support]) / (torch.norm(ug[support]) + TINY))
        rel_errs.append(rel)

    max_err = max(rel_errs)
    print("=== T1: moment-defined coordinate history gauge ===")
    print(f"  max relative error on rectified steps: {max_err:.2e}")
    assert max_err < 1e-5, "moment-defined h should predict rectified updates"
    return max_err


def check_exact_residual_identity(trace, step):
    """T4: verify the support-aware analytic residual and H == R_opt."""
    u1 = trace["u1"][step]
    ug = trace["ug"][step]
    support = torch.abs(u1) > TINY

    h = ug[support] / u1[support]
    w = u1[support].square()
    s_star = float(torch.dot(ug, u1) / (torch.dot(u1, u1) + TINY))

    analytic_sq = torch.sum(w * (h - s_star).square())
    if torch.any(~support):
        analytic_sq = analytic_sq + torch.sum(ug[~support].square())

    direct_sq = torch.sum((ug - s_star * u1).square())
    denom = torch.sum(u1.square())
    H = float(torch.sqrt(analytic_sq / (denom + TINY)))
    R_opt = float(torch.sqrt(direct_sq / (denom + TINY)))

    print("\n=== T4: exact residual / history-dispersion identity ===")
    print(f"  step={step}, s*={s_star:.6f}")
    print(f"  analytic residual^2={float(analytic_sq):.8e}")
    print(f"  direct   residual^2={float(direct_sq):.8e}")
    print(f"  H={H:.8f}, R_opt={R_opt:.8f}")

    assert torch.allclose(analytic_sq, direct_sq, rtol=1e-5, atol=1e-10)
    assert abs(H - R_opt) < 1e-6
    return H, R_opt


def main():
    dim = 64
    n = 80

    # T2: constant a_j = 1.3, which is the rectified constant-scale null.
    const_trace = run_trace(dim, np.full(n, 1.3))
    step_null = 50
    h_std = float(const_trace["h"][step_null].std())
    R_null = float(const_trace["residual"][step_null])
    s_null = float(const_trace["s_star"][step_null])

    print("=== P-R3 coordinate-wise history gauge (rev.4) ===")
    print("\n[T2] constant a_j = 1.3 (rectified null):")
    print(f"  h_k std over coords: {h_std:.2e}")
    print(f"  update residual R_opt: {R_null:.2e}")
    print(f"  update scale s*: {s_null:.6f}  (expect 1.0)")

    assert h_std < 1e-4
    assert R_null < 1e-4
    assert abs(s_null - 1.0) < 1e-3

    # T3: time-varying scalar history. Each instantaneous gradient remains an
    # exact scalar multiple, so any non-scalar update residual is history-made.
    a_alt = np.where(np.arange(n) % 20 < 10, 1.3, 0.8).astype(float)
    alt_trace = run_trace(dim, a_alt)

    print("\n[T3] alternating a_j = 1.3/0.8 (synthetic mechanism example):")
    for k in [11, 19, 21, 40, 59, 61]:
        print(
            f"  step {k:3d}: h std={alt_trace['h'][k].std().item():.4f}  "
            f"R_opt={alt_trace['residual'][k]:.4f}"
        )

    mechanism_peak = float(np.max(alt_trace["residual"][10:]))
    assert mechanism_peak > 0.05, (
        "time-varying heterogeneous history example should create a visible "
        "non-scalar update residual"
    )

    # T4 uses the real U1^2 weights, not uniform placeholder weights.
    check_exact_residual_identity(alt_trace, step=21)
    check_identity_from_moments()

    print("\nALL HISTORY-GAUGE CHECKS PASSED")


if __name__ == "__main__":
    main()

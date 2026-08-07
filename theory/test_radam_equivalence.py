"""Numeric checks for the RAdam gap-equivalence propositions (Role C).

Validates P-R1 (unrectified) and P-R2 (rectified) with the REAL torch.optim.RAdam:

  Setup: two copies of the same small net, gradient of arm-g is exactly
         G_g = a * G_1 (a fixed positive scalar), identical betas/eps/lr,
         fresh state (m_0 = v_0 = 0), step count matched.

  P-R1 (unrectified, early steps): update_u ~= a * update_1, so
         c^* = <u_g, u_1>/||u_1||^2 ~= a,  and  eta_g = eta_1 / a  matches.
  P-R2 (rectified, later steps): the mhat/sqrt(vhat) ratio cancels a, so
         c^* -> 1.

The key testable prediction: a fixed LR multiplier cannot match BOTH phases;
c^* drifts from ~a (early) toward ~1 (later).
"""
import numpy as np
import torch
from torch.optim import RAdam


def _fresh_pair(dim, a, lr, betas=(0.9, 0.999), eps=1e-8):
    """Two fresh RAdam optimizers on a linear-parameter net, gradient G_g = a G_1."""
    p1 = torch.nn.Parameter(torch.zeros(dim))
    p2 = torch.nn.Parameter(torch.zeros(dim))
    opt1 = RAdam([p1], lr=lr, betas=betas, eps=eps)
    opt2 = RAdam([p2], lr=lr, betas=betas, eps=eps)
    # same base gradient each step
    def step(grad1):
        opt1.zero_grad(); p1.grad = grad1.clone()
        opt2.zero_grad(); p2.grad = (a * grad1).clone()
        opt1.step(); opt2.step()
        u1 = p1.grad  # note: grad still set; use state or recompute
        # actual update = -(new - old)
        up1 = -(p1.detach().clone())
        up2 = -(p2.detach().clone())
        return up1, up2
    return p1, p2, opt1, opt2, step


def c_star(ug, u1):
    return float(torch.dot(ug, u1) / (torch.dot(u1, u1) + 1e-30))


def run(seed=0, dim=64, a=1.3, lr=1e-3, n_steps=300):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    p1, p2, opt1, opt2, step = _fresh_pair(dim, a, lr)
    cs = []
    for k in range(n_steps):
        g1 = torch.from_numpy(rng.standard_normal(dim)).float()
        up1, up2 = step(g1)
        cs.append(c_star(up2, up1))
    cs = np.array(cs)
    return cs


def main():
    a = 1.3
    cs = run(a=a)
    print("=== RAdam scalar-equivalence numeric check (a=1.3, lr=1e-3, dim=64) ===")
    print("c* at various steps:")
    for k in [0, 1, 2, 4, 10, 20, 50, 100, 150, 200, 250, 299]:
        print(f"  step {k:3d}: c* = {cs[k]:.4f}")
    early = cs[:5].mean()
    late = cs[250:].mean()
    print(f"\nP-R1 (unrectified, steps 0-4):     mean c* = {early:.4f}  (expect ~ a = {a})")
    print(f"P-R2 (rectified, steps 250-299):    mean c* = {late:.4f}  (expect -> 1)")
    print(f"drift = {late - early:.4f}")

    ok = True
    if abs(early - a) < 0.05:
        print("P-R1 CONFIRMED (early c* ~ a)")
    else:
        print("P-R1 NOT confirmed"); ok = False
    if late < 1.15:
        print("P-R2 CONFIRMED (late c* -> 1)")
    else:
        print("P-R2 NOT confirmed"); ok = False
    if late < early - 0.1:
        print("KEY PREDICTION CONFIRMED: c* drifts from a toward 1 as rectification activates")
    else:
        print("KEY PREDICTION NOT confirmed"); ok = False
    if ok:
        print("\nALL PROPOSITION CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

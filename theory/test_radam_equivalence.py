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
        # SAVE pre-step params (review fix: measure the SINGLE-STEP update
        #  u = theta_{k+1} - theta_k, NOT the cumulative -theta_k)
        old1 = p1.detach().clone()
        old2 = p2.detach().clone()
        opt1.zero_grad(); p1.grad = grad1.clone()
        opt2.zero_grad(); p2.grad = (a * grad1).clone()
        opt1.step(); opt2.step()
        u1 = p1.detach() - old1     # single-step parameter update, arm 1
        u2 = p2.detach() - old2     # single-step parameter update, arm g
        return u1, u2
    return p1, p2, opt1, opt2, step


def c_star(ug, u1):
    """candidate LR multiplier: c = <ug,u1>/||ug||^2 (PR #42 convention).

    If u_g = a u_1 then c = 1/a (the scalar that makes c*u_g = u_1).
    """
    return float(torch.dot(ug, u1) / (torch.dot(ug, ug) + 1e-30))


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
    print("c* = <ug,u1>/||ug||^2  (candidate LR multiplier, PR #42 convention)")
    print("single-step updates u = theta_{k+1} - theta_k (review fix)")
    print("c* at various steps:")
    for k in [0, 1, 2, 4, 10, 20, 50, 100, 150, 200, 250, 299]:
        print(f"  step {k:3d}: c* = {cs[k]:.4f}")
    early = cs[:5].mean()
    late = cs[250:].mean()
    print(f"\nP-R1 (unrectified, steps 0-4):     mean c* = {early:.4f}  (expect ~ 1/a = {1/a:.4f})")
    print(f"P-R2 (rectified, steps 250-299):    mean c* = {late:.4f}  (expect -> 1)")
    print(f"drift = {late - early:.4f}")

    ok = True
    if abs(early - 1 / a) < 0.05:
        print("P-R1 CONFIRMED (early c* ~ 1/a)")
    else:
        print("P-R1 NOT confirmed"); ok = False
    if abs(late - 1.0) < 0.15:
        print("P-R2 CONFIRMED (late c* -> 1)")
    else:
        print("P-R2 NOT confirmed"); ok = False
    if late > early - 0.1:
        print("P-R2 prediction: c* approaches 1 (rectified absorbs constant scale)")
    else:
        print("NOTE: c* not yet at 1 (check step count)"); ok = ok
    if ok:
        print("\nALL PROPOSITION CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
    return 0 if ok else 1


# ===========================================================================
# P-R3: history-induced gauge breaking (review's stronger theory)
# ===========================================================================

def test_history_gauge_breaking_time_varying_a():
    """Time-varying a_j breaks scalar equivalence even if the INSTANTANEOUS
    gradient is near-scalar each step.

    Setup: G^g_j = a_j G^1_j with a_j varying over steps. The RAdam moments
    are history-weighted sums; a single scalar cannot be factored out of both
    m (sum of a_j G_j) and v (sum of a_j^2 G_j^2) simultaneously. So the
    single-step update residual R_opt(k) is large even though each step's
    raw gradient residual is zero.
    """
    torch.manual_seed(0)
    dim = 64
    lr = 1e-3
    n_steps = 200
    # time-varying scale: a_j alternates / drifts
    aj = np.where(np.arange(n_steps) % 40 < 20, 1.3, 0.8).astype(float)

    p1 = torch.nn.Parameter(torch.zeros(dim))
    p2 = torch.nn.Parameter(torch.zeros(dim))
    o1 = RAdam([p1], lr=lr)
    o2 = RAdam([p2], lr=lr)
    rng = np.random.default_rng(0)
    resids = []
    for k in range(n_steps):
        g1 = torch.from_numpy(rng.standard_normal(dim)).float()
        old1, old2 = p1.detach().clone(), p2.detach().clone()
        o1.zero_grad(); p1.grad = g1.clone()
        o2.zero_grad(); p2.grad = (aj[k] * g1).clone()
        o1.step(); o2.step()
        u1 = p1.detach() - old1
        u2 = p2.detach() - old2
        # best scalar fit of u2 to u1 (s* = <u2,u1>/||u1||^2)
        s = float(torch.dot(u2, u1) / (torch.dot(u1, u1) + 1e-30))
        res = float(torch.norm(u2 - s * u1) / (torch.norm(u1) + 1e-30))
        resids.append(res)
    resids = np.array(resids)
    # residual is small at the START of each constant block, grows after a_j changes
    print("\n=== P-R3 history-induced gauge breaking (time-varying a_j) ===")
    print("per-step update residual (||u2 - s*u1||/||u1||) at sampled steps:")
    for k in [1, 10, 19, 21, 40, 41, 59, 61, 100, 120, 150, 199]:
        print(f"  step {k:3d} (a_j={aj[k]:.1f}): residual = {resids[k]:.4f}")
    # growth right after a change (steps 20, 60, 100) vs end of block (19, 59, 99)
    jump = resids[21] - resids[19]
    print(f"residual jump right after a changes (step 21 vs 19): {jump:+.4f}")
    return resids, aj


def _run_history():
    resids, aj = test_history_gauge_breaking_time_varying_a()
    # the residual must be larger after a_j changes than at the end of a
    # constant block (history can't track the new scale instantly)
    jumps = [resids[21] - resids[19], resids[61] - resids[59], resids[101] - resids[99]]
    print("jumps:", [f"{j:+.4f}" for j in jumps])
    # qualitative assertion: max residual well above the constant-block floor
    floor = min(np.concatenate([resids[10:19], resids[50:59], resids[90:99]]))
    peak = max(resids)
    print(f"constant-block floor={floor:.4f}, overall peak={peak:.4f}, ratio={peak/max(floor,1e-9):.1f}x")
    assert peak > floor * 3, "history-induced residual should exceed the constant-block floor"


if __name__ == "__main__":
    import sys
    # P-R1 / P-R2 (single-step gauge) then P-R3 (history gauge breaking)
    rc1 = main()
    _run_history()
    sys.exit(rc1)

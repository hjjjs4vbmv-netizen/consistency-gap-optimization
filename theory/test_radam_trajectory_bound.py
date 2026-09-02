"""Numeric check of the P-R3 trajectory bound mechanism (Role C, afternoon).

Question: does a small single-step residual eps_j get AMPLIFIED along a RAdam
trajectory, so that ||z_K^g - z_K^1|| can exceed the injected residual?

We test two regimes with the REAL torch.optim.RAdam:
  A) convex quadratic (smooth contraction) — expect amplification < 1;
  B) non-convex MLP, early/transient phase — expect possible amplification > 1.

The bound under test:
    ||z_K^g - z_K^1|| <= sum_j ( prod_{ell>j} L_ell ) eps_j.
We report the empirical amplification factor (state-diff / injected-residual)
for a single mid-trajectory injection, which is the checkable shadow of the
bound's amplification term.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.optim import RAdam


def make_quadratic(dim=8, cond=10.0, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    evals = np.geomspace(1.0, cond, dim)
    H = (Q * evals) @ Q.T
    th_star = rng.standard_normal(dim)
    return torch.from_numpy(H).float(), torch.from_numpy(th_star).float()


def run_arm_quad(H, th_star, lr, steps, inject_step=None, inject=None, seed=1, noise=0.01):
    torch.manual_seed(seed)
    th = torch.zeros_like(th_star)
    opt = RAdam([th], lr=lr)
    rng = np.random.default_rng(seed)
    for k in range(steps):
        g = H @ (th - th_star) + torch.from_numpy(rng.standard_normal(th_star.shape)).float() * noise
        opt.zero_grad(); th.grad = g
        opt.step()
        if inject_step is not None and k == inject_step:
            with torch.no_grad():
                th.add_(inject)
    return th.detach().clone()


class MLP(nn.Module):
    def __init__(self, d=8, h=16):
        super().__init__()
        self.fc = nn.Linear(d, h)
        self.out = nn.Linear(h, 1)
    def forward(self, x):
        return self.out(torch.tanh(self.fc(x)))


def run_arm_mlp(dim, lr, steps, inject_step=None, inject=None, seed=1, noise=0.05):
    torch.manual_seed(seed)
    net = MLP(dim)
    opt = RAdam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    xs = torch.from_numpy(rng.standard_normal((64, dim))).float()
    ys = (torch.sin(xs[:, 0:1]) * 0.5).float()
    flat = lambda: torch.cat([p.detach().reshape(-1) for p in net.parameters()])
    for k in range(steps):
        opt.zero_grad()
        out = net(xs)
        # NOTE: no detached regularizer here — a detached 'noise * ||flat()||'
        # term would contribute no gradient; removed to avoid ambiguity.
        loss = ((out - ys) ** 2).mean()
        loss.backward()
        opt.step()
        if inject_step is not None and k == inject_step:
            with torch.no_grad():
                i = 0
                for p in net.parameters():
                    n = p.numel()
                    p.add_(inject[i:i+n].reshape(p.shape))
                    i += n
    return flat().clone()


def amplification(base, pert, inject_norm):
    diff = torch.norm(pert - base).item()
    return diff / inject_norm


def main():
    # --- A: convex quadratic ---
    H, th_star = make_quadratic()
    dim = th_star.shape[0]
    eps = 1e-3
    inject = torch.full((dim,), eps / np.sqrt(dim))
    base = run_arm_quad(H, th_star, 1e-2, 200, seed=3)
    pert = run_arm_quad(H, th_star, 1e-2, 200, inject_step=20, inject=inject, seed=3)
    ampA = amplification(base, pert, torch.norm(inject).item())
    print("=== P-R3 amplification check ===")
    print(f"[A] convex quadratic:  amplification = {ampA:.3f}  (<1 => contraction)")

    # --- B: non-convex MLP, early phase ---
    dim = 8
    eps_n = 1e-3
    inject = torch.full((1, dim * 16 + 16), eps_n / np.sqrt(dim * 16 + 16))[0]
    n_mlp = dim * 16 + 16 + 16 + 1
    inject = torch.full((n_mlp,), eps_n / np.sqrt(n_mlp))
    baseB = run_arm_mlp(dim, 1e-2, 30, seed=5)
    pertB = run_arm_mlp(dim, 1e-2, 30, inject_step=10, inject=inject, seed=5)
    ampB = amplification(baseB, pertB, torch.norm(inject).item())
    print(f"[B] non-convex MLP (early): amplification = {ampB:.3f}  "
          f"({'AMPLIFIED > 1' if ampB > 1.05 else 'not amplified'})")
    print()
    print("Interpretation: the P-R3 bound's product term prod L_ell either")
    print("contracts (convex regime, A) or can amplify (adaptive/transient, B).")
    print("This is the checkable 'small residual -> finite-horizon difference'")


if __name__ == "__main__":
    main()

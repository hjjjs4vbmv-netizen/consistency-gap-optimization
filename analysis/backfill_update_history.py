"""Backfill full per-step update history for an existing real-history sweep.

The original sweep saved only the FINAL-step updates (u1.npy / ug.npy); the
per-step update history (u1_history.npy / ug_history.npy, shape (T,d)) was not
persisted. `scalar_history_predictor.py --eval-step t` needs the per-step update
at the SAME index t — otherwise the 1-step control compares a step-0 predictor
against the step-(T-1) actual (endpoint mismatch).

This script reconstructs the history EXACTLY by replaying RAdam from the real
optimizer state (m0/v0/step0) AND the real parameter values (net params), over
the already-saved paired gradient history.

Why exact: RAdam's per-step update is
    p <- p - lr * r * m_hat / sqrt(v_hat)
where m_hat, v_hat, r are functions of (m, v, step, grad) only — INDEPENDENT of
the current parameter value. Replaying with the SAME (params, m0, v0, step0,
grad_hist) as the sweep therefore reproduces the real optimizer's per-step
updates to FLOAT32 MACHINE PRECISION (the residual is the float32 ulp of the
O(1)-magnitude params, ~2^-23 ~ 1.2e-7). We assert u1_history[-1] ~ u1.npy and
ug_history[-1] ~ ug.npy within that tolerance as the correctness check.

Outputs:
  u1_history.npy     (T, d) float64 reference per-step updates
  ug_history.npy     (T, d) float64 candidate per-step updates
  backfill_meta.json provenance + verification (exact-match flags, max abs diff)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import RAdam

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.scalar_history_predictor import BETA1, BETA2, flatten_opt_state


def flatten_params(net) -> torch.Tensor:
    """Flatten the model's parameters (float32) in net.parameters() order."""
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()])


def replay_exact(grad_hist, params0, m0, v0, step0, lr,
                 betas=(BETA1, BETA2), eps=1e-8):
    """Replay RAdam from the REAL params + moments over a history.

    Returns per-step updates as float64 (d,) arrays, matching the sweep's
    `_update_flat` convention float64(after) - float64(before).
    """
    p = torch.nn.Parameter(params0.clone())
    opt = RAdam([p], lr=lr, betas=betas, eps=eps)
    opt.state[p]["step"] = torch.tensor(step0)
    opt.state[p]["exp_avg"] = m0.clone()
    opt.state[p]["exp_avg_sq"] = v0.clone()
    updates = []
    for g in grad_hist:
        old = p.detach().clone()
        opt.zero_grad()
        p.grad = torch.from_numpy(g).float()
        opt.step()
        updates.append((p.detach().double() - old.double()).numpy())
    return updates


def _verify(name, hist_arr, final_ref_path):
    ref = np.load(final_ref_path)
    diff = np.abs(hist_arr[-1] - ref)
    max_abs = float(np.max(diff))
    exact = bool(np.array_equal(hist_arr[-1], ref))
    # The reconstruction reproduces the real updates up to the float32 ulp of
    # the O(1)-magnitude params (2^-23 ~ 1.2e-7). max_abs < 2^-22 (~2.4e-7)
    # means "exact to float32 rounding" (bit-exact would require matching the
    # exact step-tensor dtype in RAdam's rectification, a ~1-ulp scalar).
    float32_exact = bool(max_abs < 2.4e-7)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = diff / np.maximum(np.abs(ref), 1e-30)
    max_rel = float(np.max(rel[~np.isnan(rel)])) if np.isfinite(rel).any() else float("nan")
    return {"exact": exact, "float32_exact": float32_exact,
            "max_abs_diff": max_abs,
            "max_rel_diff_on_nonzero": max_rel,
            "final_ref_path": str(final_ref_path)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-state", type=Path, required=True)
    ap.add_argument("--grad-history-1", type=Path, required=True)
    ap.add_argument("--grad-history-g", type=Path, required=True)
    ap.add_argument("--u1", type=Path, required=True)
    ap.add_argument("--ug", type=Path, required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    # real state: params + moments (shared by both arms in the sweep)
    data = torch.load(a.training_state, map_location="cpu", weights_only=False)
    m0, v0, step0 = flatten_opt_state(data["optimizer_state"])
    params0 = flatten_params(data["net"])
    dim = params0.numel()
    del data

    G1 = np.load(a.grad_history_1)   # (T, d)
    Gg = np.load(a.grad_history_g)   # (T, d)
    T = G1.shape[0]

    a.out.mkdir(parents=True, exist_ok=True)

    # reference arm (g=1.0)
    u1_hist = replay_exact([G1[j] for j in range(T)], params0, m0, v0, step0, a.lr)
    u1_arr = np.stack(u1_hist)
    np.save(a.out / "u1_history.npy", u1_arr)
    ver_u1 = _verify("u1", u1_arr, a.u1)
    del u1_hist, u1_arr

    # candidate arm (g=1.3)
    ug_hist = replay_exact([Gg[j] for j in range(T)], params0, m0, v0, step0, a.lr)
    ug_arr = np.stack(ug_hist)
    np.save(a.out / "ug_history.npy", ug_arr)
    ver_ug = _verify("ug", ug_arr, a.ug)
    del ug_hist, ug_arr

    meta = {
        "training_state": str(a.training_state),
        "n_steps": T,
        "dim": dim,
        "lr": a.lr,
        "u1_verify": ver_u1,
        "ug_verify": ver_ug,
    }
    (a.out / "backfill_meta.json").write_text(json.dumps(meta, indent=2))

    print("=== backfill per-step update history ===")
    print(f"steps={T}, dim={dim}, lr={a.lr}")
    print(f"u1_history[-1] vs u1.npy: float32_exact={ver_u1['float32_exact']}, "
          f"max_abs={ver_u1['max_abs_diff']:.3e}")
    print(f"ug_history[-1] vs ug.npy: float32_exact={ver_ug['float32_exact']}, "
          f"max_abs={ver_ug['max_abs_diff']:.3e}")
    ok = ver_u1["float32_exact"] and ver_ug["float32_exact"]
    print(("FLOAT32-EXACT" if ok else "MISMATCH"), "- saved to", a.out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

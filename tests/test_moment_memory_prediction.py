"""Self-contained check of the moment-memory prediction pipeline.

Generates a synthetic paired RAdam trajectory (reference arm + candidate arm
with a known per-step δ_j scale), records each step's gradients, runs one
paired step at the end to get the actual update ratio h^update, and verifies
the predicted ĥ from the δ history:

  1. the pipeline's ĥ (from δ_j history, no moments) should correlate strongly
     with h^update;
  2. weighted RMSE(ĥ, h^update) should be small;
  3. Disp(ĥ) should track R_opt (the update residual).

This is the exact chain the real audit needs; here it is validated end-to-end
on a controlled example (no server / no real checkpoint required).
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import RAdam

SCRIPT = Path(__file__).resolve().parents[1] / "analysis" / "moment_memory_prediction.py"
SPEC = importlib.util.spec_from_file_location("moment_memory_prediction", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def run_paired(dim=64, n_steps=60, delta_blocks=(0.3, -0.2), seed=0, lr=1e-3):
    """Paired RAdam: reference arm G, candidate arm G^g = (1+δ_j)G.

    Records per-step gradients; at the last step also records the actual
    single-step updates u1, ug (for h^update). Returns the gradient histories,
    the updates, and the δ schedule.
    """
    torch.manual_seed(seed)
    p1 = torch.nn.Parameter(torch.zeros(dim))
    pg = torch.nn.Parameter(torch.zeros(dim))
    o1 = RAdam([p1], lr=lr)
    og = RAdam([pg], lr=lr)
    rng = np.random.default_rng(seed)
    block = len(delta_blocks)
    sched = np.array([delta_blocks[j % block] for j in range(n_steps)])

    G1_hist = []
    Gg_hist = []
    for k in range(n_steps):
        g1 = torch.from_numpy(rng.standard_normal(dim)).float()
        G1_hist.append(g1.detach().numpy().copy())
        old1, oldg = p1.detach().clone(), pg.detach().clone()
        o1.zero_grad(); p1.grad = g1.clone()
        og.zero_grad(); pg.grad = ((1 + sched[k]) * g1).clone()
        Gg_hist.append(pg.grad.detach().numpy().copy())
        o1.step(); og.step()
        if k == n_steps - 1:
            u1 = p1.detach() - old1
            ug = pg.detach() - oldg
    return (np.stack(G1_hist), np.stack(Gg_hist), sched,
            u1.detach().numpy(), ug.detach().numpy())


def test_pipeline_predicts_actual_h():
    G1, Gg, sched, u1, ug = run_paired(dim=64, n_steps=60, seed=0)
    T = G1.shape[0]
    t = T - 1
    g1 = [G1[j] for j in range(t + 1)]
    gg = [Gg[j] for j in range(t + 1)]
    A1, A2, B2, dhist = M.moment_memory_terms(g1, gg, t)
    h_pred = M.predict_h(A1, A2, B2)
    h_act = M.actual_update_h(u1, ug)
    w = u1 ** 2

    rmse = M.weighted_rmse(h_pred, h_act, w)
    r = M.corr(h_pred, h_act, w)
    disp = M.dispersion(h_pred, w)
    s_opt = float(np.sum(ug * u1) / max(np.sum(u1 * u1), 1e-30))
    R_opt = float(np.linalg.norm(ug - s_opt * u1) / max(np.linalg.norm(u1), 1e-30))

    print("=== test_moment_memory_prediction ===")
    print(f"δ_j recovered: mean={dhist.mean():.4f}, std={dhist.std():.4f} "
          f"(schedule blocks {sched[:10].tolist()[:2]})")
    print(f"weighted RMSE(ĥ, h^actual) = {rmse:.4e}")
    print(f"Corr(ĥ, h^actual)         = {r:.4f}")
    print(f"Disp(ĥ) = {disp:.4f}, R_opt = {R_opt:.4f}, ratio = {disp/R_opt:.4f}")

    # The prediction should be accurate (this is the same construction the
    # theorem is exact for): RMSE small, correlation high.
    assert rmse < 0.3, f"prediction RMSE too large: {rmse}"
    assert r > 0.9, f"prediction should correlate with actual h: {r}"
    # Dispersion should track R_opt (both ~ the same residual scale).
    assert abs(disp / R_opt - 1.0) < 0.5, f"Disp/R_opt={disp/R_opt:.3f} should be ~1"
    print("ALL MOMENT-MEMORY PREDICTION CHECKS PASSED")


if __name__ == "__main__":
    test_pipeline_predicts_actual_h()

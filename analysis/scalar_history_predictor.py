"""Scalar-History Predictor: how much of the real optimizer residual is
explained by a scalar gradient-scale history through RAdam moment memory?

This is the SCIENTIFIC mechanism test (vs the coordinate-wise algebraic sanity check, which is
only an algebra/implementation sanity check).

Protocol (per step j, from the stored paired gradient history):
  1. global scalar  a_j* = <G_j^1.3, G_j^1.0> / ||G_j^1.0||^2   (ONE scalar)
  2. scalar-predicted gradient  Ĝ_j^1.3 = a_j* G_j^1.0
  3. replay RAdam from the REAL optimizer state (m,v,step) with two histories:
       reference:  G_j^1.0
       scalar:     Ĝ_j^1.3
     -> two virtual optimizer trajectories
  4. at target step t, the predicted update ratio
       ĥ^scalar_{t,i} = U^scalar_{t,i} / U^1_{t,i}
     vs the real observed update ratio h^actual_{t,i} = U^1.3_{t,i}/U^1_{t,i}

Outputs:
  wRMSE(ĥ^scalar, h^actual)
  Corr(ĥ^scalar, h^actual)
  Weighted R²(ĥ^scalar, h^actual)       -- the statistically meaningful "explained variance"
  ρ_scalar = Disp(ĥ^scalar) / R_opt     -- dispersion RATIO only, NOT an explained fraction
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import RAdam

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BETA1, BETA2 = 0.9, 0.999


def global_scalar(grad_g: np.ndarray, grad_1: np.ndarray) -> float:
    """ONE global scalar: a* = <Gg,G1>/||G1||²."""
    denom = float(np.sum(grad_1 * grad_1))
    if denom <= 0:
        return 1.0
    return float(np.sum(grad_g * grad_1) / denom)


def flatten_opt_state(opt_state: dict) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Flatten the real optimizer m/v/step across all params into (d,) tensors."""
    ms, vs = [], []
    step = 0
    for st in opt_state["state"].values():
        ms.append(st["exp_avg"].reshape(-1))
        vs.append(st["exp_avg_sq"].reshape(-1))
        step = max(step, int(st["step"]))
    return torch.cat(ms), torch.cat(vs), step


def replay_from_state(grad_hist: list[np.ndarray], dim: int, m0: torch.Tensor,
                      v0: torch.Tensor, step0: int, lr: float,
                      betas=(BETA1, BETA2), eps: float = 1e-8):
    """Replay RAdam from a given (flattened) optimizer state over a history.

    Returns per-step parameter updates. The real m/v/step are set manually on a
    dummy parameter of shape (dim,), so the moments start from the real state.
    """
    p = torch.nn.Parameter(torch.zeros(dim))
    opt = RAdam([p], lr=lr, betas=betas, eps=eps)
    # manually seed the optimizer state (RAdam state is lazy-created)
    opt.state[p]["step"] = torch.tensor(step0)
    opt.state[p]["exp_avg"] = m0.clone()
    opt.state[p]["exp_avg_sq"] = v0.clone()
    updates = []
    for g in grad_hist:
        old = p.detach().clone()
        opt.zero_grad()
        p.grad = torch.from_numpy(g).float()
        opt.step()
        updates.append((p.detach() - old).numpy())
    return updates


def weighted_rmse(h_pred, h_act, w):
    sup = w > 0
    if not sup.any():
        return math.nan
    return math.sqrt(float(np.sum(w[sup] * (h_pred[sup] - h_act[sup]) ** 2) / np.sum(w[sup])))


def corr(h_pred, h_act, w):
    sup = w > 0
    if sup.sum() < 2:
        return math.nan
    wp = w[sup]; x, y = h_pred[sup], h_act[sup]
    xm = np.sum(wp * x) / np.sum(wp); ym = np.sum(wp * y) / np.sum(wp)
    cov = np.sum(wp * (x - xm) * (y - ym))
    vx = np.sum(wp * (x - xm) ** 2); vy = np.sum(wp * (y - ym) ** 2)
    if vx <= 0 or vy <= 0:
        return math.nan
    return float(cov / math.sqrt(vx * vy))


def dispersion(h, w):
    sup = w > 0
    if not sup.any():
        return math.nan
    wp = w[sup]
    m = np.sum(wp * h[sup]) / np.sum(wp)
    return math.sqrt(float(np.sum(wp * (h[sup] - m) ** 2) / np.sum(wp)))


def weighted_r2(h_pred, h_act, w):
    """Weighted coefficient of determination of h_pred against h_actual.

    R2 = 1 - Σ w (h_act - h_pred)² / Σ w (h_act - ĥ_act)²
    where ĥ_act is the weighted mean of h_actual. Measures the fraction of the
    WEIGHTED VARIANCE of h_actual explained by the predictor (this is the
    statistically meaningful "explained" metric, not Disp(h_pred)/R_opt).
    """
    sup = w > 0
    if not sup.any():
        return math.nan
    wp = w[sup]; x = h_pred[sup]; y = h_act[sup]
    wsum = float(np.sum(wp))
    ybar = float(np.sum(wp * y) / wsum)
    ss_res = float(np.sum(wp * (y - x) ** 2))
    ss_tot = float(np.sum(wp * (y - ybar) ** 2))
    if ss_tot <= 0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def select_eval_update(u1_history, ug_history, u1, ug, t, T):
    """Return (u1_t, ug_t, u1_src, ug_src) at eval step t.

    If a full per-step history is provided (u1_history/ug_history, shape (T,d)),
    use the update at index t so the actual update and the predictor are
    evaluated at the SAME step. Otherwise fall back to the single final-step
    update and warn if t != T-1 (the endpoint mismatch this helper exists to
    prevent).
    """
    if u1_history is not None and ug_history is not None:
        u1_t = np.load(u1_history)[t]
        ug_t = np.load(ug_history)[t]
        return u1_t, ug_t, u1_history, ug_history
    u1_t = np.load(u1)
    ug_t = np.load(ug)
    if t != T - 1:
        print(f"[WARNING] eval_step={t} but no --u1-history/--ug-history: "
              f"actual update is step {T-1}, predictor is step {t} "
              f"(endpoint mismatch). Pass --u1-history/--ug-history to fix.")
    return u1_t, ug_t, u1, ug


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-state", type=Path, required=True,
                    help="real training-state (for the optimizer m/v/step)")
    ap.add_argument("--grad-history-1", type=Path, required=True)
    ap.add_argument("--grad-history-g", type=Path, required=True)
    ap.add_argument("--u1", type=Path, required=True)
    ap.add_argument("--ug", type=Path, required=True)
    ap.add_argument("--u1-history", type=Path, default=None,
                    help="(T,d) full per-step reference updates; overrides --u1/--ug at eval_step")
    ap.add_argument("--ug-history", type=Path, default=None,
                    help="(T,d) full per-step candidate updates; overrides --u1/--ug at eval_step")
    ap.add_argument("--eval-step", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260809,
                    help="seed used to generate the paired gradient history (sweep)")
    ap.add_argument("--out", type=Path, default=Path("analysis/scalar_history_prediction.json"))
    ap.add_argument("--zero-initial", action="store_true",
                    help="zero-m0 control: replay from m0=0, v0=0, step=0 (control for R² source)")
    a = ap.parse_args(argv)

    # load real optimizer state, flatten m/v/step
    data = torch.load(a.training_state, map_location="cpu", weights_only=False)
    opt_state = data["optimizer_state"]
    m0, v0, step0 = flatten_opt_state(opt_state)
    dim = m0.numel()

    if a.zero_initial:
        m0 = torch.zeros(dim)
        v0 = torch.zeros(dim)
        step0 = 0
        print("[zero-initial control] m0=v0=0, step=0")

    G1 = np.load(a.grad_history_1)   # (T, d)
    Gg = np.load(a.grad_history_g)   # (T, d)
    if G1.ndim != 2 or Gg.ndim != 2 or G1.shape != Gg.shape or G1.shape[0] < 1:
        raise SystemExit("gradient histories must be non-empty, shape-matched (T, d) arrays")
    T = G1.shape[0]
    t = T - 1 if a.eval_step < 0 else min(a.eval_step, T - 1)
    d = G1.shape[1]
    if d != dim:
        raise SystemExit(f"gradient-history dimension {d} does not match source optimizer dimension {dim}")
    if a.u1_history is None or a.ug_history is None:
        history_mode = False
    else:
        u1_history = np.load(a.u1_history, mmap_mode="r")
        ug_history = np.load(a.ug_history, mmap_mode="r")
        if u1_history.shape != (T, d) or ug_history.shape != (T, d):
            raise SystemExit("full update histories must both have shape (T, d) matching gradient histories")
        history_mode = True

    # actual update at the SAME eval step t (per-step history when available)
    u1, ug, u1_src, ug_src = select_eval_update(
        a.u1_history, a.ug_history, a.u1, a.ug, t, T)

    # per-step global scalar
    a_star = [global_scalar(Gg[j], G1[j]) for j in range(t + 1)]
    # scalar-predicted gradient history
    Ghat = np.stack([a_star[j] * G1[j] for j in range(t + 1)])

    # replay from the REAL optimizer state (flattened m/v/step)
    upd_1 = replay_from_state([G1[j] for j in range(t + 1)], d, m0, v0, step0, a.lr)
    upd_scalar = replay_from_state([Ghat[j] for j in range(t + 1)], d, m0, v0, step0, a.lr)

    # predicted update ratio at step t
    u1_pred = upd_1[t]; us_pred = upd_scalar[t]
    h_pred = np.ones(d)
    sup = np.abs(u1_pred) > 1e-30
    h_pred[sup] = us_pred[sup] / u1_pred[sup]

    # actual update ratio
    h_act = np.ones(d)
    sup_act = np.abs(u1) > 1e-30
    h_act[sup_act] = ug[sup_act] / u1[sup_act]

    # weights: reference update energy, on the effective support
    w = u1 ** 2
    eff = (np.abs(u1) > 1e-5) & (np.abs(u1_pred) > 1e-5)
    w_eff = w[eff]

    rmse = weighted_rmse(h_pred[eff], h_act[eff], w_eff)
    r = corr(h_pred[eff], h_act[eff], w_eff)
    disp = dispersion(h_pred[eff], w_eff)
    r2 = weighted_r2(h_pred[eff], h_act[eff], w_eff)
    s_opt = float(np.sum(ug * u1) / max(np.sum(u1 * u1), 1e-30))
    R_opt = float(np.linalg.norm(ug - s_opt * u1) / max(np.linalg.norm(u1), 1e-30))
    rho = disp / R_opt if R_opt > 1e-12 else math.nan

    result = {
        "T_steps": T, "n_steps": T, "eval_step": t, "seed": a.seed,
        "a_star_mean": float(np.mean(a_star)),
        "a_star_std": float(np.std(a_star)),
        "h_pred_scalar_mean": float(np.mean(h_pred[eff])),
        "h_actual_mean": float(np.mean(h_act[eff])),
        "weighted_RMSE_scalar_vs_actual": rmse,
        "corr_scalar_vs_actual": r,
        "weighted_R2_scalar_vs_actual": r2,
        "Disp_h_scalar": disp,
        "R_opt": R_opt,
        "disp_ratio_rho_scalar": rho,   # Disp(ĥ)/R_opt — dispersion RATIO, NOT explained fraction
        "effective_coords": int(eff.sum()),
        # provenance
        "source_state_sha256": sha256_file(a.training_state),
        "grad_history_1_sha256": sha256_file(a.grad_history_1),
        "grad_history_g_sha256": sha256_file(a.grad_history_g),
        "u1_sha256": sha256_file(u1_src),
        "ug_sha256": sha256_file(ug_src),
        "u1_history_sha256": sha256_file(a.u1_history) if history_mode else None,
        "ug_history_sha256": sha256_file(a.ug_history) if history_mode else None,
        "update_source": "history" if history_mode else "final-step",
        "execution_command": " ".join(sys.argv),
        "lr": a.lr,
        "source_nimg": data.get("cur_nimg"),
        "source_optimizer_steps": int(step0),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    # persist per-coordinate data on the effective support so R² / Corr / wRMSE
    # can be independently recomputed (auditability)
    np.save(a.out.parent / "h_pred_scalar.npy", h_pred[eff])
    np.save(a.out.parent / "h_actual.npy", h_act[eff])
    np.save(a.out.parent / "weights.npy", w[eff])

    print("=== Scalar-History Predictor (mechanism test) ===")
    print(f"steps={T}, eval t={t}, effective coords={eff.sum()}")
    print(f"a_j*: mean={result['a_star_mean']:.4f}, std={result['a_star_std']:.4f}")
    print(f"ĥ^scalar mean={result['h_pred_scalar_mean']:.4f}, h^actual mean={result['h_actual_mean']:.4f}")
    print(f"wRMSE(ĥ^scalar, h^actual) = {rmse:.4f}")
    print(f"Corr(ĥ^scalar, h^actual) = {r:.4f}")
    print(f"Weighted R²(ĥ^scalar vs h^actual) = {r2:.4f}   (fraction of h^actual weighted variance explained)")
    print(f"Disp(ĥ^scalar) = {disp:.4f}, R_opt = {R_opt:.4f}")
    print(f"ρ_scalar = Disp/R_opt = {rho:.4f}   (dispersion RATIO, NOT an explained fraction)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

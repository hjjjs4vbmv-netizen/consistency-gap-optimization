"""Unified horizon sweep: how the scalar-history predictor's explanatory power
decays with the trajectory horizon t.

For each eval horizon t in {0,1,4,9,19} (= 1,2,5,10,20 steps) on the real K=256
g=1.0 vs g=1.3 paired history, replay RAdam from the REAL state over [0..t] with
the scalar-predicted 1.3 gradient history and report, at the SAME step t:

    R²(t)       weighted explained variance of h^actual by ĥ^scalar
    Corr(t)     weighted correlation
    wRMSE(t)    weighted RMSE
    R_opt(t)    reference-normalized update residual  ||u_g - s u_1|| / ||u_1||
    cosine(t)   matching quality of the actual paired updates  <u1,ug>/(||u1||||ug||)

This is a single-pass run: each large (T,d) array is loaded once and reused
across all horizons; only the per-horizon RAdam replay is rebuilt (then
discarded). Same state / seed / gradient history as the 20-step headline.

Outputs a JSON table + per-coordinate npy per horizon (for R² auditability).

NOTE: this reports the mechanism quantity only (scalar-history explanatory
power). It does NOT extend to FID / generation-quality causality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.scalar_history_predictor import (
    global_scalar,
    replay_from_state,
    flatten_opt_state,
    weighted_r2,
    corr,
    weighted_rmse,
    dispersion,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cosine(u1: np.ndarray, ug: np.ndarray) -> float:
    d1 = float(np.linalg.norm(u1))
    dg = float(np.linalg.norm(ug))
    if d1 <= 0 or dg <= 0:
        return math.nan
    return float(np.sum(u1 * ug) / (d1 * dg))


def horizon_metrics(G1: np.ndarray, Gg: np.ndarray,
                    u1: np.ndarray, ug: np.ndarray,
                    m0: torch.Tensor, v0: torch.Tensor, step0: int,
                    lr: float, t: int) -> dict:
    d = G1.shape[1]
    a_star = [global_scalar(Gg[j], G1[j]) for j in range(t + 1)]
    Ghat = np.stack([a_star[j] * G1[j] for j in range(t + 1)])

    upd_1 = replay_from_state([G1[j] for j in range(t + 1)], d, m0, v0, step0, lr)
    upd_scalar = replay_from_state([Ghat[j] for j in range(t + 1)], d, m0, v0, step0, lr)

    # predicted update ratio at step t
    u1_pred = upd_1[t]
    us_pred = upd_scalar[t]
    h_pred = np.ones(d)
    sup = np.abs(u1_pred) > 1e-30
    h_pred[sup] = us_pred[sup] / u1_pred[sup]

    # actual update ratio at the SAME step t
    h_act = np.ones(d)
    sup_act = np.abs(u1) > 1e-30
    h_act[sup_act] = ug[sup_act] / u1[sup_act]

    w = u1 ** 2
    eff = (np.abs(u1) > 1e-5) & (np.abs(u1_pred) > 1e-5)
    w_eff = w[eff]

    rmse = weighted_rmse(h_pred[eff], h_act[eff], w_eff)
    r = corr(h_pred[eff], h_act[eff], w_eff)
    disp = dispersion(h_pred[eff], w_eff)
    r2 = weighted_r2(h_pred[eff], h_act[eff], w_eff)
    s_opt = float(np.sum(ug * u1) / max(np.sum(u1 * u1), 1e-30))
    R_opt = float(np.linalg.norm(ug - s_opt * u1) / max(np.linalg.norm(u1), 1e-30))
    cos = cosine(u1, ug)

    return {
        "horizon_steps": t + 1,
        "eval_step": t,
        "weighted_R2": r2,
        "corr": r,
        "wRMSE": rmse,
        "R_opt": R_opt,            # reference-normalized residual norm
        "cosine": cos,             # matching quality of the actual paired updates
        "Disp_h_scalar": disp,
        "a_star_mean": float(np.mean(a_star)),
        "effective_coords": int(eff.sum()),
    }, h_pred[eff], h_act[eff], w_eff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--training-state", type=Path, required=True)
    ap.add_argument("--grad-history-1", type=Path, required=True)
    ap.add_argument("--grad-history-g", type=Path, required=True)
    ap.add_argument("--u1-history", type=Path, required=True)
    ap.add_argument("--ug-history", type=Path, required=True)
    ap.add_argument("--horizons", type=str, default="0,1,4,9,19",
                    help="comma-separated eval steps (0-based); 20 steps total")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20260809)
    ap.add_argument("--out", type=Path, default=Path("analysis/real_history/k256/horizon_sweep.json"))
    a = ap.parse_args(argv)

    ts = [int(x) for x in a.horizons.split(",") if x.strip()]

    data = torch.load(a.training_state, map_location="cpu", weights_only=False)
    opt_state = data["optimizer_state"]
    m0, v0, step0 = flatten_opt_state(opt_state)

    print(f"[horizon_sweep] loading {a.grad_history_1} ...")
    G1 = np.load(a.grad_history_1)
    Gg = np.load(a.grad_history_g)
    T = G1.shape[0]
    d = G1.shape[1]
    print(f"[horizon_sweep] loaded T={T}, d={d}")

    # the update histories are (T,d) and huge; only the requested eval rows are
    # needed, so read them via mmap and keep just those rows in memory
    u1_mm = np.load(a.u1_history, mmap_mode="r")
    ug_mm = np.load(a.ug_history, mmap_mode="r")
    ts = [min(t, T - 1) for t in ts]
    u1_rows = {t: np.array(u1_mm[t]) for t in ts}
    ug_rows = {t: np.array(ug_mm[t]) for t in ts}

    rows = []
    for t in ts:
        m, h_pred, h_act, w = horizon_metrics(
            G1, Gg, u1_rows[t], ug_rows[t], m0, v0, step0, a.lr, t)
        rows.append(m)
        # persist per-coordinate arrays for R² auditability
        out_dir = a.out.parent / f"horizon_t{t}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "h_pred_scalar.npy", h_pred)
        np.save(out_dir / "h_actual.npy", h_act)
        np.save(out_dir / "weights.npy", w)
        print(f"  t={t} (h={t+1}): R2={m['weighted_R2']:.4f} "
              f"Corr={m['corr']:.4f} R_opt={m['R_opt']:.4f} cos={m['cosine']:.4f}")

    result = {
        "training_state": str(a.training_state),
        "grad_history_1": str(a.grad_history_1),
        "grad_history_g": str(a.grad_history_g),
        "u1_history": str(a.u1_history),
        "ug_history": str(a.ug_history),
        "seed": a.seed,
        "lr": a.lr,
        "T_steps": T,
        "source_state_sha256": sha256_file(a.training_state),
        "source_nimg": data.get("cur_nimg"),
        "source_optimizer_steps": int(step0),
        "horizons": rows,
        "execution_command": " ".join(sys.argv),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

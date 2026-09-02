"""Cross-K moment-memory: replicate the #47 scalar-history predictor at four
training stages K in {32,64,128,256} kimg and sweep the prospective horizon
h = 1..T, producing the R^2(K,h) matrix.

Protocol (identical to #47, only K varies):
  per step j: global scalar  a_j* = <G_j^1.3, G_j^1.0> / ||G_j^1.0||^2
              scalar-predicted gradient  Ĝ_j^1.3 = a_j* G_j^1.0
  replay RAdam (float32, pure numpy) from the REAL state (m0/v0/step0) over the
  FULL 20-step history once, reading the update at EVERY index t, so we get the
  scalar predictor's ĥ^scalar_t and the real ĥ_actual_t for all horizons t:
      u1_hist = replay(G1)          reference updates     (== stored u1_history)
      ug_hist = replay(Gg)          real 1.3 updates      (== stored ug_history)
      us_hist = replay(Ghat)        scalar-predicted 1.3 updates
      ĥ_actual_t = ug_hist[t]/u1_hist[t]
      ĥ_scalar_t = us_hist[t]/u1_hist[t]
  metrics per horizon t (weighted by w = u1_hist[t]^2 on the effective support):
      Weighted R^2, Corr, wRMSE, R_opt (normalized residual), cosine(u1,ug)

Validation / provenance per K:
  - u1_hist[-1] vs stored u1.npy   (float32_exact ~2^-23)
  - ug_hist[-1] vs stored ug.npy
  - for k256: full u1_history.npy/ug_history.npy vs the stored torch-generated
    history (the #47 sanity anchor)

The real states come from the #49-frozen same seed-3 Arm-A trajectory:
  K=32->000001, 64->000002, 128->000004, 256->000008 (arm_a_g1_0_lr_fixed_s3).
m0/v0/step0 are provided as extracted numpy files (extracted by a working torch,
since the matpool node's own python lacks a torch-capable glibc).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from analysis.numpy_radam import replay_numpy

REPO_ROOT = Path(__file__).resolve().parents[1]


def global_scalar(grad_g: np.ndarray, grad_1: np.ndarray) -> float:
    denom = float(np.sum(grad_1 * grad_1))
    if denom <= 0:
        return 1.0
    return float(np.sum(grad_g * grad_1) / denom)


def weighted_r2(h_pred, h_act, w):
    sup = w > 0
    if not sup.any():
        return math.nan
    wp = w[sup]; x = h_pred[sup]; y = h_act[sup]
    wsum = float(np.sum(wp)); ybar = float(np.sum(wp * y) / wsum)
    ss_res = float(np.sum(wp * (y - x) ** 2)); ss_tot = float(np.sum(wp * (y - ybar) ** 2))
    if ss_tot <= 0:
        return math.nan
    return 1.0 - ss_res / ss_tot


def corr(h_pred, h_act, w):
    sup = w > 0
    if sup.sum() < 2:
        return math.nan
    wp = w[sup]; x = h_pred[sup]; y = h_act[sup]
    xm = float(np.sum(wp * x) / np.sum(wp)); ym = float(np.sum(wp * y) / np.sum(wp))
    cov = float(np.sum(wp * (x - xm) * (y - ym)))
    vx = float(np.sum(wp * (x - xm) ** 2)); vy = float(np.sum(wp * (y - ym) ** 2))
    if vx <= 0 or vy <= 0:
        return math.nan
    return cov / math.sqrt(vx * vy)


def weighted_rmse(h_pred, h_act, w):
    sup = w > 0
    if not sup.any():
        return math.nan
    return math.sqrt(float(np.sum(w[sup] * (h_pred[sup] - h_act[sup]) ** 2) / np.sum(w[sup])))


def cosine(u1, ug):
    d1 = float(np.linalg.norm(u1)); dg = float(np.linalg.norm(ug))
    if d1 <= 0 or dg <= 0:
        return math.nan
    return float(np.sum(u1 * ug) / (d1 * dg))


def _verify_hist(hist_f32, final_ref_path, name):
    ref = np.load(final_ref_path)
    diff = np.abs(hist_f32[-1].astype(np.float64) - ref.astype(np.float64))
    max_abs = float(np.max(diff))
    float32_exact = bool(max_abs < 2.4e-7)
    return {"name": name, "float32_exact": float32_exact, "max_abs_diff": max_abs,
            "final_ref_path": str(final_ref_path)}


def process_k(kdir: Path, state_np: dict, lr: float, max_h: int):
    G1 = np.load(kdir / "grad_history_1.npy")     # (T,d) float64
    Gg = np.load(kdir / "grad_history_g.npy")     # (T,d) float64
    T, d = G1.shape
    H = min(T, max_h)

    # per-step global scalar (float64, matching #47), then scalar-predicted hist
    a_star = [global_scalar(Gg[j], G1[j]) for j in range(T)]
    Ghat_f64 = np.stack([a_star[j] * G1[j] for j in range(T)])

    m0 = np.load(state_np["m0"], mmap_mode=None).astype(np.float32)
    v0 = np.load(state_np["v0"], mmap_mode=None).astype(np.float32)
    step0 = int(state_np["step0"])

    u1_hist = replay_numpy(G1[:H].astype(np.float32), m0, v0, step0, lr)
    ug_hist = replay_numpy(Gg[:H].astype(np.float32), m0, v0, step0, lr)
    us_hist = replay_numpy(Ghat_f64[:H].astype(np.float32), m0, v0, step0, lr)

    # validation vs stored final updates
    ver_u1 = _verify_hist(u1_hist, kdir / "u1.npy", "u1")
    ver_ug = _verify_hist(ug_hist, kdir / "ug.npy", "ug")

    # strongest check: full-history validation against the stored torch-generated
    # history (only k256 has it) — confirms numpy RAdam == torch RAdam over all steps
    ver_u1_full = ver_ug_full = None
    if (kdir / "u1_history.npy").exists() and (kdir / "ug_history.npy").exists():
        ref1 = np.load(kdir / "u1_history.npy")
        refg = np.load(kdir / "ug_history.npy")
        d1 = np.abs(np.stack([x.astype(np.float64) for x in u1_hist]) - ref1.astype(np.float64))
        dg = np.abs(np.stack([x.astype(np.float64) for x in ug_hist]) - refg.astype(np.float64))
        ver_u1_full = {"float32_exact": bool(np.max(d1) < 2.4e-7), "max_abs_diff": float(np.max(d1))}
        ver_ug_full = {"float32_exact": bool(np.max(dg) < 2.4e-7), "max_abs_diff": float(np.max(dg))}

    horizons = []
    for t in range(H):
        u1 = u1_hist[t].astype(np.float64)
        ug = ug_hist[t].astype(np.float64)
        us = us_hist[t].astype(np.float64)
        h_act = np.ones(d); s = np.abs(u1) > 1e-30
        h_act[s] = ug[s] / u1[s]
        h_pred = np.ones(d); sp = np.abs(u1) > 1e-30
        h_pred[sp] = us[sp] / u1[sp]
        w = u1 ** 2
        eff = (np.abs(u1) > 1e-5) & (np.abs(u1_hist[t]) > 1e-5)
        w_eff = w[eff]
        r2 = weighted_r2(h_pred[eff], h_act[eff], w_eff)
        r = corr(h_pred[eff], h_act[eff], w_eff)
        rmse = weighted_rmse(h_pred[eff], h_act[eff], w_eff)
        s_opt = float(np.sum(ug * u1) / max(np.sum(u1 * u1), 1e-30))
        R_opt = float(np.linalg.norm(ug - s_opt * u1) / max(np.linalg.norm(u1), 1e-30))
        cos = cosine(u1, ug)
        horizons.append({
            "horizon_steps": t + 1, "eval_step": t,
            "weighted_R2": r2, "corr": r, "wRMSE": rmse,
            "R_opt": R_opt, "cosine": cos,
            "a_star_mean": float(np.mean(a_star[:t + 1])),
            "a_star_std": float(np.std(a_star[:t + 1])),
            "h_pred_mean": float(np.mean(h_pred[eff])),
            "h_actual_mean": float(np.mean(h_act[eff])),
            "effective_coords": int(eff.sum()),
        })

    # persist raw per-coordinate predictions at the headline horizon (h=20) so
    # R²/Corr/scatter can be independently recomputed (auditability). ~0.5GB/K.
    raw = kdir / "raw_predictions"
    if H == T and len(horizons) == T:
        t20 = H - 1
        u1 = u1_hist[t20].astype(np.float64)
        ug = ug_hist[t20].astype(np.float64)
        us = us_hist[t20].astype(np.float64)
        h_act = np.ones(d); s = np.abs(u1) > 1e-30
        h_act[s] = ug[s] / u1[s]
        h_pred = np.ones(d); sp = np.abs(u1) > 1e-30
        h_pred[sp] = us[sp] / u1[sp]
        w = u1 ** 2
        eff = (np.abs(u1) > 1e-5) & (np.abs(u1_hist[t20]) > 1e-5)
        raw.mkdir(parents=True, exist_ok=True)
        np.save(raw / "h_pred_scalar_h20.npy", h_pred[eff])
        np.save(raw / "h_actual_h20.npy", h_act[eff])
        np.save(raw / "weights_h20.npy", w[eff])
        np.save(raw / "a_star_series.npy", np.asarray(a_star, dtype=np.float64))
    return {
        "T_steps": T, "dim": d, "a_star_mean": float(np.mean(a_star)),
        "a_star_std": float(np.std(a_star)),
        "verify_u1_final": ver_u1, "verify_ug_final": ver_ug,
        "verify_u1_full_torch": ver_u1_full, "verify_ug_full_torch": ver_ug_full,
        "horizons": horizons,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("analysis/real_history"))
    ap.add_argument("--state-np", type=str, required=True,
                    help="JSON mapping K-label -> {m0, v0, step0} npy/json paths")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-h", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("analysis/crossk_scalar_history/summary.json"))
    a = ap.parse_args(argv)

    state_map = json.loads(Path(a.state_np).read_text())
    results = {}
    for label in ["k32", "k64", "k128", "k256"]:
        kdir = a.base / label
        if not (kdir / "grad_history_1.npy").exists():
            print(f"skip {label}: no grad history")
            continue
        print(f"[crossk] processing {label} ...")
        st = state_map[label]
        res = process_k(kdir, st, a.lr, a.max_h)
        results[label] = res
        h20 = [h for h in res["horizons"] if h["horizon_steps"] == 20][0]
        print(f"  {label}: h=20 R2={h20['weighted_R2']:.4f} Corr={h20['corr']:.4f} "
              f"R_opt={h20['R_opt']:.4f} a*={res['a_star_mean']:.4f}±{res['a_star_std']:.4f} "
              f"[u1 f32exact={res['verify_u1_final']['float32_exact']}]")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

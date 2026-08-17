"""Three-predictor comparison (P1): how much explanatory power does our discrete
finite-history replay provide over the generic continuous first-order theory?

Predictors of the update ratio h_{t,i} = U^g_{t,i}/U_{t,i} (g=1.3 vs g=1.0):

  1. current scalar predictor (cross-K):
       a*_j = <G^g_j, G_j>/||G_j||^2 ;  Ghat_j = a*_j G_j ;  replay RAdam
       h^scalar_t = replay(Ghat)/u1_t
  2. 2026 first-order scale-lag predictor (arXiv:2601.21739, "Why Adam Works
     Better with beta1=beta2"; first-order moment-memory expansion):
       delta_j = a*_j - 1
       A^(1)_t = (Σ_{j<=t} β1^{t-j} δ_j G_j)/(Σ_{j<=t} β1^{t-j} G_j)
       A^(2)_t = (Σ_{j<=t} β2^{t-j} δ_j G_j²)/(Σ_{j<=t} β2^{t-j} G_j²)
       h^firstorder_t = 1 + A^(1)_t - A^(2)_t
  3. discrete finite-history replay (our exact characterization):
       h^replay_t = replay(G^g)/u1_t   (== actual h, the reference)

All three replay from a UNIFORM fresh start (m0=0, v0=0, step0=0) so the
first-order formula is complete (the full history is the 20-step gradient
history; the real extracted state would encode pre-checkpoint history the
first-order formula cannot see). Metrics: weighted R^2, Corr, wRMSE of each
predictor vs the actual h, per horizon, per K.

The answer to "how much more does finite-history provide over first-order
theory" is 1 - R^2(firstorder) (the exact replay is the reference, R^2~1).

Runs server-side (python3.6 + numpy, no torch). Output:
  analysis/predictor_comparison/summary.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analysis.numpy_radam import radam_step
from analysis.crossk_horizon_sweep import weighted_r2, corr, weighted_rmse

MAX_H = 20
LR = 1e-4
BETA1, BETA2 = 0.9, 0.999


def global_scalar(grad_g, grad_1):
    denom = float(np.sum(grad_1 * grad_1))
    if denom <= 0:
        return 1.0
    return float(np.sum(grad_g * grad_1) / denom)


def replay_stream(grad_hist_f32, m0, v0, step0, lr, beta1, beta2):
    m = np.array(m0, dtype=np.float32).copy()
    v = np.array(v0, dtype=np.float32).copy()
    step = int(step0)
    for j in range(grad_hist_f32.shape[0]):
        g = grad_hist_f32[j]
        m, v, delta = radam_step(m, v, step, g, lr, beta1, beta2)
        step += 1
        yield delta


def process_k(kdir, max_h=MAX_H):
    G1 = np.load(kdir / "grad_history_1.npy")   # (T,d) float64
    Gg = np.load(kdir / "grad_history_g.npy")
    T, d = G1.shape
    H = min(T, max_h)
    G1f = G1[:H].astype(np.float32)
    Ggf = Gg[:H].astype(np.float32)

    # per-step scalar + delta history
    a_star = [global_scalar(Gg[j], G1[j]) for j in range(H)]
    delta = [a - 1.0 for a in a_star]
    Ghat_f32 = np.stack([a_star[j] * G1[j] for j in range(H)]).astype(np.float32)
    del G1, Gg

    # replay three sequences from fresh start (streaming, lockstep)
    m0 = np.zeros(d, dtype=np.float32)
    v0 = np.zeros(d, dtype=np.float32)
    s1 = replay_stream(G1f, m0, v0, 0, LR, BETA1, BETA2)
    sg = replay_stream(Ggf, m0, v0, 0, LR, BETA1, BETA2)
    ss = replay_stream(Ghat_f32, m0, v0, 0, LR, BETA1, BETA2)

    # incremental first-order gauges (exponential sums)
    num1 = np.zeros(d, dtype=np.float64)
    den1 = np.zeros(d, dtype=np.float64)
    num2 = np.zeros(d, dtype=np.float64)
    den2 = np.zeros(d, dtype=np.float64)

    horizons = []
    for t in range(H):
        u1 = next(s1).astype(np.float64)
        ug = next(sg).astype(np.float64)
        us = next(ss).astype(np.float64)
        # update gauges
        G1t = G1f[t].astype(np.float64)
        Ggt = Ggf[t].astype(np.float64)
        num1 = BETA1 * num1 + delta[t] * G1t
        den1 = BETA1 * den1 + G1t
        num2 = BETA2 * num2 + delta[t] * G1t ** 2
        den2 = BETA2 * den2 + G1t ** 2
        A1 = num1 / (den1 + 1e-30)
        A2 = num2 / (den2 + 1e-30)

        # actual h and the three predictors
        h_act = np.ones(d); s = np.abs(u1) > 1e-30
        h_act[s] = ug[s] / u1[s]
        h_scalar = np.ones(d); sp = np.abs(u1) > 1e-30
        h_scalar[sp] = us[sp] / u1[sp]
        h_first = 1.0 + A1 - A2
        h_replay = h_act.copy()   # exact replay == actual h

        # effective support (top 1% by |u1|, scale-invariant)
        abs_u1 = np.abs(u1)
        k = max(int(0.99 * d) - 1, 0)
        thr = float(np.partition(abs_u1, k)[k])
        eff = abs_u1 >= thr if thr > 0 else np.ones(d, dtype=bool)
        w = u1 ** 2
        w_eff = w[eff]

        def metr(hp):
            return {
                "weighted_R2": weighted_r2(hp[eff], h_act[eff], w_eff),
                "corr": corr(hp[eff], h_act[eff], w_eff),
                "wRMSE": weighted_rmse(hp[eff], h_act[eff], w_eff),
            }

        horizons.append({
            "horizon_steps": t + 1, "eval_step": t,
            "scalar": metr(h_scalar),
            "firstorder": metr(h_first),
            "replay": metr(h_replay),
            "effective_coords": int(eff.sum()),
        })

    return {"T_steps": T, "dim": d, "horizons": horizons}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("analysis/real_history"))
    ap.add_argument("--out", type=Path, default=Path("analysis/predictor_comparison/summary.json"))
    a = ap.parse_args(argv)

    results = {}
    for label in ["k32", "k64", "k128", "k256"]:
        kdir = a.base / label
        if not (kdir / "grad_history_1.npy").exists():
            print(f"skip {label}: no grad history", flush=True)
            continue
        print(f"[predictor_comparison] processing {label} ...", flush=True)
        results[label] = process_k(kdir)
        h20 = results[label]["horizons"][-1]
        print(f"  {label} h20: scalar R2={h20['scalar']['weighted_R2']:.4f} "
              f"firstorder R2={h20['firstorder']['weighted_R2']:.4f} "
              f"replay R2={h20['replay']['weighted_R2']:.4f}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

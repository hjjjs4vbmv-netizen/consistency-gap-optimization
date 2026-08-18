"""Same-information-budget predictor comparison (P1, revised per review).

Ground truth: the real update ratio h_{t,i} = U^g_{t,i}/U_{t,i} (g=1.3 vs g=1.0),
where U^g = replay(G^g) and U = replay(G^1) through the real RAdam recursion.

NO predictor may access the full G^g. Each predictor is built from only:
  - G^1 (the reference gradient history), and
  - a*_j = <G^g_j, G^1_j>/||G^1_j||^2  (the per-step scalar projection of the gap).

Three predictors (same information budget, different optimizer-response models):

  A. Global scalar approximation:
       a_bar = mean_j a*_j  (one global scale)
       Ghat_j = a_bar * G^1_j ;  replay RAdam -> h^global = replay(Ghat)/u1
  B. Local / continuous scale-drift (Fernandez-Hernandez 2026, arXiv:2601.21739):
       delta_j = a*_j - 1
       A^(1)_t = (sum_{j<=t} beta1^{t-j} delta_j G^1_j)/(sum_{j<=t} beta1^{t-j} G^1_j)
       A^(2)_t = (sum_{j<=t} beta2^{t-j} delta_j (G^1_j)^2)/(sum_{j<=t} beta2^{t-j} (G^1_j)^2)
       h^local_t = 1 + A^(1)_t - A^(2)_t
  C. Discrete scalar-history replay (our method):
       Ghat_j = a*_j * G^1_j  (per-step scalar-scaled history)
       replay RAdam -> h^replay = replay(Ghat)/u1

The discrete replay uses the FULL per-step scale history {a*_j}_{j<=t} through the
real finite-history RAdam recursion; the global scalar uses a single scale; the
local/continuous uses a closed-form first-order approximation. None sees G^g.

Two regimes:
  - fresh:  m0=0, v0=0, step0=0  (zero history)
  - real:   m0/v0/step0 from the frozen checkpoint (accumulated optimizer history)

Metrics per predictor per horizon: weighted R^2, Corr, wRMSE, plus the weighted
variance of the target Var_w(h_actual) (contextualizes R^2 when the target is
near-constant, e.g. fresh-start).

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


def weighted_var(x, w):
    sup = w > 0
    if not sup.any():
        return math.nan
    wp = w[sup]; xv = x[sup]
    wsum = float(np.sum(wp))
    xbar = float(np.sum(wp * xv) / wsum)
    return float(np.sum(wp * (xv - xbar) ** 2) / wsum)


def process_k(kdir, state_np, max_h=MAX_H):
    G1 = np.load(kdir / "grad_history_1.npy")   # (T,d) float64
    Gg = np.load(kdir / "grad_history_g.npy")
    T, d = G1.shape
    H = min(T, max_h)
    G1f = G1[:H].astype(np.float32)
    Ggf = Gg[:H].astype(np.float32)

    a_star = [global_scalar(Gg[j], G1[j]) for j in range(H)]
    # global scalar uses the CAUSAL mean a_bar_t = mean(a*_j, j<=t) — no future
    # leakage (same information budget {a*_j}_{j<=t} as the discrete replay).
    cum = 0.0
    a_bar = [0.0] * H
    for j in range(H):
        cum += a_star[j]
        a_bar[j] = cum / (j + 1)
    delta = [a - 1.0 for a in a_star]
    Ghat_global = np.stack([a_bar[j] * G1[j] for j in range(H)]).astype(np.float32)
    Ghat_local = np.stack([a_star[j] * G1[j] for j in range(H)]).astype(np.float32)
    del G1, Gg

    regimes = {}
    for regime, (m0, v0, step0) in {
        "fresh": (np.zeros(d, dtype=np.float32), np.zeros(d, dtype=np.float32), 0),
        "real": (np.load(state_np["m0"]).astype(np.float32),
                 np.load(state_np["v0"]).astype(np.float32),
                 int(state_np["step0"])),
    }.items():
        # ground truth: replay the REAL G^g (the target, not a predictor)
        s_act1 = replay_stream(G1f, m0, v0, step0, LR, BETA1, BETA2)
        s_actg = replay_stream(Ggf, m0, v0, step0, LR, BETA1, BETA2)
        # predictor A: global scalar
        s_glob = replay_stream(Ghat_global, m0, v0, step0, LR, BETA1, BETA2)
        # predictor C: discrete scalar-history replay
        s_repl = replay_stream(Ghat_local, m0, v0, step0, LR, BETA1, BETA2)

        # incremental first-order gauges (predictor B)
        num1 = np.zeros(d, dtype=np.float64)
        den1 = np.zeros(d, dtype=np.float64)
        num2 = np.zeros(d, dtype=np.float64)
        den2 = np.zeros(d, dtype=np.float64)

        horizons = []
        for t in range(H):
            u1 = next(s_act1).astype(np.float64)
            ug = next(s_actg).astype(np.float64)
            u_glob = next(s_glob).astype(np.float64)
            u_repl = next(s_repl).astype(np.float64)
            G1t = G1f[t].astype(np.float64)
            num1 = BETA1 * num1 + delta[t] * G1t
            den1 = BETA1 * den1 + G1t
            num2 = BETA2 * num2 + delta[t] * G1t ** 2
            den2 = BETA2 * den2 + G1t ** 2
            A1 = num1 / (den1 + 1e-30)
            A2 = num2 / (den2 + 1e-30)

            h_act = np.ones(d); s = np.abs(u1) > 1e-30
            h_act[s] = ug[s] / u1[s]
            h_glob = np.ones(d); sg = np.abs(u1) > 1e-30
            h_glob[sg] = u_glob[sg] / u1[sg]
            h_repl = np.ones(d); sr = np.abs(u1) > 1e-30
            h_repl[sr] = u_repl[sr] / u1[sr]
            h_local = 1.0 + A1 - A2

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
                "global_scalar": metr(h_glob),
                "local_continuous": metr(h_local),
                "discrete_replay": metr(h_repl),
                "Var_w_h_actual": weighted_var(h_act[eff], w_eff),
                "h_actual_mean": float(np.mean(h_act[eff])),
                "effective_coords": int(eff.sum()),
            })
        regimes[regime] = {"horizons": horizons}

    return {"T_steps": T, "dim": d, "a_bar": a_bar, "regimes": regimes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("analysis/real_history"))
    ap.add_argument("--state-np", type=str, required=True,
                    help="JSON mapping K-label -> {m0, v0, step0} npy/json paths")
    ap.add_argument("--out", type=Path, default=Path("analysis/predictor_comparison/summary.json"))
    a = ap.parse_args(argv)

    state_map = json.loads(Path(a.state_np).read_text())
    results = {}
    for label in ["k32", "k64", "k128", "k256"]:
        kdir = a.base / label
        if not (kdir / "grad_history_1.npy").exists():
            print(f"skip {label}: no grad history", flush=True)
            continue
        print(f"[predictor_comparison] processing {label} ...", flush=True)
        results[label] = process_k(kdir, state_map[label])
        for regime in ["fresh", "real"]:
            h20 = results[label]["regimes"][regime]["horizons"][-1]
            print(f"  {label}/{regime} h20: "
                  f"global R2={h20['global_scalar']['weighted_R2']:.4f} "
                  f"local R2={h20['local_continuous']['weighted_R2']:.4f} "
                  f"discrete R2={h20['discrete_replay']['weighted_R2']:.4f} "
                  f"Var_w(h)={h20['Var_w_h_actual']:.2e}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

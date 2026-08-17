"""Balanced-β intervention on gap-induced RAdam divergence.

Question: does the gap-induced R_opt — the non-scalar residual between the
g=1.3 and g=1.0 RAdam updates — shrink when β1 = β2?

Controlled protocol (per collaborator, P0):
  - SAME real paired gradient sequence {G_t^1.0, G_t^1.3}_{t=1..H} from the
    frozen #49 Arm-A checkpoints (H=20).
  - Replay RAdam from a UNIFORMLY DEFINED starting point (m0=0, v0=0, step0=0)
    for every β config — never from the real extracted state, so no
    "old-β history + new-β step" mixing.
  - β configs: (0.9, 0.999) standard; (0.9, 0.9), (0.99, 0.99), (0.999, 0.999) balanced.
  - Per config, per horizon t: R_opt (headline), Disp(h), Corr(u1,ug), h_i stats.
  - Core prediction: R_opt^{β1=β2} < R_opt^{0.9,0.999}. If it fails, that is an
    important falsification — reported honestly.

Runs server-side (python3.6 + numpy, no torch). Output:
  analysis/balanced_beta/summary.json
  analysis/balanced_beta/{k}/raw_h20/{config}_h_actual.npy   (h_i on eff support, h=20)
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from analysis.numpy_radam import radam_step

BETA_CONFIGS = [
    ("standard", 0.9, 0.999),
    ("balanced_0.9", 0.9, 0.9),
    ("balanced_0.99", 0.99, 0.99),
    ("balanced_0.999", 0.999, 0.999),
]
MAX_H = 20
LR = 1e-4


def replay_stream(grad_hist_f32, m0, v0, step0, lr, beta1, beta2):
    """Yield per-step float32 deltas without storing the full history."""
    m = np.array(m0, dtype=np.float32).copy()
    v = np.array(v0, dtype=np.float32).copy()
    step = int(step0)
    for j in range(grad_hist_f32.shape[0]):
        g = grad_hist_f32[j]
        m, v, delta = radam_step(m, v, step, g, lr, beta1, beta2)
        step += 1
        yield delta


def process_k(kdir, outdir, max_h=MAX_H):
    G1 = np.load(kdir / "grad_history_1.npy")   # (T,d) float64
    Gg = np.load(kdir / "grad_history_g.npy")
    T, d = G1.shape
    H = min(T, max_h)
    G1f = G1[:H].astype(np.float32)
    Ggf = Gg[:H].astype(np.float32)
    del G1, Gg

    configs = {}
    for name, b1, b2 in BETA_CONFIGS:
        m0 = np.zeros(d, dtype=np.float32)
        v0 = np.zeros(d, dtype=np.float32)
        horizons = []
        raw_h20 = None
        for t, (u1, ug) in enumerate(zip(
                replay_stream(G1f, m0, v0, 0, LR, b1, b2),
                replay_stream(Ggf, m0, v0, 0, LR, b1, b2))):
            u1 = u1.astype(np.float64)
            ug = ug.astype(np.float64)
            h_act = np.ones(d)
            s = np.abs(u1) > 1e-30
            h_act[s] = ug[s] / u1[s]
            # effective support = top 1% of coordinates by |u1| (quantile mask via
            # np.partition, O(n)). Fresh-start updates span many orders of magnitude
            # across β configs (β2=0.999 rectification transient damps updates to
            # ~1e-7), so an absolute threshold would empty the support; a quantile
            # mask is scale-invariant and matches the cross-K effective support
            # (~1% of coords). R_opt is computed over the FULL vector regardless.
            abs_u1 = np.abs(u1)
            k = max(int(0.99 * d) - 1, 0)
            thr = float(np.partition(abs_u1, k)[k])
            eff = abs_u1 >= thr if thr > 0 else np.ones(d, dtype=bool)
            s_opt = float(np.sum(ug * u1) / max(np.sum(u1 * u1), 1e-30))
            R_opt = float(np.linalg.norm(ug - s_opt * u1) / max(np.linalg.norm(u1), 1e-30))
            d1 = float(np.linalg.norm(u1))
            dg = float(np.linalg.norm(ug))
            cos = float(np.sum(u1 * ug) / (d1 * dg)) if d1 > 0 and dg > 0 else math.nan
            he = h_act[eff]
            horizons.append({
                "horizon_steps": t + 1, "eval_step": t,
                "R_opt": R_opt, "Disp_h": float(np.std(he)),
                "corr_u1_ug": cos,
                "h_actual_mean": float(np.mean(he)),
                "h_actual_std": float(np.std(he)),
                "effective_coords": int(eff.sum()),
            })
            if t == H - 1:
                raw_h20 = he.copy()
        configs[name] = {"beta1": b1, "beta2": b2, "horizons": horizons}
        # persist h_i on eff support at h=20 (auditability, like cross-K raw_predictions)
        rawdir = outdir / "raw_h20"
        rawdir.mkdir(parents=True, exist_ok=True)
        np.save(rawdir / f"{name}_h_actual.npy", raw_h20)
        print(f"  {name}: h20 R_opt={horizons[-1]['R_opt']:.4f} "
              f"Disp={horizons[-1]['Disp_h']:.4f} Corr={horizons[-1]['corr_u1_ug']:.4f} "
              f"eff={int(horizons[-1]['effective_coords'])}", flush=True)

    return {"T_steps": T, "dim": d, "configs": configs}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("analysis/real_history"))
    ap.add_argument("--out", type=Path, default=Path("analysis/balanced_beta/summary.json"))
    a = ap.parse_args(argv)

    results = {}
    for label in ["k32", "k64", "k128", "k256"]:
        kdir = a.base / label
        if not (kdir / "grad_history_1.npy").exists():
            print(f"skip {label}: no grad history", flush=True)
            continue
        print(f"[balanced_beta] processing {label} ...", flush=True)
        results[label] = process_k(kdir, a.out.parent / label)
        h20 = {n: c["horizons"][-1] for n, c in results[label]["configs"].items()}
        std = h20["standard"]["R_opt"]
        print(f"  {label} h20: " + "  ".join(
            f"{n}={h20[n]['R_opt']:.4f}" for n in h20) + f"  [std={std:.4f}]", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()

"""E5 — non-scalar residual structure decomposition (free analysis on existing data).

Tests whether the non-scalar residual is STRUCTURED (not per-coordinate noise):
  (a) layer-depth structure: does R_opt_layer / the gauge deviate from 1 vary with
      layer position (encoder vs decoder, block index)?
  (b) magnitude correlation: does the residual correlate with layer update/gradient
      magnitude?
  (c) direction: gap_gradient_layerwise.csv 'direction_residual' across gap values.

Outputs a summary + correlations. No GPU needed (reads committed CSVs).
"""
from __future__ import annotations
import csv, math, re
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
LAYERWISE = REPO / "analysis" / "radam_update_stateful_layerwise.csv"
GAPGRAD = REPO / "analysis" / "gap_gradient_layerwise.csv"

def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return math.nan
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    xm, ym = rx.mean(), ry.mean()
    num = ((rx-xm)*(ry-ym)).sum()
    den = math.sqrt(((rx-xm)**2).sum() * ((ry-ym)**2).sum())
    return num/den if den else math.nan

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2: return math.nan
    xm, ym = x.mean(), y.mean()
    den = math.sqrt(((x-xm)**2).sum() * ((y-ym)**2).sum())
    return ((x-xm)*(y-ym)).sum()/den if den else math.nan

def depth_of(name):
    """Approx depth: 0 = model.top/input, 1 = enc.*, 2 = mid, 3 = dec.* (deeper = larger)."""
    if name.startswith("model."):
        name = name[len("model."):]
    m = re.match(r"(\w+)\.(\d+)x(\d+)_block(\d+)", name)
    if m:
        stage, hx, wx, blk = m.groups()
        base = {"input":0,"top":0,"enc":1,"mid":2,"dec":3,"output":4,"top.":0}.get(stage,0)
        # deeper block index = more depth within stage
        return base*100 + int(blk)*10
    # non-conv blocks (norm, affine, skip, conv) inherit a coarse depth
    for i,k in enumerate(["input","enc","mid","dec","output"]):
        if name.startswith(k): return i*100
    return 50

def main():
    # ---- (a)(b) stateful layerwise ----
    rows = []
    with open(LAYERWISE, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"stateful layerwise rows: {len(rows)}")
    layers = [r["layer"] for r in rows]
    r_opt = [float(r["R_opt_layer"]) for r in rows]
    r_cstar = [float(r["layer_residual_with_global_c_star"]) for r in rows]
    s_layer = [float(r["s_K_star_layer"]) for r in rows]
    h_std = [float(r["h_update_weighted_std"]) for r in rows]
    upd_mag = [float(r["update_1_l2"]) for r in rows]
    h_mean = [float(r["h_update_weighted_mean"]) for r in rows]
    h_dev = [abs(float(r["h_update_weighted_mean"])-1.0) for r in rows]
    depths = [depth_of(l) for l in layers]

    print("\n(a) LAYER-DEPTH STRUCTURE: correlation of residual/gauge-deviation with layer depth")
    print(f"  R_opt_layer    vs depth : pearson={pearson(depths,r_opt):+.3f} spearman={spearman(depths,r_opt):+.3f}")
    print(f"  resid_cstar     vs depth : pearson={pearson(depths,r_cstar):+.3f} spearman={spearman(depths,r_cstar):+.3f}")
    print(f"  |h_mean - 1|    vs depth : pearson={pearson(depths,h_dev):+.3f} spearman={spearman(depths,h_dev):+.3f}")

    print("\n(b) MAGNITUDE CORRELATION: is the residual larger on high-magnitude layers?")
    print(f"  R_opt_layer  vs update_mag : pearson={pearson(upd_mag,r_opt):+.3f} spearman={spearman(upd_mag,r_opt):+.3f}")
    print(f"  |h_mean-1|   vs update_mag : pearson={pearson(upd_mag,h_dev):+.3f} spearman={spearman(upd_mag,h_dev):+.3f}")

    # spread of gauge deviation: is h_mean concentrated near 1 (weak) or spread (structured)?
    print("\n  gauge deviation |h_mean-1| stats:")
    arr = np.array(h_dev)
    print(f"    n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f} p05={np.quantile(arr,.05):.4f} p95={np.quantile(arr,.95):.4f} frac>0.1={(arr>0.1).mean():.3f}")

    # top/bottom layers by residual
    order = np.argsort(r_opt)[::-1]
    print("\n  Top-5 layers by R_opt_layer (largest residual):")
    for i in order[:5]:
        print(f"    {layers[i]:<45} R_opt={r_opt[i]:.4f} h_mean={h_mean[i]:.4f} depth~{depths[i]}")
    print("  Bottom-5 (smallest residual):")
    for i in order[-5:]:
        print(f"    {layers[i]:<45} R_opt={r_opt[i]:.4f} h_mean={h_mean[i]:.4f} depth~{depths[i]}")

    # ---- (c) direction residual across gaps ----
    if GAPGRAD.exists():
        gd = {}
        with open(GAPGRAD, newline="") as f:
            for r in csv.DictReader(f):
                gd.setdefault(r["gap"], []).append(float(r["direction_residual"]))
        print("\n(c) DIRECTION RESIDUAL vs GAP (gap_gradient_layerwise.csv):")
        for gap in sorted(gd, key=float):
            v = np.array(gd[gap])
            print(f"    gap={gap}: n={len(v)} mean_dir_resid={v.mean():.4f} std={v.std():.4f}")
    else:
        print("\n(c) gap_gradient_layerwise.csv not found; skipping")

if __name__ == "__main__":
    main()

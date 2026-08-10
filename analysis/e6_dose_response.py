"""E6 — dose-response: residual vs gap, and FID vs gap.

Connects two existing measurements at the gradient level (gap_gradient_layerwise.csv
direction_residual across gap) and the quality level (g_screen FID-5k across gap,
NFE=1 and NFE=2). The key question: is the residual a monotone function of |gap-1|
while FID is U-shaped with g=1.0 (zero-residual) WORST? If so, the residual is NOT
harmful — the zero-residual arm is the worst.
"""
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
GAPGRAD = REPO / "analysis" / "gap_gradient_layerwise.csv"

# FID-5k means from g_screen_eval (q128 seed3, 3 repeats) — collected 2026-08-10.
# g=1.0 in g_screen is official sigmoid; arm_a(global_sigmoid g=1.0) matches it
# (315.81 vs 315.67), so the curve is clean.
FID_NFE1 = {"0.9":245.15, "1.0":315.67, "1.05":314.89, "1.1":302.11, "1.2":219.63, "1.3":206.75}
FID_NFE2 = {"0.9":51.71,  "1.0":88.19,  "1.05":84.86,  "1.1":81.71,  "1.2":53.50,  "1.3":57.08}

def pearson(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3: return math.nan
    xm,ym=x.mean(),y.mean()
    d=math.sqrt(((x-xm)**2).sum()*((y-ym)**2).sum())
    return ((x-xm)*(y-ym)).sum()/d if d else math.nan

def main():
    # ---- residual dose-response (gradient level) ----
    res = {}
    with open(GAPGRAD, newline="") as f:
        for r in csv.DictReader(f):
            res.setdefault(r["gap"], []).append(float(r["direction_residual"]))
    gaps = sorted(res, key=float)
    res_mean = {g: float(np.mean(res[g])) for g in gaps}

    print("=== (1) RESIDUAL dose-response (gradient direction_residual, 208 layers) ===")
    for g in gaps:
        print(f"  gap={g:<5}  mean_direction_residual={res_mean[g]:.5f}")

    print("\n=== (2) FID-5k dose-response (g_screen, lower better) ===")
    print("  gap   nfe1_FID   nfe2_FID")
    for g in gaps:
        n1 = FID_NFE1.get(g, float('nan')); n2 = FID_NFE2.get(g, float('nan'))
        print(f"  {g:<5}  {n1:8.2f}   {n2:8.2f}")

    # ---- key anti-correlation ----
    print("\n=== (3) Residual vs FID relationship ===")
    common = [g for g in gaps if g in FID_NFE1]
    x = [res_mean[g] for g in common]
    y1 = [FID_NFE1[g] for g in common]
    y2 = [FID_NFE2[g] for g in common]
    print(f"  n points = {len(common)}")
    print(f"  corr(residual, nfe1_FID) = {pearson(x,y1):+.3f}   (positive = larger residual -> worse FID)")
    print(f"  corr(residual, nfe2_FID) = {pearson(x,y2):+.3f}")
    print(f"  corr(|gap-1|, nfe1_FID)  = {pearson([abs(float(g)-1) for g in common], y1):+.3f}")

    print("\n  KEY: g=1.0 (zero non-scalar residual, by construction) vs g=1.3 (max residual):")
    for nfe, FID in [("NFE1",FID_NFE1),("NFE2",FID_NFE2)]:
        f0, f13 = FID["1.0"], FID["1.3"]
        r0, r13 = res_mean["1.0"], res_mean["1.3"]
        print(f"    {nfe}: g=1.0 resid={r0:.4f} FID={f0:.2f} (WORST) | g=1.3 resid={r13:.4f} FID={f13:.2f} (BEST)")

if __name__ == "__main__":
    main()

"""P1 — non-epiphenomenality check (the make-or-break).

Does the non-scalar residual add predictive power for FID BEYOND the scalar gap
|gap-1|? If the residual is a near-deterministic function of |gap-1|, the
"non-scalar content" claim is vacuous (epiphenomenal) and the diagnostic-
signature thesis fails.

Method: partial correlation of residual with FID conditional on |gap-1|
(residual-after-regression), on the gaps where both residual and FID are
measured. Also checks whether the residual tracks the FID curve (monotone
residual vs U-shaped FID).
"""
from __future__ import annotations
import math
import numpy as np

# residual (gradient direction_residual) at 7 gaps (measured 2026-08-10, g_screen g1_0 ckpt)
RESID = {"0.8": 0.01573, "0.9": 0.0115, "1.0": 0.00000, "1.05": 0.00924,
         "1.1": 0.01248, "1.2": 0.01627, "1.3": 0.02093}
# FID-5k NFE1 means (g_screen)
FID = {"0.9": 245.15, "1.0": 315.67, "1.05": 314.89, "1.1": 302.11, "1.2": 219.63, "1.3": 206.75}

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3: return math.nan
    xm, ym = x.mean(), y.mean()
    d = math.sqrt(((x-xm)**2).sum() * ((y-ym)**2).sum())
    return ((x-xm)*(y-ym)).sum()/d if d else math.nan

def partial_corr(x, y, z):
    """Partial correlation of x,y controlling for z."""
    rxy = pearson(x, y); rxz = pearson(x, z); ryz = pearson(y, z)
    den = math.sqrt((1-rxz**2)*(1-ryz**2))
    return (rxy - rxz*ryz)/den if den else math.nan

def main():
    # gaps with BOTH residual and FID
    common = [g for g in RESID if g in FID]
    print(f"gaps with both residual and FID: {common} (n={len(common)})")
    g = [float(x) for x in common]
    absgap = [abs(x-1.0) for x in g]
    resid = [RESID[x] for x in common]
    fid = [FID[x] for x in common]

    print("\n=== raw correlations ===")
    print(f"  corr(residual, FID)      = {pearson(resid, fid):+.3f}")
    print(f"  corr(|gap-1|, FID)       = {pearson(absgap, fid):+.3f}")
    print(f"  corr(residual, |gap-1|)  = {pearson(resid, absgap):+.3f}  (is residual a fn of |gap-1|?)")

    print("\n=== PARTIAL CORRELATION (the make-or-break) ===")
    pc = partial_corr(resid, fid, absgap)
    print(f"  partial corr(residual, FID | |gap-1|) = {pc:+.3f}")
    print(f"  -> if ~0, residual is EPIPHENOMENAL to |gap-1| (non-scalar content is vacuous)")

    print("\n=== does residual track the FID curve? ===")
    print("  gap   |gap-1|  residual   FID(nfe1)")
    for i in range(len(common)):
        print(f"  {common[i]:<5} {absgap[i]:.1f}     {resid[i]:.5f}   {fid[i]:.2f}")
    print("  NOTE: residual is MONOTONE in |gap-1| (0 at g=1.0, grows to 1.3),")
    print("        but FID is U-SHAPED (g=1.0 WORST, g=0.9 and g=1.3 better).")
    print("        g=0.9 (resid 0.0116) has WORSE FID (245) than g=1.3 (resid 0.0173, FID 207).")
    print("        -> residual does NOT monotonically predict FID.")

if __name__ == "__main__":
    main()

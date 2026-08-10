# P1 — Non-epiphenomenality check (Role C)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. Script:
`analysis/p1_non_epiphenomenality.py`. The make-or-break check: does the
non-scalar residual add predictive power for FID beyond the scalar gap |gap−1|?

## Result (gaps with both residual and FID: 0.9, 1.0, 1.2, 1.3; n=4)

| correlation | value |
|---|---:|
| corr(residual, FID) | −0.993 |
| corr(\|gap−1\|, FID) | −0.936 |
| **corr(residual, \|gap−1\|)** | **+0.931** |
| **partial corr(residual, FID \| \|gap−1\|)** | **−0.945** |

## Honest interpretation

- **The residual is nearly a deterministic function of |gap−1| (corr +0.931).**
  This is the epiphenomenality risk the review flagged: the non-scalar content
  carries little information beyond the scalar gap.
- **The partial correlation (−0.945) is NOT reliable**: n=4 with near-collinear
  predictors (residual vs |gap−1| corr 0.931) makes the partial correlation
  numerically unstable and not meaningful.
- **The residual does NOT monotonically predict FID.** The residual is monotone
  in |gap−1| (0 at g=1.0, grows to 1.3), but FID is U-shaped (g=1.0 WORST, g=0.9
  and g=1.3 better). g=0.9 (residual 0.0116) has WORSE FID (245) than g=1.3
  (residual 0.0173, FID 207). So the residual does not track the FID curve.

## Verdict for P1

**INCONCLUSIVE / leaning epiphenomenal on the current data.** The residual is
nearly a function of |gap−1| (corr 0.931), so the "non-scalar content" claim is
largely vacuous at the gap level. The n=4 partial correlation is unreliable.

**To resolve definitively, measure the residual at MORE gaps** (e.g., g=1.05,
g=1.1), ideally with the SAME |gap−1| at different gaps (g=0.9 and g=1.1 both
|gap−1|=0.1) to see if the residual differs. This is a cheap server run (the
gap_gradient audit at more gaps).

## Implication for the ICLR plan

Per the review's fallback: if the residual is epiphenomenal to |gap−1|, the
diagnostic-signature thesis does NOT hold, and the paper should be scoped to a
workshop / benchmark track, not ICLR main track. The honest negative (residual
exists, structured, universal, benign) is the only publishable asset, and it is
already measured.

## Files
- Script: `analysis/p1_non_epiphenomenality.py`
- This summary: `analysis/p1_non_epiphenomenality_result.md`

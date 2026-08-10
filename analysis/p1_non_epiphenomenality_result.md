# P1 — Non-epiphenomenality check (Role C)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. Script:
`analysis/p1_non_epiphenomenality.py`. The make-or-break check: does the
non-scalar residual add predictive power for FID beyond the scalar gap |gap−1|?

## Data (residual measured at 7 gaps, 2026-08-10, g_screen g1_0 ckpt)

| gap | direction_residual | |gap−1| | FID-5k (NFE1) |
|---|---:|---:|---:|
| 0.8 | 0.01573 | 0.2 | — |
| 0.9 | 0.01150 | 0.1 | 245.15 |
| 1.0 | 0.00000 | 0.0 | 315.67 |
| 1.05 | 0.00924 | 0.05 | 314.89 |
| 1.1 | 0.01248 | 0.1 | 302.11 |
| 1.2 | 0.01627 | 0.2 | 219.63 |
| 1.3 | 0.02093 | 0.3 | 206.75 |

## Result (6 gaps with both residual and FID: 0.9, 1.0, 1.05, 1.1, 1.2, 1.3)

| correlation | value |
|---|---:|
| corr(residual, FID) | −0.817 |
| corr(\|gap−1\|, FID) | −0.899 |
| **corr(residual, \|gap−1\|)** | **+0.938** |
| **partial corr(residual, FID \| \|gap−1\|)** | **+0.168** |

## Honest interpretation

- **The residual is nearly a deterministic function of |gap−1| (corr +0.938).**
- **The partial correlation of residual with FID conditional on |gap−1| is
  +0.168 ≈ 0** (and even slightly positive, opposite sign). Conditional on the
  scalar gap, the residual adds essentially NO predictive power for FID.
- **The residual is EPIPHENOMENAL to |gap−1|.** The "non-scalar content" claim
  is vacuous at the gap level: the residual carries no information beyond the
  scalar gap for predicting FID.
- The residual does differ slightly at the same |gap−1| (g=0.9: 0.0115 vs g=1.1:
  0.0125; g=0.8: 0.0157 vs g=1.2: 0.0163), but this small variation does not
  translate into FID-predictive power.

## Verdict for P1

**DECISIVE: the residual is epiphenomenal to |gap−1|.** The diagnostic-signature
thesis (residual predicts FID beyond the scalar gap) is NOT supported by the
measured data. The partial correlation is ≈0.

## Implication for the ICLR plan

Per the review's fallback: the diagnostic-signature thesis does NOT hold, so the
paper should be scoped to a **workshop / benchmark track, not ICLR main track**.
The honest negative (residual exists, structured, universal, benign) is the only
publishable asset, and it is already measured. Do NOT submit the diagnosis alone
to ICLR main track.

## Files
- Script: `analysis/p1_non_epiphenomenality.py`
- This summary: `analysis/p1_non_epiphenomenality_result.md`
- Server data: `/data/raw/ECT/ect_runs/p1_more_gaps_0809/gap_gradient_layerwise.csv`

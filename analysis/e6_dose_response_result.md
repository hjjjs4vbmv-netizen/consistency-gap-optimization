# E6 — Dose-response: residual vs gap, FID vs gap (Role C)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. Free analysis from existing
data. Script: `analysis/e6_dose_response.py`.

## Hypothesis (E6)
The non-scalar residual is a systematic (dose-response) function of the gap, and
connecting it to the FID-vs-gap curve reveals whether the residual is harmful.

## Data
- Residual: `gap_gradient_layerwise.csv` mean gradient direction_residual across gap
  (0.9 / 1.0 / 1.2 / 1.3).
- FID-5k: g_screen eval (q128, seed3), NFE=1 and NFE=2. g=1.0 arm matches the
  measured arm_a value (315.67 vs 315.81), so the curve is clean across gaps.

## Results

### (1) Residual dose-response (gradient level)
| gap | mean direction_residual (208 layers) |
|---|---:|
| 0.9 | 0.01164 |
| 1.0 | 0.00000 (reference, zero by construction) |
| 1.2 | 0.01356 |
| 1.3 | 0.01731 |

Residual is a **monotone function of |gap−1|** (zero at the reference g=1.0).

### (2) FID-5k dose-response
| gap | NFE=1 FID | NFE=2 FID |
|---|---:|---:|
| 0.9 | 245.15 | 51.71 |
| 1.0 | **315.67 (worst)** | **88.19 (worst)** |
| 1.2 | 219.63 | 53.50 |
| 1.3 | **206.75 (best)** | 57.08 |

### (3) Residual vs FID
| correlation | value |
|---|---:|
| corr(residual, NFE1 FID) | **−0.993** |
| corr(residual, NFE2 FID) | −0.900 |
| corr(\|gap−1\|, NFE1 FID) | −0.936 |

## Interpretation (honest)

- The residual is **strongly ANTI-correlated with FID**: larger residual → better
  FID. The zero-residual arm (g=1.0) is the WORST; the max-residual arm (g=1.3) is
  the BEST. This directly contradicts any "residual is harmful" claim.
- Caveats: only 4 gap points have both residual and FID; g=1.0 is zero-residual BY
  CONSTRUCTION (it is the reference); and g=0.9 (moderate residual, FID 245) is
  worse than g=1.3 (max residual, FID 207), so the relationship is not a strict
  monotone "more residual → better" — it is dominated by g=1.0 being an outlier
  (worst) and g=1.3 best.
- Cleanest honest statement: **deviating from g=1.0 in either direction improves
  FID, and g=1.0 — the only arm with zero non-scalar residual — is the worst.**
  The residual does not harm quality; if anything, its presence correlates with
  better FID, and it is a systematic (monotone in |gap−1|) consequence of the gap.

## Verdict for E6
- **PASS (dose-response, systematic):** residual is monotone in |gap−1|.
- The residual is NOT harmful: the zero-residual arm is the worst on FID.
- This strengthens the honest diagnosis and further rules out the mechanism+method
  premise.

## Files
- Analysis: `analysis/e6_dose_response.py`
- This summary: `analysis/e6_dose_response_result.md`

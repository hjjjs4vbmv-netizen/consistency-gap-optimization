# Moment-Memory Prediction on Real ECT History — Result (Role C, corrected)

Date: 2026-08-09. Branch: `role-c/moment-memory-real-history` (PR #47).
Data: real paired sweep from `gap_lr_matched_q128_s3_v1` arm_a (g=1.0) 256-kimg
state; 20 replay steps pairing g=1.0 (reference) vs g=1.3 (candidate) with
identical (batch, t, noise, dropout) per step.

## Correction note

An earlier version of this doc reported RMSE ≈ 0.19 and Corr ≈ 0.28 from a
top-5000 subsample. **That subsample was not representative and is retracted.**
The full, support-aware diagnosis below is the correct result.

## Support-aware full result

| atol on \|u1\| | coords | RMSE(ĥ,h^upd) | Corr | Disp(ĥ) | R_opt |
|---|---:|---:|---:|---:|---:|
| 1e-6 | 43.5M (78%) | 104.7 | 0.0002 | 104.7 | 0.106 |
| **1e-5** | **0.99M (1.8%)** | **0.42** | **0.005** | **0.38** | **0.057** |

- 78% of coordinates have `|u1| < 1e-5`; their `h = ug/u1` is numerical noise
  (dominates the unfiltered RMSE=104.7).
- On the true effective support (`|u1| > 1e-5`, ~1.8% of coords) the metrics
  are RMSE = 0.42, **Corr = 0.005**.

## Why Corr ≈ 0 (the mechanism, not noise)

On the effective support:

| quantity | mean | std | notes |
|---|---:|---:|---|
| h^actual | **0.837** | 0.058 | tightly concentrated, 100% in [0.5,1.5] |
| ĥ (predicted) | 1.002 | 0.441 | median 1.003, p5-p95 = [0.975, 1.025], min -161 / max +266 |
| A^(1) | -0.231 | **0.342** | signed first-moment gauge, wide spread |
| A^(2) | -0.233 | 0.013 | stable |
| B^(2) | 0.055 | 0.006 | stable |

- **ĥ is ≈ 1 on 90% of effective coords** (A^(1) ≈ A^(2) ≈ -0.23 cancel), the
  #45 Corollary-1 scalar null. Its extreme tail (|ĥ| ≫ 1) comes from
  coordinates where the *signed* first-moment denominator `Σ p G` is near zero
  — the sign-qualification caveat of #45; those are excluded by a support-atol
  on A^(1)'s denominator in any rigorous use.
- **h^actual is ≈ 0.837 on the same coords** — tightly concentrated, and
  systematically below 1.

So both ĥ (on its central 90%) and h^actual are near-constant but at **different
values** (≈1 vs ≈0.837): the scalar chain predicts the null value 1, while the
actual update ratio is 0.837. Hence Corr ≈ 0 (no coordinate-level co-variation)
even though both are stable — they are stably different.

## Honest answer to "how much does moment-history explain of real R_opt?"

1. **The scalar moment-memory chain predicts h ≈ 1** (the Corollary-1 null),
   and this is what the scalar part of the gradient history supports
   (A^(1) ≈ A^(2) ≈ -0.23, δ_j stable).
2. **The actual update ratio is 0.837, not 1.** The systematic offset is a
   non-scalar effect: the 3.2% per-step gradient residual E_j shifts the
   rectified-RAdam update ratio away from the scalar null. The scalar chain
   **cannot** predict this offset.
3. **R_opt − R_grad = +0.028** is a real (small) memory increment, but the
   dominant update distortion is the non-scalar-gradient offset, which the
   scalar δ_j model does not capture.

**Implication for GFCT (now more precise):** the scalar equivalence story holds
only as a null at the level of the coordinate-conditional gauge; the actual
optimizer distortion on a real 256-kimg state is dominated by non-scalar
gradient content (h ≈ 0.837 vs scalar-predicted 1), not by scalar-scale memory.
This is a measurable gap effect in optimizer space beyond scalar rescaling —
qualitatively supportive of GFCT, but it is the **non-scalar residual**, not the
moment-memory of the scalar history, that drives it.

## Files
- `analysis/real_history_sweep.py` — paired replay (real state → history).
- `analysis/moment_memory_prediction.py` — δ_j → A/B → ĥ pipeline.
- Server diagnostics: `analysis/real_history_full_diag.py` (support-aware),
  `real_history_diag.py` (subsampled, retracted from conclusion).

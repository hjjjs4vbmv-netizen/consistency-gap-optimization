# Moment-Memory Prediction on Real ECT History — Result (Role C, corrected)

Date: 2026-08-09. Branch: `role-c/moment-memory-real-history` (PR #47).
Data: real paired sweep from `gap_lr_matched_q128_s3_v1` arm_a (g=1.0) 256-kimg
state; 20 replay steps pairing g=1.0 (reference) vs g=1.3 (candidate) with
identical (batch, t, noise, dropout) per step.

## Correction note (two fixes)

1. An earlier version reported RMSE ≈ 0.19 / Corr ≈ 0.28 from a top-5000
   subsample. **That subsample was not representative and is retracted.**
2. The pipeline originally recovered a **global** best-fit δ_j and multiplied
   it back coordinate-wise. Self-review found this is **wrong under non-scalar
   gradients** (A1's coordinate spread ~100x too small). The correct gauges use
   the paired gradients directly:
   ```
   A1 = Σ_j p_j (G^g_j − G_j) / Σ_j p_j G_j          (exact: δ_i G_i = G^g_i − G_i)
   A2 = Σ_j q_j (G^g_j − G_j) G_j / Σ_j q_j G_j²
   B2 = Σ_j q_j (G^g_j − G_j)² / Σ_j q_j G_j²
   ```
   Verified exact to machine precision, and on a non-scalar synthetic (δ spread
   0.1) it gives RMSE(ĥ,h_actual)=0.0000, Corr=1.0000. The results below use
   this corrected formula.

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

On the effective support (corrected formula):

| atol | n | RMSE | Corr | ĥ median | h^actual median |
|---|---:|---:|---:|---:|---:|
| 1e-6 | 43.5M | 592 | 0.000 | 0.998 | 0.841 |
| 1e-5 | 0.99M | 3.44 | -0.001 | **1.001** | **0.837** |

| quantity | mean (effective support) | notes |
|---|---:|---|
| h^actual | **0.837** | tightly concentrated (std 0.058), 100% in [0.5,1.5] |
| ĥ (predicted) | **1.001** | median 1.001, scalar null |
| A^(1), A^(2), B^(2) | -0.23, -0.23, 0.055 | A1≈A2 cancel |

- **ĥ ≈ 1 on the effective support** (A^(1) ≈ A^(2) ≈ -0.23 cancel), the #45
  Corollary-1 scalar null, now computed with the correct paired-gradient
  formula.
- **h^actual ≈ 0.837 on the same coords** — tightly concentrated and
  systematically below 1.

So both ĥ and h^actual are near-constant but at **different values** (≈1 vs
≈0.837): the scalar chain predicts the null 1, the actual update ratio is 0.837.
Corr ≈ 0 because both are stable at different values — they are stably
different. This is **not a pipeline artifact** (the corrected formula is exact
to machine precision); it is a real non-scalar-gradient effect.

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

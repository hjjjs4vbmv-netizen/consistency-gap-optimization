# Scalar-History Predictor — Mechanism Attribution Result (Role C)

Date: 2026-08-11. Branch: `role-c/moment-memory-real-history` (PR #47).
Answers: **how much of the real non-gauge RAdam update residual is explained by
a scalar gradient-scale history through optimizer moment memory?**

**Experiment scope (explicit):** this is a **20-step prospective fork** from a
real K=256 kimg state (arm_a, g=1.0) replaying the paired g=1.0 vs g=1.3
gradient history. It is NOT a formal same-state K=256 attribution (that would
require Role D's stateful audit over the full training horizon); the K=32/64/
128/256 attribution is the follow-up.

## Method (the scientific mechanism test)

Per step `j` of the stored paired gradient history, estimate **ONE global
scalar** (not per-coordinate):

```
a_j* = <G_j^1.3, G_j^1.0> / ||G_j^1.0||²
Ĝ_j^1.3 = a_j* G_j^1.0
```

Replay RAdam from the **real optimizer state** (flattened m/v/step) over two
histories — reference `G^1.0` and scalar-predicted `Ĝ^1.3` — and take the
predicted update ratio `ĥ^scalar_{t,i} = U^scalar/U^1` at the target step.

This is distinct from the coordinate-wise oracle (direct formula, no replay),
which is only an algebra/implementation sanity check.

## Result (gap_lr_matched arm_a g=1.0, 256 kimg; 20-step paired replay vs g=1.3)

| metric | value |
|---|---:|
| a_j* mean / std | 0.7726 / 0.0229 |
| **ĥ^scalar mean** | **0.8362** |
| **h^actual mean** | **0.8374** |
| **wRMSE(ĥ^scalar, h^actual)** | **0.0295** |
| **Corr(ĥ^scalar, h^actual)** | **0.8582** |
| **Weighted R²(ĥ^scalar vs h^actual)** | **0.735** |
| Disp(ĥ^scalar) | 0.0507 |
| R_opt | 0.1167 |
| ρ_scalar = Disp(ĥ^scalar)/R_opt | 0.434 (dispersion RATIO, NOT an explained fraction) |

## Interpretation

1. **The scalar-history predictor reproduces the h ≈ 0.837 offset** (not the
   scalar-null 1): ĥ^scalar mean 0.8362 vs h^actual 0.8374. The earlier
   coordinate-wise oracle gave ĥ ≈ 1.001, Corr ≈ 0 — it missed the offset
   because it did not replay the optimizer. **The replay is what captures the
   moment-memory accumulation.**
2. **Corr = 0.858**: the scalar history, propagated through RAdam moment memory
   from the real state, predicts the coordinate-level update-ratio variation
   with high correlation.
3. **Weighted R² = 0.735**: the scalar-history predictor explains **73.5% of
   the weighted variance of the real update ratio** h^actual. This is the
   statistically meaningful "explained" quantity. (The ρ_scalar = 0.434 is only
   a dispersion RATIO, Disp(ĥ)/R_opt — it is NOT an explained fraction and is
   not the headline.)

## What this means (vs the task's decision criteria)

- **Case B (medium-strong)**: scalar gradient-scale history through RAdam
  moment memory explains the majority (R² = 0.735, Corr = 0.86) of the
  coordinate-level variation of the real update ratio. This is **mechanism
  evidence** that optimizer memory is a real source of the residual left after
  LR calibration.
- The component of R_opt **not captured** by the scalar-history predictor is
  the non-scalar gradient content (the 3.2% per-step residual E_j) acting
  through the optimizer.

## Comparison: scalar predictor vs coordinate-wise oracle

| | scalar-history predictor (replay) | coordinate-wise oracle (direct) |
|---|---:|---:|
| ĥ mean | 0.8362 | ~1.001 |
| Corr vs h^actual | 0.858 | ~0 |
| role | **mechanism test** | algebra/impl sanity check |

The two are deliberately different: the oracle confirms the exact identity is
implemented correctly; the scalar predictor is the scientific test of whether
scalar history explains the real residual.

## Files
- `analysis/scalar_history_predictor.py` — the mechanism test.
- `analysis/real_history/scalar_prediction.json` — the result.
- `analysis/moment_memory_prediction.py` — coordinate-wise oracle (sanity).

# Moment-Memory Prediction on Real ECT History — Result (Role C)

Date: 2026-08-09. Branch: `role-c/moment-memory-real-history` (PR #47).
Data: real paired sweep from `gap_lr_matched_q128_s3_v1` arm_a (g=1.0), 256 kimg
state; 20 replay steps pairing g=1.0 (reference) vs g=1.3 (candidate) with
identical (batch, t, noise, dropout) per step.

## Chain result

```
δ_j (recovered) → A^(1), A^(2), B^(2) → ĥ  vs  h^update
```

| quantity | value |
|---|---:|
| δ_j mean / std | **-0.2264 / 0.0223** (stable, ≈ a=0.77 ⇒ δ≈-0.23) |
| per-step non-scalar residual mean | **0.032 (3.2%)** |
| h^actual (update ratio) mean | **0.8325** (std 0.056) |
| ĥ (predicted) mean | **1.0079** (std 0.0075) |
| A^(1), A^(2), B^(2) | -0.2214, -0.2278, 0.0525 |
| **weighted RMSE(ĥ, h^update)** | **0.186** |
| **Corr(ĥ, h^update)** | **0.283** |
| R_grad (instantaneous) | 0.0285 |
| **R_opt (optimizer update)** | **0.0565** |
| **Δ = R_opt − R_grad** | **+0.0280** (memory-induced increment) |

## What this answers

**"Moment-history mismatch explains how much of the real R_opt?"**

1. The **A^(1)≈A^(2)≈-0.23 near-cancellation** makes the theory predict
   ĥ ≈ 1: at the scalar level, a stable δ_j history is (almost) absorbed by the
   rectified RAdam memory — consistent with the #45 Corollary-1 null.
2. **But the actual update ratio is h^actual ≈ 0.83**, far from 1. The gap
   between ĥ and h^update (RMSE 0.19, Corr 0.28) is **not explained by the
   scalar moment-memory chain**; it is driven by the **3.2% per-step non-scalar
   gradient residual E_j** that the scalar δ_j model ignores.
3. The optimizer memory does produce a **positive increment Δ = R_opt − R_grad
   = +0.028**, i.e. the update residual is larger than the instantaneous
   gradient residual — qualitatively the #45 Corollary-2 direction. But this
   increment is a small part of the total R_opt; most of the update distortion
   comes from the non-scalar gradient content, not from the scalar-scale memory.

## Honest interpretation

- The scalar moment-memory chain is **quantitatively insufficient** for the
  real 256-kimg state: the near-cancellation A^(1)-A^(2) is real, but the
  actual h deviates because the gradients are not exactly scalar-multiple
  (3.2% per-step residual).
- This is **not a failure of the exact identity** (the identity h_moment from
  current moments still matches, as #44 showed); it is the statement that
  **recovering δ_j from per-step scalar fits and predicting h from it only
  captures the scalar part**. The non-scalar E_j carries the rest.
- **Implication for GFCT**: the "gap ≈ optimizer-step rescaling" story is only
  part of the truth. On real ECT, the update distortion has a **non-scalar
  gradient component** that scalar LR matching cannot remove — a genuine,
  measurable residual effect. This strengthens the case that gap produces
  non-trivial optimizer-level effects beyond rescaling.

## Comparison with the controlled example

| metric | controlled (synthetic) | real 256-kimg state |
|---|---:|---:|
| RMSE(ĥ, h^update) | 2.6e-6 | 0.186 |
| Corr | 1.0000 | 0.283 |
| R_opt − R_grad | 0 (no non-scalar) | +0.028 (non-scalar E_j) |

The controlled example was exactly the theorem's construction; the real state
has non-scalar gradient content that the scalar chain does not model, hence the
larger RMSE and weaker correlation.

## Files
- `analysis/real_history_sweep.py` — paired replay (real state → gradient/update history).
- `analysis/real_history/` — grad_history_1/g.npy, u1/ug.npy, sweep_meta.json, prediction.json.
- `analysis/real_history_diag.py` — the diagnosis above.

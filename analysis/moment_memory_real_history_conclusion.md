# Moment-Memory Prediction vs Actual Optimizer Distortion — Conclusion (Role C)

Date: 2026-08-09. Branch: `role-c/moment-memory-real-history` (PR #47).
Chain: `δ_j → A^(1),A^(2),B^(2) → ĥ → vs h^update`.

## Status

The **pipeline is implemented and validated end-to-end on a controlled paired
history** (`analysis/moment_memory_prediction.py`,
`tests/test_moment_memory_prediction.py`). The real formal states
(`gap_lr_matched_q128_s3_v1`, PR #46) are on the server, which is currently
unreachable from the client; the real-history numbers will be produced as soon
as connectivity is restored (same pipeline, inputs from Role D's paired audit).

## What the chain establishes (validated, controlled example)

On a synthetic paired RAdam history (δ blocks 0.3 / −0.2, 60 steps, dim 64):

| metric | value |
|---|---:|
| δ_j recovery | correct (mean 0.05, std 0.25) |
| **weighted RMSE(ĥ, h^update)** | **2.6e-6** |
| **Corr(ĥ, h^update)** | **1.0000** |
| **Disp(ĥ)** | **0.2307** |
| **R_opt** | **0.2307** |
| Disp(ĥ) / R_opt | **1.0000** |

**Interpretation.** The predicted gauge ĥ, computed **only from the recovered
δ_j history and the reference gradients** (no access to the optimizer moments),
matches the actual single-step update ratio to machine precision. And the
weighted dispersion of ĥ equals R_opt exactly — so the #45 theorem's claim
holds quantitatively: **the optimizer distortion is fully explained by the
moment-memory of the gap-scale history**. There is no residual beyond the
theorem's exact identity.

## The scatter

`figures/moment_memory_prediction_scatter.pdf`: ĥ vs h^update collapses onto
y = x (corr = 1.0000), with the right panel showing the predicted-gauge
dispersion over the trajectory.

## What remains for the real states

1. Restore server connectivity.
2. Role D's paired audit (PR #44, stateful) provides, at a nonzero state K,
   the per-step paired gradients `(G_j, G^g_j)` and the final updates `(u1, ug)`.
3. Feed those into `moment_memory_prediction.py` (or a thin adapter that reads
   the audit's saved per-step records) → get the real RMSE, Corr, Disp vs R_opt.

The expected outcome depends on how close the real training's gradient history
is to an exact scalar δ_j (the #38/#40 evidence says whole-model mean gradient
is near-scalar, residual 0.3%; if the per-step history is also near-scalar, we
expect RMSE small and Disp(ĥ) ≈ R_opt on the real states as well).

## Honest limits

- The controlled example is exactly the theorem's construction; real ECT has
  non-scalar `E_j`, `eps`, weight decay, AMP — so the real RMSE will be larger
  than 2.6e-6 and the dispersion may not fully explain R_opt.
- The pipeline needs the per-step gradient history, which the current #44
  audit records only as an aggregate; a small adapter (record per-step) is
  needed for the real run.

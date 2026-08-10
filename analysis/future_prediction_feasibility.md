# Future-prediction experiment — feasibility assessment (Role C)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. The decisive experiment
for the leader's Go/No-Go: does early optimizer residual predict FUTURE FID
(out-of-sample)? Assessed against the 1-GPU constraint (gpu0003, ECT002).

## What the experiment needs

1. **Multiple seeds (5+) with early checkpoints (32/64 kimg) AND late FID (256 kimg)**
2. Compute early residual (R_grad, R_opt, c_K*) at early checkpoints
3. Predict late FID / gap ranking, leave-one-seed-out
4. Control g, q, budget

## Data availability gap

- arm states (gap_lr_matched_q128_s3_v1): **seed 3 only**, but have 8 ticks
  (32 kimg intervals) with training-state + network-snapshot at each.
- q256 runs (seed 3/4/5 × fixed/global110): 3 seeds, but **only latest
  checkpoint** (no early 32/64k snapshots).
- g_screen: 6 gaps, seed 3 only, 256 kimg FID.

**Conclusion: no existing multi-seed early-checkpoint data. The experiment
requires NEW training.**

## Compute cost (1 GPU, gpu0003)

Training rate: ~40 min per 256 kimg run (from the arm runs, 8 ticks).

| config | runs | training time |
|---|---|---|
| Minimal: 2 gaps (g=1.0, 1.3) × 5 seeds | 10 | ~6.7 h |
| Reuse seed 3 arm states, train 4 more seeds | 8 | ~5.3 h |
| Robust: 3 gaps × 5 seeds | 15 | ~10 h |

Plus overhead:
- Early residual audits: ~90 s each × (runs × 2 early checkpoints) ≈ 30-45 min
- Late FID: ~2 min × (runs × 2 NFE) ≈ 40-60 min

**Total: ~6-8 hours GPU (minimal), feasible in ~1 day on 1 GPU.**

## Feasibility verdict

**FEASIBLE.** The minimal version (reuse seed 3, train 4 more seeds × 2 gaps =
8 runs) is ~5-6 hours of GPU, doable in ~1 day. The robust version (3 gaps × 5
seeds) is ~10-12 hours, ~1.5 days.

## Key risk (honest)

Given the Arm C result (g=1.3 FID improvement is 84-100% trivial LR rescaling),
there is a **real chance the future-prediction experiment gives a NEGATIVE
result**: the early residual may not predict future FID out-of-sample, because
the quality consequence is mostly trivial rescaling. This would confirm the
"optimizer-memory has no meaningful quality consequence" conclusion and settle
the Go/No-Go toward workshop.

The experiment is worth running because it is the ONLY way to settle the
question, and it is feasible (~1 day). But it should be run with the
expectation that a negative result is plausible.

## What to run (if approved)

1. Train 4 new seeds (e.g., 0,1,2,4) × 2 gaps (g=1.0, g=1.3), 256 kimg, with
   early checkpoints (32/64k) — reuse the arm training setup.
2. Compute early residual (R_grad, R_opt, c_K*) at 32/64k via the direct audit
   (run_e7_direct.py).
3. Compute late FID (256k) for all runs.
4. Leave-one-seed-out: fit residual→FID on 4 seeds, predict the held-out seed.
5. Report whether early residual predicts future FID / gap ranking.

## Files
- This assessment: `analysis/future_prediction_feasibility.md`
- Arm C result: `analysis/arm_c_scale_matched_control_result.md`

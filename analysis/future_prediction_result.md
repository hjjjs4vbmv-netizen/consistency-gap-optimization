# Future-prediction experiment — result (Role C)

Date: 2026-08-11. Branch `role-c/g13-vs-g10-fid-0809`. The decisive experiment
for the leader's Go/No-Go: does early gradient residual predict FUTURE FID
(out-of-sample)?

## Setup

- Trained 4 new seeds (0,1,2,4) × 2 gaps (g=1.0, g=1.3), 256 kimg, global_sigmoid,
  lr 1e-4, identical config (only seed + global_gap_scale differ).
- Early gradient residual R_grad (direction_residual at g=1.3, mean over 208
  layers) computed at snapshot 000001 (32 kimg) and 000002 (64 kimg) via
  gap_gradient_hook (gate disabled).
- Future FID: 256 kimg FID-5k (NFE=1, NFE=2), 3 repeats.
- NOTE: early R_opt is NOT available (training only saved the final
  training-state, not early optimizer states). So this is a GRADIENT-LEVEL
  future prediction (R_grad), not optimizer-level (R_opt).

## Data

| seed | gap | R_grad_32k | R_grad_64k | FID_256k (NFE1) | FID_256k (NFE2) |
|---|---|---:|---:|---:|---:|
| 0 | 1.0 | 0.00000 | 0.00000 | 257.7 | 53.21 |
| 0 | 1.3 | 0.00895 | 0.00945 | 222.7 | 53.98 |
| 1 | 1.0 | 0.00000 | 0.00000 | 247.3 | 52.30 |
| 1 | 1.3 | 0.00854 | 0.00968 | 241.2 | 52.81 |
| 2 | 1.0 | 0.00000 | 0.00000 | 313.9 | 82.03 |
| 2 | 1.3 | 0.00956 | 0.00979 | 250.9 | 51.36 |
| 4 | 1.0 | 0.00000 | 0.00000 | 250.1 | 52.41 |
| 4 | 1.3 | 0.00896 | 0.00974 | 219.5 | 53.99 |

## Results

### Q1: Does g=1.3 win FID across seeds?
- NFE1: g=1.3 wins **4/4** seeds (improvement 6.1–63.1 FID)
- NFE2: g=1.3 wins 1/4 seeds (only seed 2; others ~tied, g=1.3 slightly worse)
→ g=1.3's NFE1 advantage is robust across seeds; NFE2 advantage is not.

### Q2: Does early R_grad correlate with future FID (within g=1.3)?
- R_grad_32k vs FID(NFE1): pearson +0.386
- R_grad_64k vs FID(NFE1): pearson +0.527
- R_grad_32k vs FID(NFE2): pearson −0.586
- R_grad_64k vs FID(NFE2): pearson −0.613
→ Signs are inconsistent (positive for NFE1, negative for NFE2). n=4. Not reliable.

### Q3: Does early R_grad predict the MAGNITUDE of g=1.3's improvement?
- R_grad_32k vs (FID_g1.0 − FID_g1.3): pearson +0.992
- R_grad_64k vs improvement: pearson +0.260
→ The +0.992 at 32k is a 4-point coincidence (the residual barely varies: 0.0085–
  0.0096, a 0.001 spread), driven almost entirely by seed 2 having both the
  largest residual (0.00956) and the largest improvement (63.1). Not a reliable
  prediction; with n=4 and near-constant x, one point swings the correlation.

### Q4: Does early R_grad predict the gap ranking?
- Ranking is degenerate: g=1.3 wins 4/4 (NFE1). R_grad cannot discriminate a
  uniform ranking. The discriminative question is magnitude (Q3), which is
  unreliable.

## Honest verdict

**NEGATIVE for the future-prediction thesis.**

- Early gradient residual R_grad is **nearly constant across seeds**
  (0.0085–0.0096 at 32k, a ~0.001 spread) and does NOT vary enough to predict
  the large future-FID variation (improvement ranges 6–63 FID).
- The correlations are **not reliable**: n=4, near-constant predictor, signs
  flip between NFE1/NFE2, and the one high correlation (+0.992) is a single-point
  coincidence (seed 2).
- The future-FID variation across seeds is driven by **seed/training
  stochasticity**, not by the (near-constant) early gradient residual.

## Caveat (important)

This is a **gradient-level** test (early R_grad → future FID). The
**optimizer-level** test (early R_opt → future FID) is NOT available because
training only saved the final training-state, not early optimizer states. Early
R_opt might carry more seed-varying information than R_grad (it depends on the
optimizer state, which varies more). So this negative result does NOT fully rule
out the optimizer-memory future-prediction thesis — it rules out the
gradient-level version.

To fully settle: re-run training saving early training-states (ckpt_ticks=1
already saves snapshots, but training-state saving needs state_dump_ticks),
then compute early R_opt and repeat the leave-one-seed-out.

## Implication for the ICLR plan

This negative result, combined with:
- Arm C (g=1.3 improvement is 84–100% trivial LR rescaling)
- P1 (residual epiphenomenal to |gap−1|, partial corr ≈ 0)

...consistently points to: **the non-scalar gradient residual is a real but
non-consequential phenomenon at the gradient level.** The optimizer-memory
mechanism is real (h_actual≠h_pred) but its quality consequence is largely
trivial rescaling, and its early gradient-level signal does not predict future
FID out-of-sample.

The remaining live possibility is the **optimizer-level** future prediction
(early R_opt), which requires re-training with early training-state saving.
Whether that is worth the compute depends on the leader's judgment.

## Files
- Analysis: `analysis/future_prediction_analysis.py`
- This summary: `analysis/future_prediction_result.md`
- Server data: `/data/raw/ECT/ect_runs/future_pred_0809/` (training, fid/, early_residual2/)

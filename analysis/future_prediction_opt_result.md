# Optimizer-level future prediction — result (Role C)

Date: 2026-08-11. Branch `role-c/g13-vs-g10-fid-0809`. The FINAL decisive test:
does early OPTIMIZER residual (R_opt) predict future FID out-of-sample?

## Setup

- Retrained 4 seeds (0,1,2,4) × 2 gaps (g=1.0, g=1.3), 256 kimg, WITH early
  training-state saving (--dump=1), so early optimizer state is available.
- Early R_opt (optimizer-state residual) computed at 32k (tick1) and 64k (tick2)
  via run_stateful_pair (gate bypassed).
- Future FID at 256k (NFE1) from future_pred_0809 (same seed/config).
- Leave-one-seed-out over 4 seeds.

## Data (g=1.3)

| seed | R_opt_32k | R_opt_64k | FID_256k | improvement |
|---|---:|---:|---:|---:|
| 0 | 0.0184 | 0.0233 | 222.7 | 35.0 |
| 1 | 0.0183 | 0.0209 | 241.2 | 6.1 |
| 2 | 0.0203 | 0.0192 | 250.9 | 63.0 |
| 4 | 0.0207 | 0.0228 | 219.5 | 30.6 |

Cross-seed spread of early R_opt (g=1.3):
- 32k: mean 0.0194, std 0.0011, range [0.0183, 0.0207]
- 64k: mean 0.0215, std 0.0016, range [0.0192, 0.0233]

## Results

| correlation | pearson | spearman |
|---|---:|---:|
| R_opt_32k vs FID(g=1.3) | **−0.004** | −0.400 |
| R_opt_32k vs improvement | +0.587 | +0.400 |
| R_opt_64k vs FID(g=1.3) | −0.975 | −0.800 |
| R_opt_64k vs improvement | −0.400 | −0.200 |

## Honest verdict

**NEGATIVE — the optimizer-level future-prediction thesis FAILS.**

- **Early R_opt is nearly CONSTANT across seeds** (0.0183–0.0207 at 32k, std
  0.0011), just like R_grad was. The cross-seed variation in future FID
  (219–251) and improvement (6–63 FID) is far larger than the predictor spread.
- The correlations are **unreliable**: n=4, near-constant predictor, signs flip
  between 32k/64k and between FID/improvement (R_opt_32k vs FID ≈ 0 but
  R_opt_64k vs FID = −0.975; R_opt_32k vs improvement +0.587 but R_opt_64k vs
  improvement −0.400). The −0.975 is a 4-point coincidence driven by seed 2.
- Early R_opt IS gap-sensitive (diff(1.3−1.0) = +0.005 to +0.008, all positive),
  confirming the mechanism exists. But it does NOT predict future FID
  out-of-sample.

## The fourfold evidence (all consistent)

| experiment | result |
|---|---|
| Arm C (scale-matched control) | g=1.3 improvement is 84–100% trivial LR rescaling |
| P1 (epiphenomenality) | residual epiphenomenal to \|gap−1\| (partial corr ≈ 0) |
| Gradient-level future prediction | early R_grad does NOT predict future FID |
| **Optimizer-level future prediction** | **early R_opt does NOT predict future FID** |

## Go/No-Go

**The leader's thesis (optimizer-memory → future performance) is now FULLY
tested at both the gradient and optimizer level, and both fail.** The
optimizer-memory mechanism is REAL (R_opt gap-sensitive, h_actual≠h_pred), but
its quality consequence is mostly trivial rescaling (Arm C), and neither the
gradient nor optimizer early residual predicts future FID out-of-sample.

**Verdict: No-Go for the ICLR main-track mechanism thesis.** The honest,
publishable contribution is the workshop-level structural characterization:
non-scalar gradient/optimizer content exists, is structured, universal, benign,
and does not predict future quality. This is the final answer.

## Files
- Analysis: `analysis/future_prediction_opt_analysis.py`
- This summary: `analysis/future_prediction_opt_result.md`
- Server data: `/data/raw/ECT/ect_runs/future_pred_opt_0811/`

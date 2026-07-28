# Factorized global/local gap experiment

This directory contains the compact, reproducible summary of the final
global/local gap factorial experiment.

## Design

- Baseline: fixed ECT sigmoid schedule.
- Global-only: multiply every sigmoid gap by one shared scale `g`.
- Local-only: apply a four-bin, raw-pair-loss controller whose local scales
  have geometric mean one before realized-gap clipping effects.
- Global + local: combine the explicit global scale with the geometrically
  normalized local scale factors.
- Global scale selected by the stage-1 response curve: `g* = 1.10`.
- Final matrix: three training seeds, NFE 1 and 2, and conservative/aggressive
  local-controller profiles.

## Main result

Because seed 0 participated in selecting `g*`, the cleanest descriptive
comparison uses held-out seeds 1 and 2. Relative to fixed sigmoid:

| Method | NFE=1 KID-5k | NFE=1 FID-5k | NFE=2 KID-5k | NFE=2 FID-5k |
| --- | ---: | ---: | ---: | ---: |
| Global-only (`g=1.10`) | -2.58% | -4.23% | -19.09% | -18.62% |
| Local conservative | -0.01% | -0.04% | -0.18% | -0.08% |
| Local aggressive | +0.77% | +0.40% | +6.54% | +6.15% |
| Global + aggressive local | -2.87% | -4.39% | -20.18% | -19.42% |

Lower KID/FID is better, so negative percentages indicate improvement. The
dominant contribution comes from global gap calibration. The aggressive local
controller adds only a small improvement on top of global calibration and is
harmful when used alone at NFE=2.

Each headline percentage compares the arithmetic metric means:

`100 × (mean(metric_arm, seeds 1/2) / mean(metric_fixed, seeds 1/2) - 1)`.

It is not the mean of the two per-seed percentage changes.

### NFE=2 seed sensitivity

Both held-out seeds improve directionally under global-only and global plus
aggressive-local, but the magnitudes differ substantially:

| Method | Metric | Seed 1 change | Seed 2 change | Seed 1 share of absolute two-seed decrease |
| --- | --- | ---: | ---: | ---: |
| Global-only | KID-5k | -70.50% | -5.26% | 78.30% |
| Global-only | FID-5k | -60.83% | -4.12% | 83.55% |
| Global + aggressive local | KID-5k | -69.86% | -6.82% | 73.38% |
| Global + aggressive local | FID-5k | -61.17% | -5.07% | 80.55% |

The approximately 19–20% held-out-mean NFE=2 improvement is therefore strongly
influenced by seed 1 and should not be described as a uniform 20% per-seed
effect.

## Realized-gap diagnostics

The implementation now records:

- `gap_over_sigmoid_gap_mean`: batch mean of
  `(t - r_realized) / (t - r_sigmoid)`;
- `lower_gap_clip_rate`: fraction for which the lower-gap constraint increases
  the realized gap above the pre-clipping factorized target;
- `upper_gap_clip_rate`: fraction for which the `r >= 0` constraint reduces the
  realized gap below the pre-clipping factorized target.

The completed 2026-07-27 runs predate these telemetry fields. Their exact
training-time values cannot be reconstructed from the compact metric summaries,
so they are reported as `not_recorded_pre_instrumentation` rather than
estimated. Future runs write all three values to `train_summary.csv` and
`stats.jsonl`.

## Files

- `factorial_summary.md`: full paired-effect definitions, three-seed results,
  confidence intervals, and interpretation caveats.
- `factorial_summary.csv` / `.json`: machine-readable aggregate effects.
- `heldout_headlines.csv`: arithmetic-mean headline calculations and per-seed
  sensitivity for held-out seeds 1/2.
- `per_cell_metrics.csv`: every evaluated cell.
- `per_seed_effects.csv`: every paired per-seed contrast.

These are 5,000-sample proxy metrics, not standard 50,000-sample benchmarks.
With only three seeds, confidence intervals are wide; results should be treated
as descriptive rather than population-level significance claims.

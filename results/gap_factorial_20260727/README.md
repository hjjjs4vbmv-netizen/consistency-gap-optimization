# Factorized global/local gap experiment

This directory contains the compact, reproducible summary of the final
global/local gap factorial experiment.

## Design

- Baseline: fixed ECT sigmoid schedule.
- Global-only: multiply every sigmoid gap by one shared scale `g`.
- Local-only: apply a four-bin, raw-pair-loss controller whose local scales
  have geometric mean one.
- Global + local: combine the explicit global scale with the geometrically
  neutral local controller.
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

## Files

- `factorial_summary.md`: full paired-effect definitions, three-seed results,
  confidence intervals, and interpretation caveats.
- `factorial_summary.csv` / `.json`: machine-readable aggregate effects.
- `per_cell_metrics.csv`: every evaluated cell.
- `per_seed_effects.csv`: every paired per-seed contrast.

These are 5,000-sample proxy metrics, not standard 50,000-sample benchmarks.
With only three seeds, confidence intervals are wide; results should be treated
as descriptive rather than population-level significance claims.

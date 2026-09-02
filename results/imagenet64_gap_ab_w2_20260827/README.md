# ImageNet-64 W2 gap A/B results

## Result

The comparison is paired by training seed and reported as `IA - IB` for both
FID50k and KID50k. Lower is better, so a negative delta favors IA. IA is the
`global_gap_scale=1.0` arm and IB is the `global_gap_scale=1.1` arm.

Across three paired training seeds, IA has lower FID and KID in all 30 repeated
seed-by-checkpoint-by-NFE comparisons through 6,400 kimg. These cells comprise
five checkpoints and two NFE settings within each seed; the training seed is
the independent unit. At 7,680 kimg the gap is small and seed-dependent. The
NFE1 mean still favors IA, while the NFE2 mean favors IB. Across the complete
descriptive matrix through 7,680 kimg, IA has lower values in 33 of 36 pairs
for each metric.

The 8,960-kimg extension contains only one complete IA/IB pair (seed 102), so
it is descriptive rather than a replicated conclusion. For that seed, IA has
lower FID at both NFEs and lower KID at NFE1; IB has slightly lower KID at
NFE2.

| kimg | NFE | pairs | mean IA FID | mean IB FID | mean delta FID | SD delta FID | mean IA KID | mean IB KID | mean delta KID | SD delta KID |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,280 | 1 | 3 | 71.0832 | 75.5328 | -4.4496 | 0.4552 | 0.051304 | 0.055056 | -0.003752 | 0.000474 |
| 1,280 | 2 | 3 | 29.3503 | 32.2883 | -2.9381 | 0.1106 | 0.017708 | 0.019597 | -0.001890 | 0.000089 |
| 2,560 | 1 | 3 | 65.4124 | 70.1823 | -4.7698 | 0.6230 | 0.047024 | 0.050876 | -0.003852 | 0.000573 |
| 2,560 | 2 | 3 | 28.1867 | 31.2395 | -3.0528 | 0.2813 | 0.016982 | 0.018926 | -0.001944 | 0.000204 |
| 3,840 | 1 | 3 | 28.2718 | 31.6354 | -3.3636 | 0.3363 | 0.018101 | 0.020414 | -0.002313 | 0.000329 |
| 3,840 | 2 | 3 | 8.7476 | 9.7703 | -1.0227 | 0.0900 | 0.004816 | 0.005459 | -0.000643 | 0.000088 |
| 5,120 | 1 | 3 | 22.1484 | 24.5344 | -2.3859 | 0.2699 | 0.013914 | 0.015529 | -0.001615 | 0.000226 |
| 5,120 | 2 | 3 | 7.6012 | 8.4164 | -0.8153 | 0.0960 | 0.004156 | 0.004637 | -0.000481 | 0.000088 |
| 6,400 | 1 | 3 | 21.5476 | 25.0607 | -3.5131 | 2.4105 | 0.013480 | 0.015816 | -0.002336 | 0.001700 |
| 6,400 | 2 | 3 | 7.5166 | 8.5955 | -1.0788 | 0.5755 | 0.004050 | 0.004604 | -0.000554 | 0.000289 |
| 7,680 | 1 | 3 | 9.3517 | 9.6072 | -0.2555 | 0.5284 | 0.005810 | 0.005924 | -0.000115 | 0.000329 |
| 7,680 | 2 | 3 | 4.0417 | 3.7882 | +0.2535 | 0.5711 | 0.002239 | 0.002109 | +0.000130 | 0.000246 |
| 8,960 | 1 | 1 | 8.4808 | 8.7404 | -0.2595 | - | 0.005446 | 0.005530 | -0.000083 | - |
| 8,960 | 2 | 1 | 3.8192 | 3.8621 | -0.0430 | - | 0.002342 | 0.002308 | +0.000034 | - |

### Endpoint seed effects

The replicated 7,680-kimg endpoint is not directionally uniform. Seed 101
favors IA at both NFEs and both metrics; seed 103 favors IB at both NFEs and
both metrics. Seed 102 favors IA at NFE1 and is effectively split at NFE2.

| kimg | seed | NFE | delta FID IA-IB | delta KID IA-IB |
|---:|---:|---:|---:|---:|
| 7,680 | 101 | 1 | -0.752467 | -0.000394321 |
| 7,680 | 101 | 2 | -0.148906 | -0.000072271 |
| 7,680 | 102 | 1 | -0.313674 | -0.000198649 |
| 7,680 | 102 | 2 | +0.002259 | +0.000059085 |
| 7,680 | 103 | 1 | +0.299589 | +0.000248299 |
| 7,680 | 103 | 2 | +0.907177 | +0.000403453 |
| 8,960 | 102 | 1 | -0.259537 | -0.000083412 |
| 8,960 | 102 | 2 | -0.042976 | +0.000033775 |

## Frozen protocol

- Training seeds: 101, 102, and 103, paired between IA and IB.
- Training: batch 128, per-GPU batch 32, two GPUs per trajectory, one data
  worker, FP32, TF32 disabled, and AMP disabled.
- Replicated checkpoints: 1,280 through 7,680 kimg in 1,280-kimg increments.
- Extension: seed101-IA, seed102-IA, and seed102-IB at 8,960 kimg. Only seed
  102 forms an IA/IB pair at this endpoint.
- Evaluation: 50,000 generated examples per trajectory/checkpoint/NFE, fixed
  generation seeds 0-49,999, fixed label sequence, and NFE 1 or 2.
- FID: official ImageNet-64 `img64.pkl` statistics.
- KID: 100 subsets of size 1,000, fixed random seed 20260730, using the same
  canonical real ImageNet feature bank for every job.
- Numeric environment for the unified recomputation: NumPy 1.26.4 and SciPy
  1.13.1.

The evaluation scope contains 78 unique rows: 72 rows from six paired
trajectories at six replicated checkpoints and two NFEs, plus six rows from
the three 8,960-kimg extensions. It yields 38 paired IA/IB rows and 14
checkpoint/NFE summaries.

## Data quality and files

All 78 expected keys are present and unique, all 156 metric values are finite,
and every row records the SHA-256 of its generated feature array. Previously
recorded feature hashes were checked against the current arrays with no
identity mismatch. An older scalar cache showed a small BLAS/LAPACK-dependent
FID drift, so no old and new scalar environments were mixed: all 78 FID and
KID results in this directory were recomputed together in the numeric
environment above.

- `scoring_results.json`: complete per-job values and feature/reference hashes.
- `per_trajectory.csv`: flat 78-row FID/KID table.
- `paired_differences.csv`: 38 same-seed `IA - IB` comparisons.
- `paired_summary.csv`: mean deltas, sample SDs, and directional win counts.

These are three-seed descriptive estimates, not a basis for strong
distributional or significance claims. The single-pair 8,960-kimg extension
is especially limited.

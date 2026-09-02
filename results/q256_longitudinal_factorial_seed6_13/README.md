# q256 seed6--13 longitudinal factorial results

This package records the frozen 50k-sample FID/KID evaluation for the q256
continuation study completed on 2026-08-24.

## Coverage and protocol

- 336/336 evaluation receipts are `PASS`.
- Seeds 6--7 contribute arms C/D; seeds 8--13 contribute complete A/B/C/D
  factorial trajectories.
- Each trajectory is evaluated at 384, 512, 640, 768, 896, and 1024 kimg,
  with NFE1 and NFE2 (`mid_t=0.821`).
- Every job uses 50,000 samples, sample seeds `0-49999`, and metric seed
  `20260730`.
- FID and KID share the generated Inception features within each job.  The
  validated receipts bind the checkpoint and generated-feature SHA-256 values.

The balanced factorial analyses below use seeds 8--13 only.  Seeds 6--7 are
reported separately because this continuation added only C/D for those seeds.

## Main longitudinal result

The normalized FID AULC is the trapezoidal average over 384--1024 kimg; lower
is better.  Multiplying it by 640 gives the unnormalized area.

| NFE | A baseline | B complete | C target-only | D denominator-only |
|---:|---:|---:|---:|---:|
| 1 AULC | 36.138 | 34.689 | 37.898 | 35.027 |
| 2 AULC | 32.740 | **18.852** | 31.942 | 21.221 |
| 1 FID at 1024 | 10.013 | **8.917** | 9.981 | 9.187 |
| 2 FID at 1024 | 3.045 | **2.914** | 3.082 | 2.930 |

NFE2 exposes the clearest compute-to-quality separation.  B and D improve the
whole learning path, while the target-only C effect is small.  The four arms
are much closer at 1024 kimg, so endpoint-only reporting understates the
finite-budget effect.

## Paired factorial contrasts

Negative values favor the intervention because lower FID AULC is better.

| NFE | B-A | C-A | D-A | B-C-D+A |
|---:|---:|---:|---:|---:|
| 1 | -1.449 | +1.760 | -1.111 | -2.099 |
| 2 | **-13.888** | -0.797 | **-11.519** | -1.572 |

For NFE2, D-A is negative in 6/6 paired seeds and B-A in 5/6.  The
deterministic paired bootstrap intervals for their means are respectively
`[-26.763, -2.330]` and `[-29.732, -2.992]`.  C-A and the outcome-level
interaction both have intervals crossing zero.

`B-C-D+A` is a separately-trained, outcome-level interaction statistic.  It
is not an objective-level causal decomposition.

## Sustained time to FID <= 10

All NFE2 trajectories attain the threshold.  Median sustained times are 640
kimg for A, 576 for B, 640 for C, and 640 for D.  Thus the complete arm enters
the sustained quality region about 64 kimg earlier at the median.

At NFE1, A/C attain the threshold in 3/6 seeds and B/D in 4/6.  Component and
arm rankings therefore remain seed-, horizon-, and NFE-dependent.

## Seed-level heterogeneity

- NFE2 AULC winners split evenly: B wins 3 seeds and D wins 3.
- NFE2 1024-kimg endpoint winners split B/C/D at 2 seeds each.
- NFE1 endpoint winners are B in 4 seeds, C in 1, and D in 1.

The result supports a denominator-dominant NFE2 learning-path effect without
claiming a universal arm ranking.

For the seed6--7 C/D extension, D has lower mean AULC than C by 2.786 FID
units at NFE1 and 5.507 at NFE2.  NFE2 remains heterogeneous: seed6 slightly
favors C in AULC, whereas seed7 strongly favors D.

## Files

- `evaluation_receipts.json`: consolidated, lossless copy of all 336 receipts.
- `evaluation_results.csv`: tidy per-job metrics and content hashes.
- `longitudinal_summary.csv`: balanced seed8--13 arm-by-budget aggregates.
- `aulc_per_seed.csv`: per-seed AULC, endpoint, and sustained-threshold data.
- `factorial_contrasts.csv`: per-seed AULC and endpoint factorial contrasts.
- `factorial_summary.json`: aggregate contrasts, bootstrap intervals, winner
  counts, and seed6--7 C/D extension summary.
- `SHA256SUMS.txt`: hashes of the generated result package.

Regenerate the package with:

```bash
python scripts/analyze_q256_longitudinal_factorial.py \
  --receipts-dir /path/to/receipts \
  --output-dir results/q256_longitudinal_factorial_seed6_13
```

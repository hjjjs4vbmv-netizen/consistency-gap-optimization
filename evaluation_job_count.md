# Frozen matrix evaluation job count

An evaluation **job** is one checkpoint at one NFE setting. Each job runs two
metrics, so every job produces two metric records. This count excludes an
optional single-checkpoint smoke, which is diagnostic only and is not part of
either frozen matrix.

| Matrix | Budget (kimg) | Checkpoints | NFE settings | Jobs | Metrics/job | Metric records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q=256 budget | 512 | 6 | 2 | 12 | 2 (KID/FID-5k) | 24 |
| q=256 budget | 768 | 6 | 2 | 12 | 2 (KID/FID-5k) | 24 |
| q=256 budget | 1024 | 6 | 2 | 12 | 2 (KID/FID-50k) | 24 |
| q=256 budget subtotal | — | 18 | 2 | 36 | — | 72 |
| q=128 confirmatory | 256 | 6 | 2 | 12 | 2 (KID/FID-50k) | 24 |
| **Total** | — | **24** | — | **48** | — | **96** |

The 24 checkpoints are exactly seeds 3/4/5 × fixed/global110 across the
predeclared budgets. All 48 jobs are required: a failed or missing job leaves
its matrix incomplete. The six q=256 1024-kimg jobs per NFE (12 jobs total)
and all q=128 jobs are formal; quick results from the q=256 512/768-kimg
contracts cannot decide the q=256 formal set.

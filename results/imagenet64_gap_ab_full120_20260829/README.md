# ImageNet-64 full-120 gap A/B results

## Result

This directory extends the W2 report to the complete frozen matrix: three
training seeds, IA and IB, ten checkpoints from 1,280 through 12,800 kimg,
and NFE 1 or 2. Differences are reported as `IA - IB`; lower FID and KID are
better, so a negative difference favors IA.

IA has a strong and consistent advantage through 6,400 kimg. It wins all 30
same-seed comparisons in that range on both FID and KID. Across the complete
matrix it wins 50 of 60 FID pairs and 46 of 60 KID pairs, but those totals
should not be read as a uniform late-training advantage.

The main late-training finding is seed heterogeneity. Seeds 101 and 102 remain
well behaved and continue improving, reaching NFE2 FID values near 3.25--3.31
at 12,800 kimg. Seed 103 becomes unstable after 7,680 kimg in both arms, with
especially severe degradation for IB at 8,960 kimg. Consequently, the
three-seed means after 8,960 kimg are dominated by the unstable seed rather
than representing the typical behavior of all three runs.

The best individual row in the frozen matrix is seed 101, IB, 12,800 kimg,
NFE2: FID50k 3.254632 and KID50k 0.001893475. This is essentially tied in FID
with seed 101 IA at the same checkpoint and NFE (3.255185), while IB has the
lower KID for that pair.

### Mean paired differences

| kimg | NFE1 delta FID | NFE1 delta KID | NFE2 delta FID | NFE2 delta KID |
|---:|---:|---:|---:|---:|
| 1,280 | -4.449642 | -0.003751526 | -2.938056 | -0.001889720 |
| 2,560 | -4.769837 | -0.003851843 | -3.052823 | -0.001943525 |
| 3,840 | -3.363599 | -0.002312961 | -1.022706 | -0.000643099 |
| 5,120 | -2.385949 | -0.001615112 | -0.815274 | -0.000480944 |
| 6,400 | -3.513060 | -0.002335952 | -1.078848 | -0.000554444 |
| 7,680 | -0.255517 | -0.000114890 | +0.253510 | +0.000130089 |
| 8,960 | -103.491881 | -0.122601946 | -79.399791 | -0.081520524 |
| 10,240 | -16.257945 | -0.027993960 | +13.139957 | +0.001139537 |
| 11,520 | +17.082985 | +0.031361935 | +10.975258 | +0.012254395 |
| 12,800 | +12.438636 | +0.015751068 | +12.043924 | +0.009670329 |

The extreme late mean differences reflect seed 103's instability. The
per-seed rows in `per_trajectory.csv` and `paired_differences.csv` should be
used when interpreting those checkpoints.

## Frozen protocol

- Training seeds: 101, 102, and 103, paired between IA and IB.
- IA: `global_gap_scale=1.0`; IB: `global_gap_scale=1.1`.
- Checkpoints: 1,280 through 12,800 kimg in 1,280-kimg increments.
- Evaluation: 50,000 generated examples per job, fixed generation seeds
  0--49,999, fixed labels, FP32, and NFE 1 or 2.
- FID: official ImageNet-64 `img64.pkl` statistics.
- KID: 100 subsets of size 1,000, metric seed 20260730, using one canonical
  real ImageNet feature bank for every job.

## Data quality and files

All 120 expected keys are present and unique. All 240 FID/KID values are
finite, and every result row records the generated-feature SHA-256. The
published summaries contain 60 same-seed IA/IB pairs and 20 checkpoint/NFE
groups.

- `scoring_results.json`: complete per-job FID50k and KID50k values.
- `per_trajectory.csv`: flat 120-row metric table.
- `paired_differences.csv`: 60 same-seed `IA - IB` comparisons.
- `paired_summary.csv`: three-seed means, sample standard deviations, and win
  counts for each checkpoint and NFE.
- `training_provenance.csv`: verified parent-resume and 10,240-kimg full-state
  identities, counters, schema, RNG/sampler presence, and pair-config checks.
- `TRAINING_PROVENANCE.md`: provenance interpretation, receipt basis, and the
  IA/IB configuration-equality audit.

The results are descriptive estimates from three seeds. The observed seed 103
instability is part of the frozen result and has not been filtered or used to
alter the training or stopping protocol.

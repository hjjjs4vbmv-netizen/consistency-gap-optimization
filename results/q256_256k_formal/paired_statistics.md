# Fixed vs global-only paired robustness statistics

Pairing key: `training_seed + budget_kimg + nfe + metric`.
Delta: `global_only - fixed`; negative values favor global-only.
Relative improvement: `100 × (fixed - global_only) / fixed`; positive values favor global-only.
Independent units are training seeds; the pair count is reported for each metric/NFE stratum.

The exact two-sided sign test is reported only as a low-resolution directional check. Bootstrap intervals resample these same seeds and are descriptive sensitivity intervals, not additional independent-sample inference.

## Paired effect summary

| Metric | Budget (kimg) | NFE | Pairs | Arithmetic relative improvement | Geometric relative improvement | Median delta | Worst-case improvement | Seed CV | Rank consistency (Spearman) | Wins | Exact sign p (two-sided) | Bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fid50k_full | 256 | 1 | 3 | 4.544298% | 4.547537% | -12.485467394 | 3.768644% | 21.164185% | 1.000000 | 3/0/0 | 0.250000 | [3.768644, 5.620427]% |
| fid50k_full | 256 | 2 | 3 | 47.441515% | 55.593373% | -167.596988744 | 3.885378% | 80.519432% | 0.500000 | 3/0/0 | 0.250000 | [3.885378, 75.249473]% |
| kid50k_full | 256 | 1 | 3 | 7.501846% | 7.531510% | -0.023481160 | 5.023325% | 38.111785% | 1.000000 | 3/0/0 | 0.250000 | [5.023325, 10.629779]% |
| kid50k_full | 256 | 2 | 3 | 50.599851% | 61.818842% | -0.201894470 | 0.249718% | 86.826602% | 0.500000 | 3/0/0 | 0.250000 | [0.249718, 81.146694]% |

## Leave-one-seed-out arithmetic relative improvement

| Metric | NFE | Omitted seed | Retained pairs | Mean relative improvement | Global/fixed/tie wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| fid50k_full | 1 | 3 | 2 | 4.932124% | 2/0/0 |
| fid50k_full | 1 | 4 | 2 | 4.006233% | 2/0/0 |
| fid50k_full | 1 | 5 | 2 | 4.694535% | 2/0/0 |
| fid50k_full | 2 | 3 | 2 | 33.537536% | 2/0/0 |
| fid50k_full | 2 | 4 | 2 | 39.567426% | 2/0/0 |
| fid50k_full | 2 | 5 | 2 | 69.219583% | 2/0/0 |
| kid50k_full | 1 | 3 | 2 | 7.826552% | 2/0/0 |
| kid50k_full | 1 | 4 | 2 | 5.937880% | 2/0/0 |
| kid50k_full | 1 | 5 | 2 | 8.741107% | 2/0/0 |
| kid50k_full | 2 | 3 | 2 | 35.326430% | 2/0/0 |
| kid50k_full | 2 | 4 | 2 | 40.698206% | 2/0/0 |
| kid50k_full | 2 | 5 | 2 | 75.774917% | 2/0/0 |

## NFE effect heterogeneity

Effect change is the per-seed relative improvement at NFE=2 minus that at NFE=1, in percentage points. Positive values indicate a larger global-only advantage at NFE=2.

| Metric | Pairs | Mean change | Median change | Range | NFE=2 larger / NFE=1 larger / ties | Exact sign p (two-sided) | Bootstrap 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fid50k_full | 3 | 42.897217 pp | 57.569266 pp | [-0.358444, 71.480829] pp | 2 / 1 / 0 | 1.000000 | [-0.358444, 71.480829] pp |
| kid50k_full | 3 | 43.098004 pp | 59.773362 pp | [-4.773607, 74.294259] pp | 2 / 1 / 0 | 1.000000 | [-4.773607, 74.294259] pp |

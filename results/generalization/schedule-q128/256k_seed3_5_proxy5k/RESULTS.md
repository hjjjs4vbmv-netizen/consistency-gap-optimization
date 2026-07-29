# q=128 Generalization Screening at 256 kimg

> Preliminary KID/FID-5k proxy evaluation. This is not a formal FID-50k result.

Comparison: fixed sigmoid versus global-only gap scale 1.10.
Training seeds: 3, 4, 5. Lower metric values are better.

| NFE | Metric | Fixed mean | Global-only mean | Delta (global-fixed) | Global wins |
|---:|---|---:|---:|---:|---:|
| 1 | fid5k_full | 256.37801 | 257.44015 | 1.062141 | 2/3 |
| 1 | kid5k_full | 0.25682499 | 0.25911209 | 0.0022871097 | 2/3 |
| 2 | fid5k_full | 64.588932 | 60.767211 | -3.821721 | 2/3 |
| 2 | kid5k_full | 0.049424714 | 0.045031207 | -0.0043935068 | 2/3 |

Negative paired delta favors global-only. Each training-seed cell is the mean of three metric repetitions.

These results are screening evidence only because q=128 has not yet been formally approved and the experiment used the unmerged PR #23 source archive.

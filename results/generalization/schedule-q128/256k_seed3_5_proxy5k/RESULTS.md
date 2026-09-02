# Legacy Retrospective q=128 Screening at 256 kimg

> Legacy retrospective exploratory KID/FID-5k proxy evidence produced from a pre-merge implementation. This is not confirmatory generalization evidence or a formal FID-50k result.

Comparison: fixed sigmoid versus global-only gap scale 1.10.
Training seeds: 3, 4, 5. Lower metric values are better.

| NFE | Metric | Fixed mean | Global-only mean | Delta (global-fixed) | Global wins |
|---:|---|---:|---:|---:|---:|
| 1 | fid5k_full | 256.37801 | 257.44015 | 1.062141 | 2/3 |
| 1 | kid5k_full | 0.25682499 | 0.25911209 | 0.0022871097 | 2/3 |
| 2 | fid5k_full | 64.588932 | 60.767211 | -3.821721 | 2/3 |
| 2 | kid5k_full | 0.049424714 | 0.045031207 | -0.0043935068 | 2/3 |

Negative paired delta favors global-only. Each cell uses one generated 5k sample set. The reported mean and recomputation standard deviation summarize three numerical recomputations on that same sample set; they are not independent sampling repetitions. The statistical unit is the training seed (`n=3`).

The q=128 setting was not formally frozen before its results were observed. The training source differs materially from the reference merged implementation, and canonical dataset-content equivalence was unavailable. These results are therefore retained only as legacy retrospective exploratory screening evidence.

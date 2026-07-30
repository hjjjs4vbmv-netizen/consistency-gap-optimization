# q=256 confirmatory formal results

This package is the completed, predeclared three-seed (3/4/5) fixed-sigmoid versus global-only-sigmoid (`g=1.10`) q=256 comparison. It contains all 24 formal metric records: 6 checkpoints × 2 NFE modes × 2 metrics. Both KID-50k and FID-50k are lower-is-better.

All evaluations used 50,000 generated samples (seeds `0-49999`), metric seed `20260730`, and FP32. The evaluator commit is `8375d46ca4c65e85ab399fcf1effe22ebb766790`; the dataset SHA-256 is `9fd64620e37bfc0c995535fa52701c9641bcd07635008bfda0c9fbddde1a4ed6`. Full portable runtime and checkpoint provenance is in `environment_manifest.json`.

## Package contents

| File | Purpose |
| --- | --- |
| `evaluation_results.csv` | 24 long-form absolute metric records, including checkpoint IDs and SHA-256s. `run_id` is a portable logical identifier. |
| `paired_differences.csv` | 12 seed-level global-only minus fixed differences, with both absolute values and checkpoint provenance. |
| `paired_statistics.json` | Paired descriptive statistics derived from the difference CSV. |
| `environment_manifest.json` | Frozen evaluator environment, data identity, NFE settings, and six checkpoint identities without machine paths. |

## Per-seed absolute values

| Metric | NFE | Seed | Fixed | Global-only |
| --- | ---: | ---: | ---: | ---: |
| KID-50k | 1 | 3 | 0.342668861 | 0.319187701 |
| KID-50k | 1 | 4 | 0.340228200 | 0.304062694 |
| KID-50k | 1 | 5 | 0.313131928 | 0.297402292 |
| FID-50k | 1 | 3 | 320.589536040 | 308.507658018 |
| FID-50k | 1 | 4 | 319.965575461 | 301.982144550 |
| FID-50k | 1 | 5 | 294.203365273 | 281.717897879 |
| KID-50k | 2 | 3 | 0.296107829 | 0.055826116 |
| KID-50k | 2 | 4 | 0.286769122 | 0.084874652 |
| KID-50k | 2 | 5 | 0.034109969 | 0.034024790 |
| FID-50k | 2 | 3 | 280.897620128 | 69.523640469 |
| FID-50k | 2 | 4 | 265.228363223 | 97.631374479 |
| FID-50k | 2 | 5 | 45.804373638 | 44.024700620 |

## Descriptive summary

Values are mean ± sample SD across the three training seeds. The paired delta is `global_only - fixed`; negative favors global-only. Every number below is recomputable from the CSV files.

| Metric | NFE | Fixed mean ± SD | Global-only mean ± SD | Mean paired delta ± SD | Global-only wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| KID-50k | 1 | 0.332009663 ± 0.016394080 | 0.306884229 ± 0.011163413 | -0.025125434 ± 0.010316682 | 3 / 3 |
| FID-50k | 1 | 311.586158925 ± 15.057173309 | 297.402566815 ± 13.969689018 | -14.183592109 ± 3.296938320 | 3 / 3 |
| KID-50k | 2 | 0.205662306 ± 0.148642041 | 0.058241853 ± 0.025510860 | -0.147420454 ± 0.129031614 | 3 / 3 |
| FID-50k | 2 | 197.310118996 ± 131.441525251 | 70.393238523 ± 26.813914692 | -126.916880474 ± 110.560376043 | 3 / 3 |

Use `metric_value` grouped by `metric_name`, `nfe`, and `method` in `evaluation_results.csv` to calculate absolute-value means and sample SDs (`n - 1` denominator). Use `delta` in `paired_differences.csv`, grouped by `metric` and `nfe`, for the paired columns. The complete-precision source values are in the CSVs; Markdown values are rounded to nine decimal places.

This is descriptive paired evidence (`n=3` independent training seeds), not a significance claim.

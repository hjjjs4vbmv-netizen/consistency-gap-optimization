# q256 terminal-history n=30 final result

## Outcome

The frozen primary estimand was
`log(FID50k_BA) - log(FID50k_AA)` at 1024 kimg and NFE=1. Among the 26
complete AA/BA seed pairs, the mean contrast was `-0.0899461` (95% Student-t
CI `[-0.1298864, -0.0500057]`, `t(25)=-4.6381`, two-sided
`p=9.5201e-05`). The paired standardized effect was `dz=-0.9096`; 22 of 26
pairs favored BA. On the geometric FID scale this corresponds to an estimated
BA/AA ratio of `0.91398`, or an `8.60%` reduction for BA.

The two-sided 90% CI was `[-0.1230718, -0.0568203]`. It lies outside the
practical-equivalence band `±log(1.03)=±0.0295588`, so TOST did not establish
equivalence (`p_TOST=0.9977`). Under the preregistered decision precedence the
final classification is therefore **DIRECTIONAL_NEGATIVE**, not practical
equivalence or inconclusive.

## Coverage and missingness

- Planned: 30 consecutive seeds, 60 AA/BA endpoints.
- Successfully trained and evaluated: 55 endpoints.
- Complete AA/BA pairs: 26.
- Scientific numerical failures retained without retry or replacement:
  `seed58-AA`, `seed58-BA`, `seed65-AA`, `seed67-AA` (upstream prefix-A
  failure), and `seed68-AA`.
- Planned-endpoint failure rates: AA `4/30` (13.3%), BA `1/30` (3.3%);
  Fisher exact two-sided `p=0.3533`.
- Prefix-history failures: A `1/30`, B `0/30`; Fisher exact two-sided `p=1.0`.

The primary result is a complete-case estimand. The concentration of missing
endpoints in AA is scientifically reportable informative missingness even
though the small failure-count comparison is not statistically significant.

## Metric summaries

| Arm | Available n | Mean KID50k | Mean FID50k |
|---|---:|---:|---:|
| AA | 26 | 0.006224921 | 9.855882 |
| BA | 29 | 0.005544038 | 8.972883 |

The mean paired KID contrast (BA minus AA) was `-0.000662688`.

## Integrity and artifacts

All 55 evaluated endpoints passed the checkpoint, binding, receipt, decoded
metric, and shared generated-feature hash chain. The first-wave workers and all
four second-wave workers are sealed PASS; no evaluation job failed. The
second-wave static assignments cover all 30 planned endpoints exactly once.

Machine-readable results are in [`final_results/`](final_results/):

- `combined_results.csv` and `.json`: all 55 valid endpoints;
- `paired_results.csv`: the 26 complete pairs;
- `scientific_failures.csv`: the five preserved numerical failures;
- `statistics.json`: primary CI, TOST, effect size, arm summaries, and failure
  estimands;
- `integrity_verification.json`: endpoint-level evidence-chain audit;
- `VALIDATION_REPORT.md`: statistical interpretation and 11/11 fallacy scan;
- `SHA256SUMS.txt`: hashes for the compact final result set.

The full operational audit archive is intentionally not committed because it
is 252 MB. Its SHA-256 is
`940ed69968dd229ecb020f666d7e46c3e38315d159e8133f27ea29688778a513`.
The full 463 GB retained data store includes every prefix and
640/768/896/1024 network/training-state checkpoint plus evaluation artifacts.

Artifact integrity is verified, but no independent full rerun was performed;
the reproducibility verdict remains `CANNOT_VERIFY` rather than
`REPRODUCIBLE`.

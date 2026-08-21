# q256 seed14–18 secondary seed/sensitivity extension

## Status and scope

This is a post-preregistration secondary seed/sensitivity extension of the
q256 target-geometry × denominator-weighting factorial.  It contains five
independent training seeds (14–18), four arms (A/B/C/D), and the frozen NFE1
and NFE2 evaluation readouts.  It does not replace, enlarge, or retrospectively
relabel the preregistered seed3/4/5 cohort.

Training source was frozen at `dcca41b19e7c45512b5fbe98776520396a1bf9ac`.
Evaluation source was frozen at
`d6aba02fb88e9db0993623895eb2228ed717d810`.  All 40 evaluation jobs passed:
FP32, 50,000 samples, sample seeds `0..49999`, metric seed `20260730`, KID then
FID from byte-identical retained generated features, and `mid_t=0.821` for
NFE2.  Earlier failed orchestration attempts are preserved on the experiment
server and are not counted as results; only the fresh v4 native-runtime PASS
chain is included here.

## Full-intervention sensitivity

Lower is better.  The full intervention is B versus baseline A.

| Readout | B−A mean, seed14–18 | Favorable seeds |
|:---|---:|:---:|
| FID-50k @ NFE1 (primary) | -18.510300 | 4/5 |
| KID-50k @ NFE1 | -0.019390206 | 3/5 |
| FID-50k @ NFE2 | -9.954243 | 3/5 |
| KID-50k @ NFE2 | -0.014153901 | 3/5 |

Seed17 reverses B−A for both FID and KID at both NFE modes.  Seed18 also has a
small adverse B−A reversal at NFE2.  The new cohort therefore weakens a claim
that the full intervention is uniformly favorable, even though the mean B−A
contrast remains favorable in every readout.

For transparency, the observed sign pattern by separately labeled cohort is:

| Cohort | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |
|:---|:---:|:---:|:---:|:---:|
| preregistered seed3–5 | 3/3 | 3/3 | 3/3 | 3/3 |
| secondary seed6–7 | 2/2 | 1/2 | 2/2 | 2/2 |
| secondary seed14–18 | 4/5 | 3/5 | 3/5 | 3/5 |
| all observed seeds, descriptive only | 9/10 | 7/10 | 8/10 | 8/10 |

The last row is a transparency count, not a pooled confirmatory `n=10` study.
The cohorts were authorized at different times and retain their original
evidence labels.

## Factorial mechanism boundary

The seed14–18 primary FID contrasts do not support a universal
target-geometry-dominant decomposition:

| NFE1 FID contrast | Mean | Favorable/negative seeds |
|:---|---:|:---:|
| C−A | -3.386725 | 4/5 |
| B−D | +7.529308 | 2/5 |
| D−A | -26.039608 | 4/5 |
| B−C | -15.123576 | 3/5 |
| I = B−C−D+A | +10.916032 | 1/5 |

The paired KID directions are similarly heterogeneous.  In particular, B−D
is not directionally stable, denominator contrasts are often favorable, and
the additive interaction is positive in four of five seeds for both FID and
KID at NFE1.  NFE2 shows the same four-of-five positive interaction pattern.

Accordingly:

- the preregistered seed3/4/5 target-dominant interpretation remains a valid
  description of that cohort;
- seed6/7 and seed14–18 show that isolated component effects and interaction
  signs are trajectory dependent;
- no causal percentage decomposition or universal mechanism claim follows
  from the combined observed seeds; and
- this extension must be disclosed anywhere the original mechanism result is
  summarized at paper level.

## Included evidence

- `q256_factorial_seed14_18_extension_results.csv` contains the 40
  full-precision job rows and artifact hashes.
- `q256_factorial_seed14_18_extension_results.json` contains machine-readable
  protocol, cell summaries, per-seed contrasts, and completion bindings.
- `q256_factorial_seed14_18_extension_results.md` contains the complete
  per-seed tables and seed14–18 descriptive statistics.
- `seed14_18_extension_frozen_evaluation/` contains the five evaluation plans,
  five `WORKER_PASS` records, 40 PASS receipts, 80 raw metric records, 40
  sampling-block diagnostics, and accepted-launch provenance.  Large retained
  samples and feature arrays remain on the experiment server and are bound by
  the included receipt hashes.

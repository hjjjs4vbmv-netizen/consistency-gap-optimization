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
small adverse B−A reversal at NFE2.  Seed17 is the clearest observed
counterexample to a universal full-intervention claim: B is worse than A on
both FID and KID at both NFE modes.  It is a valid observation and is retained
in every summary, not designated or removed as an outlier.

The new cohort therefore weakens both a universal target-dominance claim and
any wording that the coupled B intervention is uniformly robust or generally
the best arm.  B retains a favorable average and directional tendency versus A
in this cohort, but it does not dominate the isolated components.  For primary
NFE1 FID, B−A is favorable in 4/5 seeds and D−A is also favorable in 4/5;
the cohort mean is lower for D (299.4704) than for B (306.9997), while B−D is
favorable in only 2/5 seeds.

For transparency, the observed B−A sign pattern across all readouts is:

| Cohort | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |
|:---|:---:|:---:|:---:|:---:|
| preregistered seed3–5 | 3/3 | 3/3 | 3/3 | 3/3 |
| secondary seed6–7 | 2/2 | 1/2 | 2/2 | 2/2 |
| secondary seed14–18 | 4/5 | 3/5 | 3/5 | 3/5 |
| all observed seeds, descriptive only | 9/10 | 7/10 | 8/10 | 8/10 |

The last row is a transparency count, not a pooled confirmatory `n=10` study.
The cohorts were authorized at different times and retain their original
evidence labels.

The primary NFE1 FID factorial contrasts make the cross-cohort sensitivity
visible.  Entries are favorable/negative sign counts; lower FID is better.

| Evidence group | B−A | C−A | D−A | B−D |
|:---|:---:|:---:|:---:|:---:|
| original preregistered seeds3–5 | 3/3 | 3/3 | 2/3 | 3/3 |
| secondary seeds6–7 | 2/2 | 1/2 | 2/2 | 0/2 |
| secondary seeds14–18 | 4/5 | 4/5 | 4/5 | 2/5 |
| all observed seeds, descriptive only | 9/10 | 8/10 | 8/10 | 5/10 |

These direction changes show that exact objective-level factorization does not
yield a seed-stable endpoint factorization.

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

- target and denominator contributions remain exactly factorized at the
  objective definition level;
- seed6/7 and seed14–18 show that endpoint component effects and interaction
  signs are strongly trajectory dependent;
- B retains an average/directional tendency versus A in the observed
  sensitivity cohorts, but neither B nor the target component is uniformly
  dominant at the endpoint;
- no causal percentage decomposition or universal mechanism claim follows
  from the combined observed seeds; and
- this extension must be disclosed anywhere the original mechanism result is
  summarized at paper level.

## Paper-facing evidence boundary

Seeds14–18 are a post-preregistration secondary sensitivity extension.  They
must not be pooled with or relabeled as the held-out confirmation of the
revised B−A hypothesis.  [PR #72](https://github.com/hjjjs4vbmv-netizen/consistency-gap-optimization/pull/72)
seeds8–12 remain the prospective Cohort III validation cohort.  The prospective
question is still unresolved: does the favorable average/directional B−A NFE1
FID tendency replicate in that held-out cohort?

The manuscript interpretation should therefore center on (i) exact
objective-level target/denominator factorization, (ii) strongly
trajectory-dependent endpoint component effects and interaction signs, and
(iii) the unresolved prospective Cohort III test.  RAdam or optimizer-scale
theory is not a novelty claim of this result.

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

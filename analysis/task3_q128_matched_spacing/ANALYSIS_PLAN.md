# Task 3 analysis plan: q128 matched-spacing follow-up

Status: prospective analysis item to be fixed before any Task 3 outcome is
examined. This file does not alter the Task 3 primary endpoint or verdict.

## Prespecified secondary question

This question is motivated by the post-seal TASK 7 observation that, for q128
seed 3 at 1024 kimg and NFE2, the historical B0
`Cmatch-Bmatch` log-FID contrast was positive whereas the same contrast was
negative in generation blocks B1--B5.

For every prespecified new Task 3 training seed, estimate

`delta_s = log(FID50k(Cmatch_s@1024, NFE2)) - log(FID50k(Bmatch_s@1024, NFE2))`.

Use paired generation blocks for Cmatch and Bmatch. Report each seed's B0
contrast and its mean over the prespecified new blocks separately; do not pool
generation blocks as independent samples. Across training seeds, report every
block-mean `delta_s`, the mean, a two-sided 95% t interval, and the negative/zero/
positive sign count.

## Interpretation fixed before outcomes

- A negative mean is described only as directionally concordant with the TASK 7
  observation.
- A zero or positive mean is described as not directionally concordant.
- No outcome is called a Bmatch or Cmatch winner unless its own cell-level
  generation SD is shown; if the ordering changes across the prespecified
  blocks, label it `not interpreted`.
- This is not a primary endpoint, TIE or equivalence test, multiplicity rescue,
  or confirmatory mechanism result. It does not modify Task 3's primary verdict
  or any previously frozen inference.

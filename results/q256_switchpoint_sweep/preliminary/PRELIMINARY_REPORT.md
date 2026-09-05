# TASK 2 archive-recovery descriptive summary

Status: `EXPLORATORY_ONLY — incomplete archive recovery`.

Artifact status: `EXPLORATORY_ONLY` (`INCOMPLETE_ARCHIVE_RECOVERY`). Formal primary disposition: `PRIMARY_ABORTED_INSUFFICIENT_COMPLETE_SEEDS`. The archive preserved receipt-matched checkpoints for all eight fixed-chase evaluation cells in six seeds (81–86). We completed 48 recovery evaluation jobs, yielding 24 matched BA–CTRL contrasts. These evaluations were not part of a recovered 132-job opaque terminal seal. With six four-point-complete recovered seeds, the frozen minimum of nine was not met. The prespecified Page test was not run; neither `ORDERED_PREFIX_DEPENDENCE` nor `ORDERING_NOT_RESOLVED` is assigned.

## Fixed 512-kimg A-chase contrasts

Positive G means the BA trajectory has worse FID than its same-seed CTRL; negative G means better FID.

| switch (kimg) | n | mean G | median | sample SD | descriptive 95% mean t-interval | signs −/0/+ |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 6 | -0.13861 | -0.13916 | 0.25122 | [-0.40226, 0.12503] | 4/0/2 |
| 256 | 6 | -0.12664 | -0.12872 | 0.22566 | [-0.36345, 0.11017] | 4/0/2 |
| 384 | 6 | -0.12992 | -0.11385 | 0.21421 | [-0.35471, 0.09488] | 4/0/2 |
| 512 | 6 | -0.11562 | -0.07790 | 0.18293 | [-0.30760, 0.07636] | 5/0/1 |

## Adjacent paired differences

| contrast | n | mean | median | sample SD | descriptive 95% mean t-interval | signs −/0/+ |
|---|---:|---:|---:|---:|---:|---:|
| G_256_minus_G_128 | 6 | 0.01197 | 0.05282 | 0.08411 | [-0.07630, 0.10024] | 2/0/4 |
| G_384_minus_G_256 | 6 | -0.00327 | -0.01198 | 0.05986 | [-0.06609, 0.05955] | 3/0/3 |
| G_512_minus_G_384 | 6 | 0.01430 | 0.02127 | 0.05834 | [-0.04693, 0.07552] | 3/0/3 |

## Seed-level G values

| seed | G_128 | G_256 | G_384 | G_512 |
|---:|---:|---:|---:|---:|
| 81 | -0.15751 | -0.26430 | -0.20393 | -0.14342 |
| 82 | -0.12082 | -0.07626 | -0.11202 | -0.11807 |
| 83 | 0.17464 | 0.09156 | 0.10321 | 0.08368 |
| 84 | -0.24228 | -0.18118 | -0.11568 | -0.03772 |
| 85 | -0.54641 | -0.46451 | -0.50012 | -0.45154 |
| 86 | 0.06070 | 0.13485 | 0.04905 | -0.02664 |

## Interpretation boundary

Recovery depended on which checkpoints had completed and been archived before the archive interruption, not on a prespecified subset-selection rule. Missingness is not assumed random; means, intervals, sign counts, and adjacent differences describe the six recovered seeds only. These summaries do not establish ordered B-prefix dependence or distinguish an early-window account, accumulated exposure, persistence or decay, or any mechanism. They are confined to the recovered q256 CIFAR-10 trajectories. Missing seeds are not imputed or replaced, no reduced-arm test is performed, and these data are not pooled with prior cohorts.

## Provenance

Evaluator commit `d6aba02fb88e9db0993623895eb2228ed717d810`; FP32; NFE1; FID50k and KID50k from shared generated features; generation seeds 0–49999; metric seed 20260730.
These identities and all 48 checkpoint, option, artifact, and shared-feature checks are recorded in `verification.json`.

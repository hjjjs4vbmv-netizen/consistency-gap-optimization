# q256 factorial seed6/7 extension training report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_factorial_seed6_7_extension_report_v1

## Scope

Seeds 6 and 7 are a secondary precision extension. They are independent
training seeds outside the original seeds3/4/5 preregistration, do not
replace seeds3/4/5, and must not be described as preregistered replications.
Training used `--metrics=none`; all loss contrasts below are training-objective
diagnostics and do not support a generation-quality conclusion.

## Training endpoints

| Seed | Arm | Attempts | Accepted updates | AMP skips | Final-row loss | Last-200 mean ± SD | Semantic non-finite | Raw-grad/skip mismatch | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | A | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,38) | 17.22322369 | 16.37617182 ± 1.11711757 | 0 | 0 | 0 |
| 6 | B | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,15) | 17.09337294 | 16.28101440 ± 1.10459834 | 0 | 0 | 0 |
| 6 | C | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,15) | 18.92388940 | 17.93113834 ± 1.23866386 | 0 | 0 | 0 |
| 6 | D | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,15) | 15.64823806 | 14.85780734 ± 1.02474916 | 0 | 0 | 0 |
| 7 | A | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,1317) | 15.46983504 | 16.27821821 ± 1.01354271 | 0 | 0 | 0 |
| 7 | B | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,13,1287) | 15.29010594 | 16.06378486 ± 0.98567848 | 0 | 0 | 0 |
| 7 | C | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,1046) | 16.87773883 | 17.75639274 ± 1.09380566 | 0 | 0 | 0 |
| 7 | D | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,13,599) | 14.23506236 | 14.81675965 ± 0.93172171 | 0 | 0 | 0 |

## Integrity

- Both seeds completed all four A/B/C/D arms at exactly 2000 attempts and 256.000 kimg.
- Dataset, source checkpoint, source commit, optimizer, AMP, batch, LR, gap factors,
  checkpoint cadence, and `--metrics=none` identities passed the frozen-option audit.
- Every semantic non-finite counter is zero. Raw-gradient non-finites occurred only
  on recorded AMP-skipped attempts and never changed parameters.
- All realized denominators are positive; denominator non-finite/nonpositive counters are zero.
- Per-attempt identities passed for A/D same target, B/C same target, A/C same
  denominator, and B/D same denominator in both seeds.
- Final training states and snapshots are loadable, finite, mutually EMA-identical,
  and bind the same within-seed initial state across arms.

## Training-objective diagnostics

The values use last-200 mean loss and the requested order
`[C-A, B-D, D-A, B-C]`. Interaction is `B-C-D+A`.

| Seed | C-A | B-D | D-A | B-C | B-A | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1.55496652 | 1.42320706 | -1.51836448 | -1.65012394 | -0.09515742 | -0.13175946 |
| 7 | 1.47817454 | 1.24702522 | -1.46145856 | -1.69260788 | -0.21443334 | -0.23114932 |

- Target geometry stably raises the raw objective: **True**.
- Denominator scaling stably lowers the raw objective: **True**.
- B versus A remains approximately cancelling (descriptive): **True**.
- The factorial interaction remains small relative to the component contrasts (descriptive): **True**.

These statements describe only the training objective. Frozen FID/KID evaluation
is required before any quality interpretation.

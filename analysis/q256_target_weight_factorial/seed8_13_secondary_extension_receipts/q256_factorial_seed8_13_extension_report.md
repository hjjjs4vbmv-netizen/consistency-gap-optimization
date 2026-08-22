# q256 factorial seed8-13 extension training report

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_factorial_seed8_13_extension_report_v1

## Scope

Seeds 8 through 13 are a secondary precision extension. They are independent
training seeds outside the original seeds3/4/5 preregistration, do not
replace seeds3/4/5, and must not be described as preregistered replications.
Training used `--metrics=none`; all loss contrasts below are training-objective
diagnostics and do not support a generation-quality conclusion.

## Training endpoints

| Seed | Arm | Attempts | Accepted updates | AMP skips | Final-row loss | Last-200 mean ± SD | Semantic non-finite | Raw-grad/skip mismatch | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | A | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,10) | 15.25097823 | 16.37985846 ± 1.04673216 | 0 | 0 | 0 |
| 8 | B | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,11) | 15.33547056 | 16.24654768 ± 1.04164687 | 0 | 0 | 0 |
| 8 | C | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,10) | 16.67282414 | 17.96733661 ± 1.18097408 | 0 | 0 | 0 |
| 8 | D | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,1976) | 13.83735609 | 14.80070168 ± 0.93815818 | 0 | 0 | 0 |
| 9 | A | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,10,1557) | 16.74358785 | 16.37479959 ± 1.07299832 | 0 | 0 | 0 |
| 9 | B | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,10) | 16.57133639 | 16.30764402 ± 1.07008428 | 0 | 0 | 0 |
| 9 | C | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,10) | 18.40686893 | 17.90355740 ± 1.16013874 | 0 | 0 | 0 |
| 9 | D | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,10) | 15.22586429 | 14.92909934 ± 0.99067357 | 0 | 0 | 0 |
| 10 | A | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,14) | 15.13810670 | 16.40599109 ± 1.03408320 | 0 | 0 | 0 |
| 10 | B | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,14,391) | 15.32871175 | 16.22816132 ± 1.02928278 | 0 | 0 | 0 |
| 10 | C | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,14) | 16.64956319 | 17.94673627 ± 1.12253802 | 0 | 0 | 0 |
| 10 | D | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,23,391) | 13.93911231 | 14.82238706 ± 0.93590422 | 0 | 0 | 0 |
| 11 | A | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,14) | 15.23509467 | 16.32616225 ± 1.02161174 | 0 | 0 | 0 |
| 11 | B | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,888) | 14.84328318 | 16.15765199 ± 0.99223307 | 0 | 0 | 0 |
| 11 | C | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,13) | 16.22023129 | 17.90322223 ± 1.10326906 | 0 | 0 | 0 |
| 11 | D | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,11) | 13.47223234 | 14.86960258 ± 0.92497931 | 0 | 0 | 0 |
| 12 | A | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,15) | 17.17863107 | 16.46686129 ± 1.12107182 | 0 | 0 | 0 |
| 12 | B | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,11) | 16.84728515 | 16.23764294 ± 1.09149037 | 0 | 0 | 0 |
| 12 | C | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,15) | 18.60162652 | 17.99585883 ± 1.22426749 | 0 | 0 | 0 |
| 12 | D | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,11,692) | 15.30972040 | 14.85360233 ± 1.00821167 | 0 | 0 | 0 |
| 13 | A | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,12,1693) | 16.97200716 | 16.37218458 ± 1.00870145 | 0 | 0 | 0 |
| 13 | B | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,14) | 17.29902744 | 16.34008034 ± 1.00569927 | 0 | 0 | 0 |
| 13 | C | 2000 | 1991 | 9 (1,2,3,4,5,6,7,8,11) | 18.73427939 | 17.95139481 ± 1.10199079 | 0 | 0 | 0 |
| 13 | D | 2000 | 1990 | 10 (1,2,3,4,5,6,7,8,14,1693) | 15.88375235 | 14.92754814 ± 0.93361204 | 0 | 0 | 0 |

## Integrity

- All six seeds completed all four A/B/C/D arms at exactly 2000 attempts and 256.000 kimg.
- Dataset, source checkpoint, source commit, optimizer, AMP, batch, LR, gap factors,
  checkpoint cadence, and `--metrics=none` identities passed the frozen-option audit.
- Every semantic non-finite counter is zero. Raw-gradient non-finites occurred only
  on recorded AMP-skipped attempts and never changed parameters.
- All realized denominators are positive; denominator non-finite/nonpositive counters are zero.
- Per-attempt identities passed for A/D same target, B/C same target, A/C same
  denominator, and B/D same denominator in all six seeds.
- Final training states and snapshots are loadable, finite, mutually EMA-identical,
  and bind the same within-seed initial state across arms.

## Training-objective diagnostics

The values use last-200 mean loss and the requested order
`[C-A, B-D, D-A, B-C]`. Interaction is `B-C-D+A`.

| Seed | C-A | B-D | D-A | B-C | B-A | Interaction |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.58747816 | 1.44584600 | -1.57915678 | -1.72078894 | -0.13331078 | -0.14163216 |
| 9 | 1.52875780 | 1.37854468 | -1.44570026 | -1.59591338 | -0.06715558 | -0.15021312 |
| 10 | 1.54074518 | 1.40577425 | -1.58360403 | -1.71857496 | -0.17782977 | -0.13497093 |
| 11 | 1.57705999 | 1.28804941 | -1.45655966 | -1.74557024 | -0.16851026 | -0.28901058 |
| 12 | 1.52899754 | 1.38404061 | -1.61325896 | -1.75821589 | -0.22921835 | -0.14495693 |
| 13 | 1.57921023 | 1.41253220 | -1.44463644 | -1.61131447 | -0.03210424 | -0.16667803 |

- Target geometry stably raises the raw objective: **True**.
- Denominator scaling stably lowers the raw objective: **True**.
- B versus A remains approximately cancelling (descriptive): **True**.
- The factorial interaction remains small relative to the component contrasts (descriptive): **True**.

These statements describe only the training objective. Frozen FID/KID evaluation
is required before any quality interpretation.

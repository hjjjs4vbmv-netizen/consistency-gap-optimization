# seed13 extension integrity audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_seed13_extension_integrity_v1

This seed is a secondary precision extension. It is not part of the original
seeds3/4/5 preregistration and does not replace any preregistered seed.

Overall status: **PASS**.

## Cell summary

| Arm | Attempts | Accepted | AMP skips | Final loss | Last-200 mean ± SD | Semantic non-finite | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2000 | 1990 | 10 | 16.97200716 | 16.37218458 ± 1.00870145 | 0 | 0 |
| B | 2000 | 1991 | 9 | 17.29902744 | 16.34008034 ± 1.00569927 | 0 | 0 |
| C | 2000 | 1991 | 9 | 18.73427939 | 17.95139481 ± 1.10199079 | 0 | 0 |
| D | 2000 | 1990 | 10 | 15.88375235 | 14.92754814 ± 0.93361204 | 0 | 0 |

## Integrity conclusions

- Four-arm completion: `True`.
- Denominator integrity: `True`.
- Telemetry identity checks: `True`.
- Common initial-state identity: `True`.
- Raw-gradient non-finite rows correspond exactly to AMP-skipped attempts;
  they are reported separately from semantic non-finite counters.

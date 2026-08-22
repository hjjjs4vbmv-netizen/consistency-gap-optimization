# seed10 extension integrity audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_seed10_extension_integrity_v1

This seed is a secondary precision extension. It is not part of the original
seeds3/4/5 preregistration and does not replace any preregistered seed.

Overall status: **PASS**.

## Cell summary

| Arm | Attempts | Accepted | AMP skips | Final loss | Last-200 mean ± SD | Semantic non-finite | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2000 | 1990 | 10 | 15.13810670 | 16.40599109 ± 1.03408320 | 0 | 0 |
| B | 2000 | 1990 | 10 | 15.32871175 | 16.22816132 ± 1.02928278 | 0 | 0 |
| C | 2000 | 1990 | 10 | 16.64956319 | 17.94673627 ± 1.12253802 | 0 | 0 |
| D | 2000 | 1990 | 10 | 13.93911231 | 14.82238706 ± 0.93590422 | 0 | 0 |

## Integrity conclusions

- Four-arm completion: `True`.
- Denominator integrity: `True`.
- Telemetry identity checks: `True`.
- Common initial-state identity: `True`.
- Raw-gradient non-finite rows correspond exactly to AMP-skipped attempts;
  they are reported separately from semantic non-finite counters.

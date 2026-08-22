# seed11 extension integrity audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_seed11_extension_integrity_v1

This seed is a secondary precision extension. It is not part of the original
seeds3/4/5 preregistration and does not replace any preregistered seed.

Overall status: **PASS**.

## Cell summary

| Arm | Attempts | Accepted | AMP skips | Final loss | Last-200 mean ± SD | Semantic non-finite | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2000 | 1990 | 10 | 15.23509467 | 16.32616225 ± 1.02161174 | 0 | 0 |
| B | 2000 | 1990 | 10 | 14.84328318 | 16.15765199 ± 0.99223307 | 0 | 0 |
| C | 2000 | 1990 | 10 | 16.22023129 | 17.90322223 ± 1.10326906 | 0 | 0 |
| D | 2000 | 1991 | 9 | 13.47223234 | 14.86960258 ± 0.92497931 | 0 | 0 |

## Integrity conclusions

- Four-arm completion: `True`.
- Denominator integrity: `True`.
- Telemetry identity checks: `True`.
- Common initial-state identity: `True`.
- Raw-gradient non-finite rows correspond exactly to AMP-skipped attempts;
  they are reported separately from semantic non-finite counters.

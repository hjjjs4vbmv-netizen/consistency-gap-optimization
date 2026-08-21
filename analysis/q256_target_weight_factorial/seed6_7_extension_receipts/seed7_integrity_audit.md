# seed7 extension integrity audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_seed7_extension_integrity_v1

This seed is a secondary precision extension. It is not part of the original
seeds3/4/5 preregistration and does not replace any preregistered seed.

Overall status: **PASS**.

## Cell summary

| Arm | Attempts | Accepted | AMP skips | Final loss | Last-200 mean ± SD | Semantic non-finite | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2000 | 1990 | 10 | 15.46983504 | 16.27821821 ± 1.01354271 | 0 | 0 |
| B | 2000 | 1990 | 10 | 15.29010594 | 16.06378486 ± 0.98567848 | 0 | 0 |
| C | 2000 | 1990 | 10 | 16.87773883 | 17.75639274 ± 1.09380566 | 0 | 0 |
| D | 2000 | 1990 | 10 | 14.23506236 | 14.81675965 ± 0.93172171 | 0 | 0 |

## Integrity conclusions

- Four-arm completion: `True`.
- Denominator integrity: `True`.
- Telemetry identity checks: `True`.
- Common initial-state identity: `True`.
- Raw-gradient non-finite rows correspond exactly to AMP-skipped attempts;
  they are reported separately from semantic non-finite counters.

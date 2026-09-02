# seed6 extension integrity audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-20
- Verification Status: VERIFIED
- Version Label: q256_seed6_extension_integrity_v1

This seed is a secondary precision extension. It is not part of the original
seeds3/4/5 preregistration and does not replace any preregistered seed.

Overall status: **PASS**.

## Cell summary

| Arm | Attempts | Accepted | AMP skips | Final loss | Last-200 mean ± SD | Semantic non-finite | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2000 | 1990 | 10 | 17.22322369 | 16.37617182 ± 1.11711757 | 0 | 0 |
| B | 2000 | 1990 | 10 | 17.09337294 | 16.28101440 ± 1.10459834 | 0 | 0 |
| C | 2000 | 1990 | 10 | 18.92388940 | 17.93113834 ± 1.23866386 | 0 | 0 |
| D | 2000 | 1990 | 10 | 15.64823806 | 14.85780734 ± 1.02474916 | 0 | 0 |

## Integrity conclusions

- Four-arm completion: `True`.
- Denominator integrity: `True`.
- Telemetry identity checks: `True`.
- Common initial-state identity: `True`.
- Raw-gradient non-finite rows correspond exactly to AMP-skipped attempts;
  they are reported separately from semantic non-finite counters.

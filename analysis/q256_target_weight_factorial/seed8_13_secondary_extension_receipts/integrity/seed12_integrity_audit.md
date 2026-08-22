# seed12 extension integrity audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_seed12_extension_integrity_v1

This seed is a secondary precision extension. It is not part of the original
seeds3/4/5 preregistration and does not replace any preregistered seed.

Overall status: **PASS**.

## Cell summary

| Arm | Attempts | Accepted | AMP skips | Final loss | Last-200 mean ± SD | Semantic non-finite | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2000 | 1990 | 10 | 17.17863107 | 16.46686129 ± 1.12107182 | 0 | 0 |
| B | 2000 | 1991 | 9 | 16.84728515 | 16.23764294 ± 1.09149037 | 0 | 0 |
| C | 2000 | 1990 | 10 | 18.60162652 | 17.99585883 ± 1.22426749 | 0 | 0 |
| D | 2000 | 1990 | 10 | 15.30972040 | 14.85360233 ± 1.00821167 | 0 | 0 |

## Integrity conclusions

- Four-arm completion: `True`.
- Denominator integrity: `True`.
- Telemetry identity checks: `True`.
- Common initial-state identity: `True`.
- Raw-gradient non-finite rows correspond exactly to AMP-skipped attempts;
  they are reported separately from semantic non-finite counters.

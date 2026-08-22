# seed9 extension integrity audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_seed9_extension_integrity_v1

This seed is a secondary precision extension. It is not part of the original
seeds3/4/5 preregistration and does not replace any preregistered seed.

Overall status: **PASS**.

## Cell summary

| Arm | Attempts | Accepted | AMP skips | Final loss | Last-200 mean ± SD | Semantic non-finite | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2000 | 1990 | 10 | 16.74358785 | 16.37479959 ± 1.07299832 | 0 | 0 |
| B | 2000 | 1991 | 9 | 16.57133639 | 16.30764402 ± 1.07008428 | 0 | 0 |
| C | 2000 | 1991 | 9 | 18.40686893 | 17.90355740 ± 1.16013874 | 0 | 0 |
| D | 2000 | 1991 | 9 | 15.22586429 | 14.92909934 ± 0.99067357 | 0 | 0 |

## Integrity conclusions

- Four-arm completion: `True`.
- Denominator integrity: `True`.
- Telemetry identity checks: `True`.
- Common initial-state identity: `True`.
- Raw-gradient non-finite rows correspond exactly to AMP-skipped attempts;
  they are reported separately from semantic non-finite counters.

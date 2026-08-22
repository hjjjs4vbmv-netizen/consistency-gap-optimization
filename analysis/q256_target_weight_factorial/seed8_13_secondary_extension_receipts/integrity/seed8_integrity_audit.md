# seed8 extension integrity audit

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-21
- Verification Status: VERIFIED
- Version Label: q256_seed8_extension_integrity_v1

This seed is a secondary precision extension. It is not part of the original
seeds3/4/5 preregistration and does not replace any preregistered seed.

Overall status: **PASS**.

## Cell summary

| Arm | Attempts | Accepted | AMP skips | Final loss | Last-200 mean ± SD | Semantic non-finite | Nonpositive denominator |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 2000 | 1991 | 9 | 15.25097823 | 16.37985846 ± 1.04673216 | 0 | 0 |
| B | 2000 | 1991 | 9 | 15.33547056 | 16.24654768 ± 1.04164687 | 0 | 0 |
| C | 2000 | 1991 | 9 | 16.67282414 | 17.96733661 ± 1.18097408 | 0 | 0 |
| D | 2000 | 1990 | 10 | 13.83735609 | 14.80070168 ± 0.93815818 | 0 | 0 |

## Integrity conclusions

- Four-arm completion: `True`.
- Denominator integrity: `True`.
- Telemetry identity checks: `True`.
- Common initial-state identity: `True`.
- Raw-gradient non-finite rows correspond exactly to AMP-skipped attempts;
  they are reported separately from semantic non-finite counters.

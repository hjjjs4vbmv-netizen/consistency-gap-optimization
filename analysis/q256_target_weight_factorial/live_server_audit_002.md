# Live server audit 002

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: ANALYZED
- Version Label: q256_target_weight_factorial_live_server_audit_002

At `2026-08-19T07:52:09Z`, both A100 devices on `gpu0003` were idle: each
reported 4 MiB used, 0% utilization, 34 C, and the NVIDIA compute-process
query returned no rows. The prior Ray and independent gradient-audit process
blockers recorded in audit 001 are therefore resolved. No process or on-disk
artifact was deleted by this audit.

Durable storage also passes the capacity precheck. `/data/raw/ECT/ect_runs`
had 35,989,850,161,152 bytes available and `/data/temp` had
5,475,417,194,496 bytes available, both well above the frozen 60 GiB formal
minimum. Asset and runtime identities remain those verified in audit 001.

This receipt authorizes correctness-gate execution only. Formal training
remains unauthorized until a clean committed isolated source is installed and
the Role-E A/B parity, four-arm 32-attempt smoke, and exact-resume gates each
produce immutable PASS evidence. The dirty shared checkout
`/data/raw/ECT/recurrence_of_ect` remains prohibited.

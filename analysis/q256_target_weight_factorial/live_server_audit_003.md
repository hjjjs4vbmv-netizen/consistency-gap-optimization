# Live server audit 003

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: ANALYZED
- Version Label: q256_target_weight_factorial_live_server_audit_003

At `2026-08-19T08:30:45Z`, a bounded SSH health check to the previously
working internal endpoint exited 255 with `Connection to 172.16.30.17 port 22
timed out`. A pre-freeze source transfer to the new temporary directory
`/data/temp/ECT001/q256-factorial-prefreeze-20260819T1620` had produced no
progress output before the health failure was confirmed.

The incomplete temporary directory is not a formal artifact and must not be
used unless its full source identity is re-established. No correctness test,
smoke training, formal training, or evaluation was started during this event.
The last GPU observation remains audit 002; it is not carried forward as a
current claim.

All remote actions are stopped until SSH succeeds again. Recovery requires a
fresh compute-process/GPU audit, a complete isolated source installation, and
the normal clean-source and hash gates. The outage does not relax any Role-E,
smoke, resume, or formal authorization requirement.

# Live server audit 004

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: ANALYZED
- Version Label: q256_target_weight_factorial_live_server_audit_004

SSH recovered after the stalled pre-freeze transfer exited successfully. A
fresh bounded health check returned zero. Both A100 80GB PCIe devices again
reported 4 MiB used, 0% utilization, 34 C, and the compute-process query was
empty.

The mixed-time copy from audit 003 remains ineligible. A second temporary
source was built from the independently verified clean canonical base
`c01a61d767793ac52b427e4064f1f71583a17e1c` and overlaid with the current
branch differences at
`/data/temp/ECT001/q256-factorial-prefreeze-v2-20260819T1635`. A checksum-mode
rsync dry run reported no differing overlay file. This is a pre-freeze test
source only, not the eventual clean committed formal source.

The host is therefore cleared only for CUDA/AMP pre-freeze tests. Formal
authorization still requires a frozen clean commit, an immutable Role-E A/B
parity receipt, four-arm smoke, and exact-resume PASS evidence.

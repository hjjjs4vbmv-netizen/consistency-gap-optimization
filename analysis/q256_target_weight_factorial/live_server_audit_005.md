# Live server audit 005: Apptainer data mounts

At 2026-08-19T08:47:44Z the first committed Role E invocation stopped
before tests with exit code 1: the sandbox could not create its output under
`/data/raw`. No training started and no partial gate receipt or evidence file
was created.

The host exposes `/data/raw` and `/data/temp` as writable NFS mounts, with
approximately 33 TB and 5 TB available respectively. The unmodified
Apptainer invocation exposed neither path as a writable in-sandbox mount.
This is an operational P0 because the same entry path is used by training.

The corrective source change explicitly binds both paths in the shell and
Python launcher entry paths, records those bind specifications in runtime
identity, and binds them into exact-resume validation. All tests and Role E
evidence must be regenerated from the new commit. The failed attempt has no
reusable PASS evidence.

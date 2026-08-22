# Live server audit 006: nested launcher Python identity

The `smoke-uninterrupted-cfc667c` matrix passed its matrix-level preflight but
stopped before the first arm created a run directory. The outer process and
Role E receipt used `/usr/bin/python`; the inherited arm shell selected
`/usr/bin/python3`. Those paths refer to content-identical binaries, but the
frozen runtime identity deliberately includes paths and therefore rejected
the mismatch.

The matrix completion is `STOPPED_FOR_AUDIT`: no cell completed, A stopped
before `ct_train.py`, B/C/D were not started, and no GPU compute process
remained. The two matrix receipts are retained as negative operational
evidence.

The corrective source change makes `python` the sole permitted in-sandbox
launcher interpreter when `ECT_Q256_LAUNCHER_IN_SANDBOX=1`; `python3` remains
only the old-host bootstrap. A regression test ensures that an inherited
sandbox launch cannot fall back to `python3`. All Role E and authorization
evidence must be regenerated for the new commit.

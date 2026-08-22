# Live server audit 007: gate test environment isolation

The Role E gate for commit `9473dca` finished with 126 passing tests, one
failure, zero errors, and zero skips. All three required A/B parity test
identities passed; the sole failure was the launcher test that simulates an
old host Python bootstrap.

The gate process correctly sets `ECT_Q256_LAUNCHER_IN_SANDBOX=1`. The test
copied that environment without clearing the flag, so it exercised the real
inside-sandbox path rather than its mocked old-host path. Local execution had
passed only because the local process did not set the flag.

The correction explicitly removes the sandbox flag from the simulated host
environment. The FAIL receipt is retained as negative evidence. No training
started, the GPU was released, and the full gate must be regenerated from a
new commit.

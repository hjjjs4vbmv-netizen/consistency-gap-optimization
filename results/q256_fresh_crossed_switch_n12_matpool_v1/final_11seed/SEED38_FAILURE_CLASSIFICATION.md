# Seed38/AB terminal failure classification

Status: **TERMINAL NUMERICAL FAILURE; NOT A HARD TIMEOUT**

This classification is based on the unmodified files in this directory:

- `seed38_AB_terminal_training_telemetry.csv`;
- `seed38_AB_terminal_launcher.log`;
- `seed38_AB_terminal_monitor.jsonl`;
- `seed38_AB_terminal_compute_receipt.json`;
- `numeric_recovery2_authorization.json`;
- `eleven_seed_authorization.json`.

## Observed sequence

1. At attempted iteration 4857, telemetry records a non-finite loss, 55,732,739
   raw-gradient non-finite elements, zero sanitized-gradient non-finite elements,
   zero non-finite update/model/EMA/factor values, a skipped optimizer step, and
   GradScaler 64 -> 32. The launcher records this as the single protocol-allowed
   AMP-managed overflow.
2. Attempts 4858--4865 are present and finite. Training therefore continued for
   eight further attempts after the managed overflow; this was not an immediate
   abort at attempt 4857.
3. At attempted iteration 4866, telemetry records the second non-finite loss,
   again with a skipped step and GradScaler 32 -> 16. Because the authorization
   allowed at most one recoverable non-finite-loss attempt per cell, the strict
   invariant raised `FloatingPointError` and terminated the process.
4. The terminal compute receipt records `status=FAIL`, `exit_code=1`,
   `hard_timeout=false`, elapsed time 1575.710257 seconds, and label `seed38:AB`.
   The final monitor row records the process dead and the traceback marker.

The terminal receipt SHA256 is
`5f6283128f138ab716febf45122d373be8834e78a988e0eb762fee5b600dbbfb`,
which equals `terminal_failed_compute_receipt_sha256` in
`eleven_seed_authorization.json`.

## Missingness implication

The evidence supports a protocol-enforced numerical failure classification. It
does not support calling the missing seed ordinary infrastructure loss: failure
occurred in a specific AB treatment trajectory after a reproducible numerical
pattern. With one affected seed, treatment dependence is not proven, but it
cannot be ruled out. The n=11 complete-case analysis must therefore retain
informative missingness as a limitation, and the abandoned n=12 claim must not
be restored post hoc.

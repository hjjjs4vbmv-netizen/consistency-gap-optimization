# Arm A same-trajectory longitudinal RAdam audit

This directory is the canonical four-state longitudinal bundle from the
formal Arm A trajectory `arm_a_g1_0_lr_fixed_s3`.  It contains exactly the
restored states at K = 32.128, 64.128, 128.128, and 256.000 kimg.  At each
state, the audit clones the same `(theta, m, v, step, GradScaler)` and uses
the same minibatch, `t`, noise, and dropout RNG state for the two disposable
current-gap branches (`g=1.0` and `g=1.3`).  Neither branch commits its
virtual update to the source state.

`longitudinal_summary.csv` and `same_trajectory_residuals.png` are the
cross-state summary.  The per-state JSON receipts contain all scalar and
support/off-support diagnostics; the matching layerwise CSV files have 208
rows each.  `artifact_sha256.json` records SHA256 digests of the imported
payload files.

The JSON provenance path fields are canonicalized to basenames while their
recorded source SHA256 values remain unchanged.  The audit code source commit
is `6564379bb421022b3130463057a6d5c8f15a50fd` and accepts the legitimate
`global_sigmoid` g=1.0 Arm A reference checkpoint.

This is same-trajectory longitudinal evidence only.  It must not be combined
or presented as a common trajectory with any independently trained K≈1000
state, g-screen result, Arm B, or Arm C result.

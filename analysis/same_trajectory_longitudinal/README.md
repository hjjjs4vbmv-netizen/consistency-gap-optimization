# Role D: Arm A same-trajectory longitudinal measurement

## Scope

Role D measures one frozen, formal Arm A trajectory only:
`arm_a_g1_0_lr_fixed_s3`, at K = 32.128, 64.128, 128.128, and 256.000 kimg.
It does not start training, modify a source training state, or introduce any
other arm, g-screen run, or independently trained K≈1000 state.

At every K, the audit restores the complete state
`z_K = (theta_K, m_K, v_K, n_K, GradScaler_K)`, then creates two disposable
branches.  They receive the identical minibatch, `t`, noise, and dropout RNG
state; their only difference is the current gap, `g=1.0` versus `g=1.3`.
Neither virtual update is committed to the source state.

## Required measurement deliverables

For every K, the JSON receipt reports `a_K_star`, `R_grad(K)`, `s_K_star`,
`c_K_star`, `R_opt(K)`, `H_K`, and the full support/off-support diagnostics.
The matching layerwise CSV has 208 rows.  Every receipt must also establish
the pairing and provenance gates:

- same minibatch, `t`, noise, and dropout RNG state;
- complete nonzero RAdam and GradScaler state;
- `source_preserved=True`;
- neither virtual branch skipped its optimizer step.

`same_trajectory_residuals.png` is the primary figure: it plots
`K -> R_opt(K)` with `K -> R_grad(K)` overlaid.  `longitudinal_summary.csv`
is the machine-readable cross-state table.  The four per-state JSON/CSV pairs
are the underlying audit receipts, and `artifact_sha256.json` records SHA256
digests for the imported measurement artifacts.

## Provenance and interpretation boundary

The JSON provenance path fields are canonicalized to basenames while their
recorded source SHA256 values remain unchanged.  The executed audit-script
SHA256 is `601514bdab9a0a883f4e36ae3a4ad5a114eef892605aeb2b4240b34a6f086cc0`
from source commit `6564379bb421022b3130463057a6d5c8f15a50fd`; it accepts the
legitimate `global_sigmoid` g=1.0 Arm A reference checkpoint.

Role D reports these paired measurements and their provenance only.  It does
not attribute the residuals to a mechanism, infer whether they persist or
grow through training, or offer an optimizer-history explanation.  Such
interpretation is outside this delivery.

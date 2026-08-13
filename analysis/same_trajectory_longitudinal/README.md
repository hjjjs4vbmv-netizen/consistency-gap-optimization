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

`same_trajectory_residuals.pdf` and `same_trajectory_residuals.svg` are the
publication-ready vector masters; the 600-dpi PNG is a preview.  Panel (a)
plots `K -> {R_grad(K), R_opt(K)}` and labels the four measured `R_opt` values:
8.22%, 8.96%, 9.38%, and 9.90%.  Panel (b) plots `K -> c_K_star`, whose four
values span 1.029864--1.034610 (approximately 1.03 at every measured state).
The residual axis starts at zero, and the encodings remain distinguishable by
color, marker, and line style.  Connecting lines are guides to the eye, not
estimates between states.

`longitudinal_summary.csv` is the only input to the figure renderer.  The four
per-state JSON/CSV pairs remain the underlying audit receipts, and
`artifact_sha256.json` records SHA256 digests for the checked-in bundle
artifacts.  Regenerate the figure without rerunning or recalculating evidence:

```bash
python scripts/plot_same_trajectory_longitudinal.py
```

Suggested caption: **State-conditioned residuals along one frozen Arm A
trajectory.** At each restored nonzero RAdam state K, two non-committing
virtual updates share `z_K`, minibatch, `t`, noise, and dropout RNG and differ
only in the current gap (`g=1.0` versus `g=1.3`). (a) Reference-normalized raw
gradient and optimizer-update residuals; labels report `R_opt`. (b) Candidate
learning-rate rematch `c_K_star`, defined by `c_K_star U_1.3 ~= U_1`, remains
near 1.03 across the four measured states. Points are individual paired state
measurements; connecting lines guide the eye. The source states were preserved
and neither virtual step was skipped. The actual K values are 32.128, 64.128,
128.128, and 256.000 kimg. `H_K = R_opt` is an algebraic identity check, not
separate evidence.

## Provenance and interpretation boundary

The JSON provenance path fields are canonicalized to basenames while their
recorded source SHA256 values remain unchanged.  The receipts record source
commit `6564379bb421022b3130463057a6d5c8f15a50fd`.  Independently of that Git
object, the executed audit-script SHA256 is
`601514bdab9a0a883f4e36ae3a4ad5a114eef892605aeb2b4240b34a6f086cc0` and
matches the checked-in `analysis/radam_stateful_update_audit.py`; that script
accepts the legitimate `global_sigmoid` g=1.0 Arm A reference checkpoint.
The formal handoff receipt and read-only four-state runner are versioned in
the related delivery merge `c1d2d8a` as
`results/gap_lr_matched/role_d_formal_arm_a_handoff_receipt.json` and
`scripts/run_gap_lr_longitudinal_audit.sh`.  Their state IDs, artifact hashes,
optimizer steps, and GradScaler records agree with the four receipts here.

Role D reports these paired measurements and their provenance only.  "Near
1.03 at all four measured states" is descriptive, not a statistical stability
claim.  This bundle does not attribute the residuals to a mechanism, infer a
between-state trend, or offer an optimizer-history explanation.  Such
interpretation is outside this delivery.  It does not mix independent K≈1000
states, old g-screen results, Arm B, or Arm C into this trajectory.

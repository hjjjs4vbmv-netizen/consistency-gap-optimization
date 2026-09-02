# PR #89 full audit artifact archive

The expanded per-step telemetry and complete Jacobian receipt matrix are stored
outside the Git tree. The repository retains the frozen protocols, executable
code, tests, compact summaries, cell-level CSV, reports, formal manifests,
correctness receipts, and one exemplar for each Jacobian regime.

## Retrieval

- Archive: [`pr89-full-audit-artifacts-v2.zip`](https://github.com/hjjjs4vbmv-netizen/consistency-gap-optimization/releases/download/pr89-audit-artifacts-v2/pr89-full-audit-artifacts-v2.zip)
- Release tag: `pr89-audit-artifacts-v2`
- Compressed bytes: `953604`
- Uncompressed files: `166`
- Uncompressed bytes: `26711088`
- SHA256: `22dee9eba6caacf37bf5baeba35cc723df4b9883668d14f9dcd5c4c9beffad63`

## Contents

- `analysis/nonlinear_dynamics_gate/forcing_feedback_summary_full.json`:
  complete 64-step replay telemetry and state-transition receipts;
- `analysis/jacobian_failure_factorial_v2/results/raw_receipts/`:
  all 160 formal cells, correctness receipts, and both formal manifests.

The compact in-repository forcing-feedback summary records the SHA256 of the
omitted step-replay receipt sequence. The v2 Jacobian `results.csv` and
`summary.json` retain every cell's status and reported convergence statistics.

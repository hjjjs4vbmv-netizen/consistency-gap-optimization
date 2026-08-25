# q128 matched-spacing training integrity

## Completion and exposure

- Formal trajectories: **15/15 PASS**.
- Immutable state/snapshot milestones: **105/105 PASS**.
- Attempted iterations: **120,000** (`8,000` per trajectory).
- Accepted optimizer steps: **119,825**.
- AMP-skipped attempts: **175**.
- Every `train_summary.csv` contains the continuous attempted-iteration sequence
  `1..8000`; no resume/restart marker appears in any formal log.
- All 15 formal trajectories report only curriculum stage `0`.

`training_integrity.csv` gives attempted/accepted/skipped counts per cell.
`training_artifact_hashes.csv` gives every training-state, EMA snapshot, and
canonical EMA hash. The auditor recomputed SHA256 over all 105 state files and
all 105 snapshot files and matched every immutable export receipt.

## Within-seed pairing

For each of seeds 3, 4, and 5, all five arms have identical hashes for:

- transferred model and EMA initialization;
- optimizer and GradScaler state;
- rank RNG state;
- infinite-sampler state and minibatch order;
- normalized trajectory configuration after removing only the arm-specific
  target/denominator factor fields.

The shared RNG state plus identical deterministic call order binds the `t`,
noise, and dropout streams within seed. Optimizer settings, EMA, dataset,
runtime, total attempted iterations, and immutable budgets are also identical.
The only intended arm difference is the frozen target/denominator gap factors.

All three `A` cells and all three `Bsame` cells are fresh
`q128_matched_spacing_v1` launches with `attempted_iteration=0`; no historical
q128 output is reused as a formal five-arm cell.

## Hardware and runtime amendments

All immutable preflight receipts report `NVIDIA A100-PCIE-40GB, 40960 MiB`.
All cells use the same runtime SIF, dataset, and transfer checkpoint SHA256.
`hardware_assignment.csv` records node hostname and arm-to-GPU index, including
the post-start multi-GPU scheduling amendment.

The version-1 preflight schema did **not** record GPU UUID. The released nodes
cannot be queried retrospectively, so UUID values are reported as
`not_recorded_in_preflight_v1`, not reconstructed. The launcher now records
`index,name,memory.total,uuid` in future preflight receipts.

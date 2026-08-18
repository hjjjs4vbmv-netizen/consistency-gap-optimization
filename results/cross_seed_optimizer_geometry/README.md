# Cross-seed optimizer geometry operation

This directory isolates the operational contract for the 256k same-state
paired audit across Arm A seeds 3, 4, and 5.  It deliberately does **not**
store checkpoints, training states, raw receipts, or logs in Git.

Proposed PR title: `analysis: cross-seed replication of gap-induced optimizer divergence`.

The operation has two layers for each row:

- **Layer A — one-step virtual fork.** Restore the complete nonzero state
  `(theta, m, v, step, GradScaler)` and compare `g=1.0` with `g=1.3` on the
  identical minibatch, `t`, noise, and dropout RNG state. It reports `a*`,
  `R_grad`, `R_opt`, `c*`, `s*`, and support-aware `h_i` diagnostics.
- **Layer B — canonical 20-step scalar-history replay.** Run the #47/#58
  paired prospective fork from the exact Arm-A state, persist the raw paired
  gradient/update histories, then evaluate the scalar-history predictor at
  step 19 at the frozen RAdam learning rate `1e-4`. It reports weighted R²,
  Corr, and weighted RMSE.

Seed 3 is an existing, hash-bound K=256 reference artifact. Seeds 4 and 5 are
new measurements on two independent training trajectories; a repeated batch
inside one trajectory is never described as a training-seed replication.

## Server-side operation root

Create a fresh, timestamped directory outside the repository, for example:

```text
/data/raw/ECT/ect_runs/cross_seed_optimizer_geometry_YYYYMMDD/
├── bound_matrix.json
├── audit_manifest.json                 # created by the runner
├── seed3/
│   └── radam_update_audit_stateful.json
├── seed4/
│   └── radam_update_audit_stateful.json
├── seed5/
│   └── radam_update_audit_stateful.json
└── summary/
    ├── optimizer_geometry_table.csv
    └── OPTIMIZER_GEOMETRY_TABLE.md
```

Copy [`configs/cross_seed_optimizer_geometry_matrix.example.json`](../../configs/cross_seed_optimizer_geometry_matrix.example.json)
to `bound_matrix.json`, then replace every server path and placeholder SHA-256
value.  Bind each seed to its exact Arm A 256k training state and checkpoint;
do not use a `latest` alias.  The audit runner verifies those hashes before it
creates any output, and refuses an existing operation root.

For seed 4/5, first materialize a stable archived or numbered copy of the
state and checkpoint (the files named `*-latest.*` in a handoff are not an
acceptable runtime binding). Hash that exact copy and place both its path and
digest in the matrix. The known seed-4/5 checkpoint digests are already frozen
in the example; their training-state digests must be measured from the actual
received files. Do not infer a training-state hash from a checkpoint hash.

The matrix also records seed 3's historical `schedule_q=128` and seed 4/5's
confirmatory `schedule_q=256`. The final summary preserves that fact. If the
three schedule values remain different, the three rows must not be pooled as a
pure same-configuration seed estimate: seed 4/5 are independent q256
replications, while seed 3 remains a provenance-bound mechanism anchor.

## Invocation

From the repository checkout on the server, use the project virtualenv:

```bash
OP_ROOT=/data/raw/ECT/ect_runs/cross_seed_optimizer_geometry_YYYYMMDD
.venv/bin/python scripts/run_cross_seed_optimizer_geometry_audit.py \
  --manifest "$OP_ROOT/bound_matrix.json" \
  --out "$OP_ROOT"
.venv/bin/python scripts/summarize_cross_seed_optimizer_geometry.py \
  --audit-root "$OP_ROOT" \
  --out "$OP_ROOT/summary"
```

Run the three seed cells sequentially on one GPU.  This operation is a
same-state diagnostic at 256k only; it must not rerun the already completed
seed-3 32/64/128/256 trajectory audit.

The runner makes a write-once `audit_manifest.json` only after all required
hash/provenance and no-skip checks pass. If any command fails, leave the
partial root untouched for forensics and use a new timestamped root for a
retry; never mix raw histories from separate attempts. The summary table is
descriptive evidence about local optimizer geometry, not a causal proof about
endpoint-quality changes.

# Cross-seed optimizer geometry supporting analysis

This directory isolates the operational contract for the 256k same-state
paired audit across Arm A seeds 3, 4, and 5. It does **not** store
checkpoints, training states, raw tensor arrays, or logs in Git. It does store
the small, hash-verified receipts needed to inspect the reported result.

This is supporting evidence for the q256 optimizer-geometry program; it is not
a same-configuration three-seed replication and must not merge ahead of the
canonical q256 audit mainline.

## Checked-in receipt bundle

[`receipts_20260819`](receipts_20260819) contains the server-produced audit
manifest, the three-seed summary, Layer A JSON/CSV receipts, Layer B scalar
predictor receipts, and the raw-history metadata/manifests for seed4 and
seed5. `SHA256SUMS` verifies every extracted file. The received zip archive
had SHA-256
`b6c047b58a954cf011fa1e0633fab025ddc9de9813863f8124490342cb4dd545`;
the archive itself and all large `.npy` arrays intentionally remain outside
Git. The raw-artifact manifests bind those external arrays by SHA-256 and
size.

The completed server audit reports seed4/seed5 as independent q256 training
trajectories and seed3 as the q128, hash-bound historical anchor. The
three-row result is therefore evidence of training-seed heterogeneity, not a
pooled same-configuration estimate.

The operation has two layers for each row:

- **Layer A — one-step virtual fork.** Restore the complete nonzero state
  `(theta, m, v, step, GradScaler)` and compare `g=1.0` with `g=1.3` on the
  identical minibatch, `t`, noise, and dropout RNG state. It reports `a*`,
  `R_grad`, `R_opt`, `c*`, `s*`, and support-aware `h_i` diagnostics.
- **Appendix Layer B — canonical 20-step scalar-history replay.** Run the #47/#58
  paired prospective fork from the exact Arm-A state, persist the raw paired
  gradient/update histories, then evaluate the scalar-history predictor at
  step 19 at the frozen RAdam learning rate `1e-4`. It reports weighted R²,
  Corr, and weighted RMSE as supporting evidence only. The seed4/seed5
  explanatory values are heterogeneous, so this layer must not support a
  headline predictor claim.

Seed 3 is an existing, hash-bound K=256 reference artifact. Seeds 4 and 5 are
new measurements on two independent training trajectories; a repeated batch
inside one trajectory is never described as a training-seed replication.

## Server-side operation root

Create a fresh, timestamped operation directory outside the repository, for
example:

```text
/data/raw/ECT/ect_runs/cross_seed_optimizer_geometry_YYYYMMDD/
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
to a separate runtime-only binding directory, then replace every server path
and placeholder SHA-256 value. Bind each seed to its exact Arm A 256k training
state and checkpoint; do not use a `latest` alias. The audit runner verifies
those hashes before it creates any output, and refuses an existing operation
root by default.

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

From the repository checkout on the server, use the pinned NVIDIA Apptainer
runtime rather than host Python or the project virtualenv:

```bash
ECT_ROOT=/data/raw/ECT
SIF=$ECT_ROOT/runtime/containers/nvidia-pytorch-24.04.sif
OP_ROOT=$ECT_ROOT/ect_runs/cross_seed_optimizer_geometry_YYYYMMDD
BIND_ROOT=$ECT_ROOT/ect_runs/cross_seed_optimizer_geometry_bind_YYYYMMDD
apptainer exec --nv --bind "$ECT_ROOT:$ECT_ROOT" "$SIF" bash -lc '
  cd /data/raw/ECT/recurrence_of_ect
  python scripts/run_cross_seed_optimizer_geometry_audit.py \
    --manifest /data/raw/ECT/ect_runs/cross_seed_optimizer_geometry_bind_YYYYMMDD/bound_matrix.json \
    --out /data/raw/ECT/ect_runs/cross_seed_optimizer_geometry_YYYYMMDD
'
```

Run the three seed cells sequentially on one GPU.  This operation is a
same-state diagnostic at 256k only; it must not rerun the already completed
seed-3 32/64/128/256 trajectory audit.

The runner makes a write-once `audit_manifest.json` only after all required
hash/provenance and no-skip checks pass. If any command fails, leave the
partial root untouched for forensics. The default retry uses a new root; the
explicit `--resume-partial` mode is permitted only after completed components
and raw artifacts have been revalidated against their bound hashes. The
summary table is descriptive evidence about local optimizer geometry, not a
causal proof about endpoint-quality changes.

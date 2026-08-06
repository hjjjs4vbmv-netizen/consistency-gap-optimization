# Staged evaluation runbook

This runbook operationalizes `staged-checkpoint-evaluation-v1` from
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md). All GPU evaluation happens
on the remote server; the commands below are designed to be checked locally
with `--dry-run` first and then run unchanged on that server.

The prospective q=256 budget and fresh q=128 logical matrices are frozen in
[`FROZEN_EVALUATION_MATRICES.md`](FROZEN_EVALUATION_MATRICES.md). They must be
bound to completed checkpoints and integrity receipts before execution; quick
results may not select, remove, or substitute their predeclared formal cells.

## Local preparation

Copy and populate the checkpoint manifest. Every checkpoint must have an
immutable SHA256; do not reuse a result directory from a previous attempt.

```bash
cp configs/staged_evaluation_checkpoints.example.json \
  /mnt/ect_project/staged_eval/checkpoints.json
```

### Confirmatory q=256 matrix

The current six-cell fixed-sigmoid versus global-only `g=1.10` study is frozen
in [`../configs/staged_evaluation_confirmatory_q256.frozen.json`](../configs/staged_evaluation_confirmatory_q256.frozen.json).
It intentionally contains identities, hashes, schedule settings, source
commits, and receipt identities, but no server paths. Do not point
`run_staged_evaluation.py` directly at this logical manifest.

On the evaluation server, create a non-versioned runtime manifest which copies
each `checkpoint_id`, `method`, `training_seed`, `budget_kimg`, and
`checkpoint_sha256` unchanged, then adds the local `checkpoint` and
`integrity_receipt` paths. For the seed 4/5 receipt files, also recompute and
match the frozen receipt SHA256 before a formal launch. Seed 3 is pinned to the
tracked handoff integrity attestation (`D_HANDOFF.md`); its server-side
machine-readable receipt must be bound under the frozen filename before it can
pass the formal runner's receipt gate. Copy the top-level `comparison` and
`formal_promotion_policy` unchanged as well; the formal CLI rejects a manifest
without the frozen six-cell promotion policy.

Before any formal launch, validate the non-versioned manifest against the Git
matrix (without `--allow-missing-inputs`):

```bash
python scripts/validate_staged_runtime_manifest.py \
  --frozen configs/staged_evaluation_confirmatory_q256.frozen.json \
  --runtime /mnt/ect_project/staged_eval/checkpoints.json
```

The first remote job is the quick smoke for one named existing checkpoint. It
runs both NFE modes and both 5k metrics, producing four metric records total:

```bash
python scripts/run_staged_evaluation.py \
  --manifest /mnt/ect_project/staged_eval/checkpoints.json \
  --data /mnt/ect_project/datasets/cifar10-32x32.zip \
  --outdir /mnt/ect_project/staged_eval/smoke \
  --phase smoke \
  --smoke-checkpoint-id sigmoid_seed0_16k \
  --dry-run
```

Use the same command without `--dry-run` only on the remote server. The
runner rejects non-empty output roots, mismatched checkpoint hashes, and any
attempt to use missing input files outside a dry run.

## Remote execution order

Run these stages serially. Do not start a later stage after an incomplete or
failed predecessor.

1. Existing-checkpoint smoke: use the command above without `--dry-run`.
2. Fixed-seed image determinism: on the same server, run the acceptance sampler
   against the smoke checkpoint (or another named candidate) and then validate
   its emitted artifact. The sampler prints the exact checkpoint-isolated
   result directory; use that directory in the verifier command.

   ```bash
   bash scripts/sample_checkpoint.sh /mnt/ect_project/checkpoints/sigmoid_seed0_16k.pkl \
     --outdir /mnt/ect_project/staged_eval/fixed-seed \
     --seeds 0-63 --nfe 1 2 --mid-t 0.821 \
     --work-group-size 8 --verify-work-group-size 16 \
     --precision fp32 --device cuda

   python scripts/verify_fixed_seed_determinism.py \
     --result-dir /mnt/ect_project/staged_eval/fixed-seed/<checkpoint-stem>-<sha256-prefix>
   ```

   The verifier checks the exact 128-image set, both NFE/midpoint settings,
   repeated-run and 8/16 work-group assertions, and every manifest SHA256.
3. Collect the smoke table:

   ```bash
   python scripts/collect_staged_evaluation_results.py \
     --eval-root /mnt/ect_project/staged_eval/smoke \
     --outdir /mnt/ect_project/staged_eval/smoke-summary
   ```

4. Run the complete quick screening matrix (all manifest cells):

   ```bash
   python scripts/run_staged_evaluation.py \
     --manifest /mnt/ect_project/staged_eval/checkpoints.json \
     --data /mnt/ect_project/datasets/cifar10-32x32.zip \
     --outdir /mnt/ect_project/staged_eval/quick \
     --phase quick
   ```

5. Only after every candidate has a passed, SHA-matched training-integrity
   receipt, run the formal 50k matrix:

   ```bash
   python scripts/run_staged_evaluation.py \
     --frozen-matrix configs/staged_evaluation_confirmatory_q256.frozen.json \
     --manifest /mnt/ect_project/staged_eval/checkpoints.json \
     --data /mnt/ect_project/datasets/cifar10-32x32.zip \
     --outdir /mnt/ect_project/staged_eval/formal \
     --phase formal
   ```

The formal runner requires a receipt JSON for each checkpoint with
`status: "passed"` and a matching `checkpoint_sha256`. It refuses the formal
run before any metric process starts when this gate is not met. It also now
requires `--frozen-matrix` and validates the server manifest against that
matrix before it creates an evaluation record. As a server-only dry-run gate,
run:

```bash
bash scripts/preflight_formal_evaluation.sh \
  --frozen-matrix configs/staged_evaluation_confirmatory_q256.frozen.json \
  --runtime-manifest /mnt/ect_project/staged_eval/checkpoints.json \
  --data /mnt/ect_project/datasets/cifar10-32x32.zip \
  --outdir /mnt/ect_project/staged_eval/formal
```

For the frozen q=256 matrix, use all six predeclared cells in this command.
Quick 5k values are not a promotion gate: no seed, method, NFE, or checkpoint
may be removed from formal evaluation because of its quick metric performance.
The only eligibility decision is matching frozen provenance plus a passed,
SHA-matched training-integrity receipt.

## Outputs

The collector writes a long-form `evaluation_results.csv` and separate
`evaluation_statistics.json`/`.md`. The table carries the evidence class,
checkpoint and dataset identity, integrity status, NFE, both seed contracts,
metric value, evaluation revision, and run path. Statistics are grouped by
evidence class, metric, NFE, and method; quick and formal evidence are never
pooled. For the fixed/global-only confirmatory manifest it additionally writes
`paired_differences.csv`, `paired_statistics.json`, and
`paired_statistics.md`; the collector refuses missing or duplicated arms
instead of producing a partial paired summary. The paired outputs retain the
seed-level relative improvements and report arithmetic/geometric effects,
median deltas, rank consistency, worst-case effect, seed CV, exact sign-test
description, bootstrap sensitivity intervals, leave-one-seed-out summaries,
and the NFE=2-minus-NFE=1 effect-heterogeneity contrast. These remain
descriptive seed-level summaries rather than expanded independent-sample
inference.

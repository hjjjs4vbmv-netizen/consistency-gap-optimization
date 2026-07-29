# Staged evaluation runbook

This runbook operationalizes `staged-checkpoint-evaluation-v1` from
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md). All GPU evaluation happens
on the remote server; the commands below are designed to be checked locally
with `--dry-run` first and then run unchanged on that server.

## Local preparation

Copy and populate the checkpoint manifest. Every checkpoint must have an
immutable SHA256; do not reuse a result directory from a previous attempt.

```bash
cp configs/staged_evaluation_checkpoints.example.json \
  /mnt/ect_project/staged_eval/checkpoints.json
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
2. Fixed-seed image determinism: run `scripts/sample_checkpoint.sh` with seeds
   `0-63`, NFE `1 2`, `mid_t=0.821`, and work-group sizes `8`/`16` as specified
   in the protocol.
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
     --manifest /mnt/ect_project/staged_eval/checkpoints.json \
     --data /mnt/ect_project/datasets/cifar10-32x32.zip \
     --outdir /mnt/ect_project/staged_eval/formal \
     --phase formal
   ```

The formal runner requires a receipt JSON for each checkpoint with
`status: "passed"` and a matching `checkpoint_sha256`. It refuses the formal
run before any metric process starts when this gate is not met.

## Outputs

The collector writes a long-form `evaluation_results.csv` and separate
`evaluation_statistics.json`/`.md`. The table carries the evidence class,
checkpoint and dataset identity, integrity status, NFE, both seed contracts,
metric value, evaluation revision, and run path. Statistics are grouped by
evidence class, metric, NFE, and method; quick and formal evidence are never
pooled.

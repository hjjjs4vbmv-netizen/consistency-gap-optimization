# Server launch readiness

Use this checklist before starting a GPU job.  A passed local test suite is not
permission to run a blocked evaluation.

| Work item | Server status | Gate |
| --- | --- | --- |
| Historical q=128 1024-kimg metric import | **Do not run** | Retrospective/supplementary only; source run manifest, lineage, receipts, archives, and Role D verification are absent. |
| Fresh q=128 256-kimg formal evaluation | **Blocked** | The q128/q256 dataset semantic-equivalence and Role D receiver-verification blockers in `results/generalization/schedule-q128/fresh_256k_seed3_5/` must both be closed first. |
| q=256 formal evaluation | **Preflight required** | A complete server-local runtime manifest must match `configs/staged_evaluation_confirmatory_q256.frozen.json`, including all six receipt and checkpoint identities. |

## Required sequence for an eligible formal run

1. Use a committed, clean checkout of this revision.  Do not edit code or a
   frozen matrix on the server.
2. Copy the checkpoint archive and receipts through the agreed transfer
   channel; independently recompute the archive and checkpoint SHA-256 values.
3. Create the machine-local runtime manifest by adding only `checkpoint` and
   `integrity_receipt` paths to the matching frozen identities.
4. Run the non-executing gate below.  It verifies the frozen/runtime binding,
   checkpoint SHA-256 values, integrity receipts, formal inclusion policy,
   dataset existence, clean checkout, and empty output directory.

```bash
bash scripts/preflight_formal_evaluation.sh \
  --frozen-matrix configs/staged_evaluation_confirmatory_q256.frozen.json \
  --runtime-manifest /mnt/ect_project/staged_eval/checkpoints.json \
  --data /mnt/ect_project/datasets/cifar10-32x32.zip \
  --outdir /mnt/ect_project/staged_eval/formal
```

5. Only after that command prints `PASS`, launch the same immutable inputs
without `--dry-run`:

```bash
python scripts/run_staged_evaluation.py \
  --frozen-matrix configs/staged_evaluation_confirmatory_q256.frozen.json \
  --manifest /mnt/ect_project/staged_eval/checkpoints.json \
  --data /mnt/ect_project/datasets/cifar10-32x32.zip \
  --outdir /mnt/ect_project/staged_eval/formal \
  --phase formal
```

## q=128-specific prohibition

Do not pass the historical q=128 1024-kimg records to the formal runner.  They
are not a runtime manifest and are deliberately marked
`reported_unverified`.  For the fresh q=128 256-kimg handoff, the immutable
identity matrix is
`results/generalization/schedule-q128/fresh_256k_seed3_5/staged_evaluation_confirmatory_q128.frozen.json`;
however it remains blocked until both gates in the table above are closed.

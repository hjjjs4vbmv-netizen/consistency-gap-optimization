# Post-seed5 frozen evaluation plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-21
- Verification Status: UNVERIFIED
- Version Label: q256_post_seed5_frozen_evaluation_plan_v1

Status: **SCHEDULED, NOT STARTED**. Automation ID: `q256-formal-runs`.

## Trigger

Start automatically only after all seed3/4/5 × A/B/C/D training cells satisfy all of the following:

- exactly 2000 attempted iterations and 256.000 kimg;
- final checkpoint/state, telemetry, initial receipt, final image, and worker PASS are present;
- no real training Traceback, OOM, bus error, CUDA error, semantic non-finite event, or denominator failure remains;
- no `ct_train.py` process remains.

Before evaluation, record seed5/C and seed5/D counters, AMP skips, final artifact hashes, and the 12/12 completion milestone in `formal_run_record_dcca41b.json/.md`; commit, push, and sync the record.

## Frozen matrix scope

- Training source: `dcca41b19e7c45512b5fbe98776520396a1bf9ac`.
- Training root: `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/formal/formal-direct-dcca41b-deterministic-v1`.
- Bind all twelve final 256-kimg checkpoints in one matrix. No arm subset, intermediate checkpoint, or preview-based selection is permitted.
- The direct training queue did not create canonical `matrix_plan.json`, `matrix_completion.json`, or per-cell validation/hash receipts. Create only the minimal provenance/matrix binding required by the frozen evaluator from the completed immutable artifacts. Do not retrain a cell or change any checkpoint.
- If the full matrix cannot be bound reliably, stop before metric generation and notify the user.

## Evaluation order and semantics

Run on the same GPU that is currently carrying seed5: physical GPU index 0, UUID `GPU-d79117bb-8d91-4f2e-d7bb-718e347ce859`. Start only after seed5 exits and that GPU is idle with no foreign process. Do not move this evaluation to the currently idle physical GPU index 1 unless the user explicitly changes the instruction.

1. Complete NFE=1 for all 12 cells. Each frozen job computes FID-50k and KID-50k from the same 50,000 generated samples; FID-50k@NFE=1 is the primary endpoint.
2. Complete NFE=2 for all 12 cells with `mid_t=0.821`, computing FID-50k and KID-50k.
3. Run the frozen collector and report all seed-level raw values, the four preregistered contrasts, and the factorial interaction.

Frozen settings: FP32 sampling, sample seeds `0-49999`, metric seed `20260730`, final 256-kimg checkpoints, the same CIFAR-10 reference, and no metric repeats beyond the preregistered setting.

The current frozen runner enumerates jobs cell-major (`NFE=1`, then `NFE=2` per cell). The requested execution is primary-first across the whole matrix. If adaptation is necessary, it must be scheduling-only: it may reorder the already-frozen jobs but must not change checkpoint selection, sample seeds, sample count, sampler precision, NFE definitions, `mid_t`, metric implementation, reference statistics, or collector formulas. Record the adapter and its content hash.

## Paths

- Matrix binding: `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/evaluation/matrix-binding-direct-dcca41b-v1`
- Evaluation root: `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/evaluation/frozen-eval-dcca41b-primary-first-v1`
- Collected results: `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/evaluation/frozen-eval-dcca41b-primary-first-v1-results`

All three paths were absent when scheduled at 2026-08-21 00:25 CST. Evaluation must use exclusive creation and refuse overwrite.

## Monitoring and failure policy

- Run under tmux with explicit GPU binding and continue five-minute monitoring of PID, GPU, job count, logs, shared memory, and disk.
- Do not silently retry a failed metric job. Preserve its outputs and receipt, stop later jobs, record the failure, and notify the user.
- Do not touch any shared resource that cannot be confirmed as ECT001/q256-owned.
- After all 24 frozen jobs and collection pass, commit/push/sync the durable evaluation record and disable `q256-formal-runs`.

## Frozen result formulas

For each seed and endpoint:

- target geometry at baseline weighting: `Y_C - Y_A`;
- target geometry at g weighting: `Y_B - Y_D`;
- loss weighting at baseline target: `Y_D - Y_A`;
- loss weighting at g target: `Y_B - Y_C`;
- interaction: `I = Y_B - Y_C - Y_D + Y_A`.

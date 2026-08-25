# Second-q q128 A/B learning-curve protocol

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-23
- Verification Status: ANALYZED
- Version Label: second_q_q128_ab_v1

## Decision and current gate

The protocol is frozen, but training is **NO-GO until Role E returns a
conclusive dataset semantic-equivalence verdict**. `UNRESOLVED`, missing
evidence hashes, a dataset identity mismatch, a dirty source tree, a changed
training path, or a runtime mismatch all fail closed.

PR #31 records six completed fresh q128 cells on dataset ZIP SHA256
`9818e4b8...f1b3`; the q256 A/B learning curve records canonical training ZIP
SHA256 `08c9ed1b...4f372`. The old q128 checkpoints and PR #32 metrics are not
cross-q confirmation while semantic equivalence remains unresolved. They are
never mixed into the new frozen curve.

The selected execution path is a fresh canonical rerun even if Role E later
returns `SEMANTIC_EQUIVALENT`. This costs approximately the first 256 kimg of
six trajectories, but avoids a mixed-source trajectory and creates all frozen
curve checkpoints under the q256-pinned implementation. A different reuse path
would require a new protocol version and an independent full-state/source
compatibility proof before launch.

## Experiment overview

- Objective: test whether the q256 A-to-B pair-spacing effect replicates at
  q=128 with q as the only intended scientific change.
- A: baseline, target scale 1.0 and denominator scale 1.0.
- B: the same 1.10 pair-spacing intervention used by q256, target scale 1.1
  and denominator scale 1.1.
- Paired training seeds: 3, 4, 5.
- Excluded: new q256 seeds, arms C/D, tuning, checkpoint selection, and a new
  optimizer/RAdam study. RAdam remains a frozen nuisance constant because the
  q256 A/B trajectories used it; changing it would violate the only-q design.

The machine-readable authority is
`configs/second_q_ab_q128_learning_curve.frozen.json`.

## Frozen training contract

Training starts from the official EDM CIFAR-10 unconditional VP transfer
checkpoint on the q256 canonical dataset. The scientific configuration matches
the q256 A/B factorial path except `q=128`:

- ddpmpp / ECT, global batch 128, batch-gpu 16;
- RAdam, lr 1e-4, dropout 0.2, no augmentation or xflip;
- k=8, b=1, c=0, double=10000, EMA beta 0.9993;
- FP16 + AMP, TF32 off, one worker, single-GPU trajectories;
- final budget 1024 kimg;
- immutable full states and EMA snapshots at 256, 384, 512, 640, 768, 896,
  and 1024 kimg.

The training implementation must be byte-equivalent to reference commit
`c8721a05227f3ff171f8dc1f559a64d58281c0ae` over the seven frozen training
paths listed in the JSON contract. Documentation, launchers, and result files
may be descendants; the training path may not change.

## Frozen evaluation contract

Primary endpoint is NFE1 FID-50k. Every evaluation job uses FP32, exactly
50,000 generated samples with sample seeds 0-49999, metric seed 20260730, and
the same detector/reference identities as the q256 replay curve. KID-50k is
secondary and must reuse the exact generated features from its paired FID job.

Primary budgets are 512, 640, 768, 896, and 1024 kimg. Execution priority is
768, 896, 1024, 640, then 512 kimg. This order puts the q256 high-quality
crossing first: q256 mean NFE1 FID-50k is 11.01 at 768, 9.61 at 896, and 8.78
at 1024 for A; B is 10.18, 9.03, and 8.33. The 640 and 512 points bracket the
approach to that region.

Secondary NFE2 uses `mid_t=[0.821]` at 768, 896, and 1024 kimg. It must not
delay or redefine completion of the 30-job primary matrix.

Analysis is paired by training seed at each budget. Report absolute A/B values,
B-A, the mean paired delta, sample SD, range, and directional wins. With three
training seeds, results are descriptive; checkpoints and sample seeds are not
additional independent replicates.

## Launcher and runtime plan

Static validation is intentionally possible before Role E responds:

```bash
python scripts/run_second_q_ab_q128.py validate
```

Role E should copy and complete
`configs/role_e_q128_dataset_verdict.template.json`. A conclusive verdict must
be `SEMANTIC_EQUIVALENT` or `NOT_EQUIVALENT` and must contain immutable semantic
and evidence-manifest hashes. The template's `UNRESOLVED` value is rejected.

After a verdict arrives, run a no-training machine preflight:

```bash
python scripts/run_second_q_ab_q128.py preflight \
  --verdict /path/to/role_e_dataset_verdict.json
```

Only after preflight returns `status=GO`, launch one seed per GPU. Each worker
runs its paired A then B cells sequentially:

```bash
python scripts/run_second_q_ab_q128.py run \
  --verdict /path/to/role_e_dataset_verdict.json \
  --seed 3 --gpu-id 0 --master-port 29631
```

Use distinct GPUs and master ports for seeds 4 and 5. The launcher does not
retry a crash and refuses an existing seed directory. Recovery or resume needs
an explicit adjudication so a partial A/B pair cannot silently enter the
formal matrix.

The runtime is the byte-verified q256 deterministic PyTorch 2.2 / CUDA 12.3
SIF plus its extracted sandbox. Preflight verifies the SIF against the q256
release SHA256 manifest and records source, dataset, transfer, runtime, and Role
E verdict identities before any training begins.

## Checkpoint and storage plan

For every seed-arm trajectory, retain at each frozen budget:

- `training-state-kimgNNNNNN.pt` with model, EMA, optimizer, GradScaler,
  counters, RNG, and sampler state;
- `network-snapshot-kimgNNNNNN.pkl` for evaluation;
- `training_options.json`, telemetry, log, launch record, and preflight receipt.

Hash each immutable state and snapshot immediately after training, then bind
the primary and secondary evaluation manifests to those hashes. Do not use a
mutable `latest` file as a formal checkpoint identity.

## Compute estimate

Evidence basis:

- PR31 q128 0-to-256 training: about 2,160-2,192 seconds per trajectory.
- q256 256-to-1024 replay: 2.18-2.38 A100 GPU-hours per trajectory.
- q256 FID/KID shared-feature evaluation: mean 405.8 seconds per NFE1 job and
  555.8 seconds per NFE2 job.

Estimated cost:

| Work | Jobs | Estimated GPU-hours |
| --- | ---: | ---: |
| Fresh q128 training, 0-1024 | 6 | 17.3 |
| Primary NFE1 FID/KID-50k, 5 budgets | 30 | 3.4 |
| Secondary NFE2 FID/KID-50k, 3 budgets | 18 | 2.8 |
| Total evidence-basis estimate | 54 | 23.5 |
| Total with 10% operational reserve | 54 | 25.9 |

With three comparable A100-class GPUs, expect roughly 5.8 hours for training
plus 2.1 hours for evaluation if jobs remain GPU-saturated, about 7.9 hours
wall-clock before archive verification. Reserve about 55-65 GB for 42 full
states, 42 EMA snapshots, logs, hashes, and temporary manifests; generated
samples/features should be scratch assets, not part of the permanent archive.

## GO / NO-GO checklist

GO requires all of the following:

1. Role E conclusive signed verdict with matching dataset identities and
   immutable evidence hashes.
2. Canonical dataset ZIP SHA256 `08c9ed1b...4f372` present on the execution
   node, regardless of whether old q128 data is semantically equivalent.
3. Official transfer checkpoint SHA256 `4d5dcc1f...b4da`.
4. Clean repository and byte-equivalent frozen training paths.
5. Byte-verified q256 deterministic runtime and one GPU per seed worker.
6. Empty, non-overwriting run directories and at least 65 GB free archive
   capacity.

Until item 1 passes, the exact NO-GO reason is:

> PR31 records unresolved q128-versus-q256 dataset semantic equivalence; using
> the legacy q128 training as cross-q confirmation would not establish that
> only q changed.

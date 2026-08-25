# Second-q q128 A/B learning-curve protocol V2

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run preparation
- Origin Date: 2026-08-23
- Verification Status: ANALYZED
- Version Label: second_q_q128_ab_v2_canonical_dataset

## Amendment record

V2 supersedes, but does not rewrite, commit
`05157e7a0532b02184e2c38d051fe8c4c8aabac4` and protocol
`second-q-q128-ab-pair-spacing-v1`.

- Reason: unresolved dataset identity.
- Scientific results observed before amendment: **false**.
- Training started before amendment: **false**.
- Scientific design changed: **no**.
- Provenance correction: q128 now binds directly to the byte-identical q256
  canonical CIFAR-10 archive, SHA256
  `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.

This is a pre-execution provenance correction. PR31/PR32 q128 assets remain
historical exploratory evidence; V2 never uses, resumes, mixes, or substitutes
them. Role E does not need to establish semantic equivalence between the old
`9818...` ZIP and canonical `08c9...` ZIP for V2 to run.

The machine-readable authority is
`configs/second_q_ab_q128_learning_curve_v2.frozen.json`.

### Strict-protocol validation amendment

The first authorized six-cell launch was rejected before the training loop by
the inherited validation check `q256_target_weight_v1 requires q=256`. All six
GPUs remained unused; no optimizer step, checkpoint, metric, or scientific
result was produced. The failed launch directories contain only immutable
launch records and are retained outside the formal run root.

Before retry, the strict protocol's q validation was amended from `q == 256`
to `q in {128, 256}`. No schedule, target, denominator, loss, optimizer, RNG,
or checkpoint computation changed. The amended `training/loss.py` SHA256 and
four required A/B native-parity tests at q128/q256 are frozen in the config.
The remaining six training-path files remain byte-identical to the q256
reference commit.

## Scientific contract retained from V1

- q=128; paired training seeds 3, 4, and 5.
- A: baseline target/denominator scales 1.0/1.0.
- B: q256-matched 1.10 pair-spacing intervention, scales 1.1/1.1.
- No new q256 seeds, arms C/D, optimizer ablation, or RAdam mechanism study.
- Fresh training from the same official transfer checkpoint as q256.
- Immutable state and EMA checkpoints at 256, 384, 512, 640, 768, 896, and
  1024 kimg.
- Primary: NFE1 FID-50k at 512, 640, 768, 896, and 1024 kimg.
- Secondary: KID-50k from the same generated features; NFE2 with
  `mid_t=[0.821]` at 768, 896, and 1024 kimg.
- FP32 evaluation, sample seeds 0-49999, metric seed 20260730, and the frozen
  q256 detector/FID-reference/KID-reference identities.

## Canonical dataset-loader gate

Before training, the launcher invokes the formal runtime Python on
`scripts/check_canonical_dataset_loader.py`. The smoke exhaustively consumes
all 50,000 records through the repository's actual `ImageFolderDataset` and
requires:

- exact archive SHA256 `08c9...4f372`;
- loader SHA256 `f46fe15e...1c2` at the recorded Git commit;
- 50,000 RGB images, CHW uint8, 32x32, xflip disabled;
- integer labels 0-9 from `dataset.json`, one-hot mapping consistency, and
  exactly 5,000 samples per class;
- preprocessing exactly `float32(image) / 127.5 - 1.0`, with range [-1, 1];
- q256 detector and FID/KID reference hashes included in the receipt.

The receipt records hashes of sorted filenames, decoded CHW pixels, integer
labels, and the preprocessed float32 stream. This is not an equivalence test
against the old q128 ZIP. It proves that the new q128 run consumes the frozen
q256 archive through the frozen loader and preprocessing path.

The only successful preflight status for V2 is
`GO_CANONICAL_DATASET`. Any other status is a provenance NO-GO.

## Evaluation ordering is not selection

Primary execution priority remains 768, 896, 1024, 640, then 512 kimg so the
high-quality crossing region is available first for pipeline diagnostics.
This order is scheduling only:

> execution priority is not an adaptive selection policy.

All 30 frozen NFE1 jobs—3 seeds x 2 arms x 5 budgets—are mandatory. No early
FID/KID value may cancel, add, replace, or select a budget. The primary curve
is complete only after every frozen job passes.

## Commands

Local static validation:

```bash
python scripts/run_second_q_ab_q128.py validate
```

Execution-node preflight; this performs the exhaustive loader smoke but starts
no training:

```bash
python scripts/run_second_q_ab_q128.py preflight \
  --receipt-out /root/second_q_q128_ab_v2/preflight/GO_CANONICAL_DATASET.json
```

Only after the exact `GO_CANONICAL_DATASET` receipt, launch one independent
seed-by-arm cell per GPU with distinct ports:

```bash
RECEIPT=/root/second_q_q128_ab_v2/preflight/GO_CANONICAL_DATASET.json
python scripts/run_second_q_ab_q128.py run --preflight-receipt "$RECEIPT" --seed 3 --arm A --gpu-id 0 --master-port 29631
python scripts/run_second_q_ab_q128.py run --preflight-receipt "$RECEIPT" --seed 3 --arm B --gpu-id 1 --master-port 29632
python scripts/run_second_q_ab_q128.py run --preflight-receipt "$RECEIPT" --seed 4 --arm A --gpu-id 2 --master-port 29633
python scripts/run_second_q_ab_q128.py run --preflight-receipt "$RECEIPT" --seed 4 --arm B --gpu-id 3 --master-port 29634
python scripts/run_second_q_ab_q128.py run --preflight-receipt "$RECEIPT" --seed 5 --arm A --gpu-id 4 --master-port 29635
python scripts/run_second_q_ab_q128.py run --preflight-receipt "$RECEIPT" --seed 5 --arm B --gpu-id 5 --master-port 29636
```

The six processes may run concurrently. Pairing remains statistical and is
still keyed by training seed; physical simultaneity does not change the paired
design. Every cell refuses an existing arm directory and never auto-retries a
crash. Training remains disabled during `validate` and `preflight`.

## Immutable EMA snapshot export

Training uses `--snap=0` so evaluation snapshots are derived only from the
frozen full states, not from mutable periodic snapshot timing. For every one of
the 42 seed-by-arm-by-budget states,
`scripts/export_second_q_ab_snapshots.py` exports an EMA-only
`network-snapshot-kimgNNNNNN.pkl` and a receipt binding its source-state hash,
canonical EMA hash, snapshot hash, seed, arm, q, and budget. Export is CPU-only
and must leave Python, NumPy, and Torch CPU RNG unchanged.

For the already-running c1e2a19 trajectories, the exporter runs from a separate
artifact-tools checkout so the active training checkout and process-loaded code
remain unchanged. This is an I/O-only artifact completion step; immutable
training states remain the source of truth.

## Compute and claim boundary

The V1 estimate remains approximately 23.5 A100 GPU-hours, or 25.9 GPU-hours
with a 10% operational reserve. With three comparable A100-class GPUs, the
expected training-plus-evaluation wall time is about 7.9 hours before archive
verification.

If V2 completes its provenance, training, checkpoint, and evaluation gates,
the intended Methods claim is:

> For the cross-q study, both discretization settings use the same
> byte-identical CIFAR-10 archive, training configuration, paired seeds, and
> evaluation protocol; only q is changed.

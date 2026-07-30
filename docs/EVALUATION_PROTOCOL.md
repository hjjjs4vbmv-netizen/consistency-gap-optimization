# Evaluation protocol

## Active staged protocol (frozen 2026-07-30)

**Protocol ID:** `staged-checkpoint-evaluation-v1`. This is the authoritative
protocol for the next checkpoint-evaluation cycle. It freezes the evaluation
configuration *before* any new comparative result is inspected. Historical
results in this repository retain their original labels; they are not silently
promoted to quick or formal results under this protocol.

The two stages have deliberately different evidence classes:

| Stage | Purpose | Metrics per checkpoint/NFE | Reporting class |
| --- | --- | --- | --- |
| Quick | evaluator smoke test and candidate screening | KID-5k and FID-5k | screening/proxy only |
| Formal | final, eligible checkpoints only | KID-50k and FID-50k | formal benchmark |

### Frozen sampling and metric settings

The following settings apply to every checkpoint and to both stages. They may
not be changed per method, training seed, checkpoint, or after a result is
seen.

| Setting | Frozen value |
| --- | --- |
| Precision | FP32; TF32 and reduced-precision reductions disabled |
| GPU topology | one GPU (explicit sample seeds are single-GPU only) |
| NFE=1 | `mid_t=[]` |
| NFE=2 | `mid_t=[0.821]` |
| Metric repetitions | exactly one per metric/cell |
| KID subset seed / evaluator seed | `20260730` |
| Real reference | complete canonical CIFAR-10 training archive, `xflip=False` |
| Feature extractor | repository-pinned Inception detector used by `metrics/` |

`metric-repeats=1` is intentional: repeating an identical fixed sample set is
not an additional independent observation. A given sample seed must derive the
same initial latent for NFE=1 and NFE=2; NFE=2 intermediate noise must be
derived deterministically from that same seed. Every run records the complete
seed range, evaluator seed, checkpoint SHA256, dataset SHA256, Git revision,
NFE, `mid_t`, precision, device, and metric implementation names.

### Quick evaluation (screening only)

Run KID-5k (`kid5k_full`) and FID-5k (`fid5k_full`) for each checkpoint and
for each NFE separately. The generated sample set is exactly the ascending
integer range **0-4999** for every cell. Both metric files must contain one
finite record; a missing or failed metric makes the cell incomplete. Quick
numbers must always be labelled **“5k-sample screening proxy; not a formal
50k benchmark.”** They may guide triage but must not be used as final claims.

The first execution after this freeze is an evaluator smoke on one already
available checkpoint, covering both NFEs and both 5k metrics (four cells). It
validates the evaluator path and output schema only; its numbers remain quick
screening evidence.

### Formal evaluation (eligibility-gated)

Formal evaluation runs KID-50k (`kid50k_full`) and FID-50k (`fid50k_full`) for
each eligible checkpoint and each NFE separately. The generated sample set is
exactly the ascending integer range **0-49999** for every cell. The formal run
must use the same frozen settings above and must produce exactly one finite
result record for each metric.

No checkpoint is eligible for formal evaluation until its training-integrity
receipt has status `passed` *and* the evaluator recomputes a SHA256 matching
the receipt. The receipt is a machine-readable artifact produced by the
training-completeness check and must bind all of the following to the evaluated
file:

- checkpoint path/basename and SHA256;
- training run ID, method, training seed, and declared budget;
- completion at the declared budget (no early or regressed checkpoint);
- required training logs/state present and internally consistent;
- finite-loss/finite-state check passed; and
- check script version, Git revision, timestamp, and final `status: passed`.

The receipt schema is frozen as follows (additional fields are allowed):

```json
{
  "schema_version": 1,
  "status": "passed",
  "checkpoint_id": "sigmoid_seed0_16k",
  "checkpoint_path": "/mnt/ect_project/checkpoints/sigmoid_seed0_16k.pkl",
  "checkpoint_sha256": "<64-character SHA256>",
  "training_run_id": "<immutable training-run ID>",
  "method": "sigmoid",
  "training_seed": 0,
  "budget_kimg": 16,
  "completion_passed": true,
  "logs_state_consistent": true,
  "finite_loss_state_passed": true,
  "checkpoint_load_passed": true,
  "ema_present": true,
  "ema_finite_passed": true,
  "schedule_identity_passed": true,
  "global_gap_scale_identity_passed": true,
  "method_identity_passed": true,
  "checker_version": "2",
  "checker_git_commit": "<Git revision of the integrity checker>",
  "checked_at_unix": 0
}
```

An absent, malformed, stale, failed, or hash-mismatched receipt is a hard
block: do not launch a formal metric job and do not create a partial formal
table. Quick evaluation does not waive this gate.

`kid50k_full` passes the frozen evaluator seed to KID's subset sampler. This
contract is covered by the staged-evaluation test suite. A formal run must
still record that seed and must not substitute an unseeded KID result.

### Execution order and result contract

Execute the cycle in this order and stop on a failed prerequisite:

1. Freeze this protocol (this change).
2. Run the existing-checkpoint 5k evaluator smoke.
3. Verify fixed-generation-seed determinism independently for NFE=1 and
   NFE=2, including repeated runs and work-group sizes 8 and 16.
4. Build and validate the unified result table/statistics tooling using the
   smoke outputs; it must distinguish `quick` from `formal` evidence.
5. Run the complete quick 5k screening matrix.
6. Produce and verify training-integrity receipts for candidate formal
   checkpoints; fix the seeded KID-50k readiness gap if still open.
7. Run the complete formal 50k matrix only for eligible checkpoints, then
   publish its separate formal summary.

The unified per-cell table must include at least: evidence class, method,
training seed, checkpoint ID/SHA256, integrity-receipt status, NFE, `mid_t`,
metric name/value, generated-sample count and exact seed range, evaluator/KID
seed, dataset SHA256, evaluation Git revision, run path, and completion
status. Statistics must never pool checkpoints, NFEs, quick and formal rows,
or different metrics. For the fixed/global-only confirmatory matrix, pairing
is exactly `training_seed + budget_kimg + nfe + metric`, the delta is
`global_only - fixed`, and negative values favor global-only. Missing or
duplicated arms are a hard collection failure. The collector emits the
per-seed `paired_differences.csv` and separate paired statistics JSON/Markdown.

`docs/FINAL_PERFORMANCE_EVALUATION.md` and
`docs/ROLE_A_QUANTITATIVE_EVALUATION.md` describe earlier scoped experiments.
For this new cycle, this staged protocol takes precedence for metric settings,
eligibility, execution order, and evidence labels.

This document defines the reproducible Role D sampling protocol. It separates
historical evidence from results produced under the current protocol.

## Result classes

`results/preliminary_seed42_fp32_8ksteps/` contains preliminary historical
results. They were produced from an older code base and an approximately
8k-update checkpoint. They are retained for reference only, are not directly
comparable with the current B/C protocol, and must not be reported as a final
benchmark.

`results/fixed_seeds_0_63_fp32_8ksteps/` is also a preliminary historical
smoke from that checkpoint. Its directory-level README records why it does not
satisfy the current protocol. It must not be regenerated or treated as a
current-protocol result.

Current protocol results must record the checkpoint SHA256, evaluation Git
commit, seeds, NFE, `mid_t`, precision, GPU, work-group sizes, and repeated-run
determinism status in `metadata.json`. They must also record elapsed time, image
and seed counts, the complete seed list, per-mode NFE and `mid_t`, generator
implementation, and the actual forward batch size.

## Fixed-seed protocol

- Seeds 0-63 denote 64 per-sample seeds, not 64 repeated metric runs.
- Generate one 32x32 RGB PNG for each seed in 0-63.
- NFE=1 uses `mid_t=[]`.
- NFE=2 uses `mid_t=[0.821]`.
- For a given seed, NFE=1 and NFE=2 use the same initial latent.
- The NFE=2 intermediate noise is also deterministically derived from that seed.
- The 64 images per mode are used for visualization and determinism checks.
- Keep the model forward batch size at one.
- Treat 8 and 16 as work-group sizes, not model batch sizes.
- Require pixel-identical output across work-group sizes 8 and 16.
- Repeat each NFE configuration and require pixel-identical output.
- Isolate every result directory using the checkpoint filename and the first 12
  characters of its SHA256.
- Select precision explicitly. Use `fp32` for this acceptance smoke; use
  `checkpoint` only when intentionally preserving checkpoint-native precision.

The sampler writes to `<outdir>/<checkpoint-stem>-<sha256-prefix>/`. For
example, the official checkpoint is written under
`edm-cifar10-32x32-uncond-vp-4d5dcc1f1d0d/`. A run fails before publishing
metadata if either work-group or repeated-run determinism fails.

Each checkpoint directory contains:

```text
nfe1/
  images/
  grid_8x8.png
nfe2/
  images/
  grid_8x8.png
metadata.json
sha256_manifest.txt
```

The metadata schema records the evaluation Git commit, checkpoint path and
SHA256, checkpoint ID, seed list, NFE modes, `mid_t` per mode, precision,
device, GPU, elapsed time, image counts, generator implementation, actual
model forward batch size, verified work-group sizes, image dimensions, and the
overall determinism result.

## Official EDM checkpoint smoke

Download and verify the official NVIDIA EDM CIFAR-10 32x32 unconditional VP
checkpoint:

```bash
bash download_checkpoint.sh \
  --output /mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl
```

The expected checkpoint SHA256 is:

```text
4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da
```

Run the 64-image NFE=1 plus 64-image NFE=2 smoke:

```bash
bash scripts/sample_checkpoint.sh \
  /mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
  --outdir /mnt/ect_project/evaluations \
  --seeds 0-63 \
  --nfe 1 2 \
  --mid-t 0.821 \
  --work-group-size 8 \
  --verify-work-group-size 16 \
  --precision fp32 \
  --device cuda
```

The expected output root is:

```text
/mnt/ect_project/evaluations/edm-cifar10-32x32-uncond-vp-4d5dcc1f1d0d/
```

Verify the image counts and manifest after the command completes:

```bash
RESULT=/mnt/ect_project/evaluations/edm-cifar10-32x32-uncond-vp-4d5dcc1f1d0d
find "$RESULT/nfe1/images" -type f -name '*.png' | wc -l
find "$RESULT/nfe2/images" -type f -name '*.png' | wc -l
(cd "$RESULT" && sha256sum -c sha256_manifest.txt)
```

Both image counts must be 64 and every manifest entry must report `OK`.

## Metric boundary

This acceptance smoke does not run FID-50k, KID-50k, or any other distribution
metric. In particular, generating seeds 0-63 does not mean running FID 64
times. Its purpose is limited to checkpoint loading, fixed-seed generation,
visualization, output completeness, work-group invariance, and repeated-run
determinism. Historical seed42 FP32 FID/KID values remain preliminary evidence
and are not directly comparable with current B/C formal results. Formal
Formal FID-50k/KID-50k runs only under the active staged protocol's
training-integrity gate and 50k fixed-seed settings above.

## Git artifact policy

Do not commit the checkpoint or the 128 individual PNG files. For a completed
smoke run, commit only both 8x8 grids, `metadata.json`, and
`sha256_manifest.txt`. Keep the full output tree under
`/mnt/ect_project/evaluations/` as the external evaluation record.

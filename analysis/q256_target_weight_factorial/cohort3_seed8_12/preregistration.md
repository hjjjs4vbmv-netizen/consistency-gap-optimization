# q256 target-geometry × denominator-weighting Cohort III preregistration

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-21
- Verification Status: UNVERIFIED
- Version Label: q256_target_weight_factorial_cohort3_prereg_v1

Status: **frozen before any formal Cohort III training or quality evaluation**.
The five-GPU smoke and exact-resume checks are engineering gates and do not
measure generation quality.

## Prospective boundary

Cohort III is a new prospective validation cohort with seeds 8, 9, 10, 11,
and 12. Cohort I (seeds 3–5) and Cohort II (seeds 6–7) were already observed
before this design. Cohort II showed substantial seed- and trajectory-level
heterogeneity in component effects. Cohort III therefore does not
retrospectively belong to the original `n=3` preregistration.

The primary question is whether the full coupled gap intervention B is
reproducibly better than baseline A across unseen training trajectories. The
primary later-evaluation contrast is

`Delta_full = FID50k_NFE1(B) - FID50k_NFE1(A)`,

with directional hypothesis `Delta_full < 0`.

Secondary later-evaluation endpoints are KID-50k at NFE1 and FID/KID at NFE2.
Secondary structural contrasts are `C-A`, `B-D`, `D-A`, `B-C`, and
`I=B-C-D+A`. These contrasts measure trajectory-dependent component effects.
This study will not claim that target geometry universally dominates and will
not claim that optimizer state causes endpoint FID changes.

## Frozen source and training protocol

- Base merge commit: `64e56392883248668a92aa6c18c0cec3d1ef796f`.
- Formal training semantics reference:
  `dcca41b19e7c45512b5fbe98776520396a1bf9ac`.
- Execution branch: `experiment/q256-cohort3-seed8-12`.
- Training-core semantic diff between the two frozen commits must remain empty.
- q=256, 256.000 kimg, 2000 attempted iterations, 256000 processed images.
- Each run is fresh from the same immutable transfer checkpoint.
- No training-time metric evaluation; formal FID/KID is outside this task.

The four arms, always executed in order A → B → C → D within each seed, are:

| Arm | Target gap scale | Denominator gap scale | Interpretation |
| --- | ---: | ---: | --- |
| A | 1.0 | 1.0 | baseline |
| B | 1.1 | 1.1 | full coupled intervention |
| C | 1.1 | 1.0 | target geometry only |
| D | 1.0 | 1.1 | denominator weighting only |

All remaining training settings are frozen to the PR #70 q256 protocol:
DDPM++, ECT, `k=8`, `b=1`, `c=0`, double interval 10000, batch 128 with
`batch_gpu=16`, RAdam at `1e-4`, dropout 0.2, no augmentation or xflip, FP16
network with AMP, TF32 off, EMA beta 0.9993, workers=1, and no training
metrics. No new q, scale, learning rate, optimizer, schedule, Pseudo-Huber,
architecture, target, or denominator definition is authorized.

## Immutable assets and exact runtime

- CIFAR-10 archive:
  `/data/raw/ECT/datasets/cifar10-32x32-canonical-08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372.zip`
  with SHA256
  `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
- Transfer checkpoint:
  `/data/raw/ECT/pretrained/edm-cifar10-32x32-uncond-vp.pkl` with SHA256
  `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`.
- Runtime: NVIDIA NGC `nvcr.io/nvidia/pytorch:24.01-py3`, Python 3.10.12,
  PyTorch `2.2.0a0+81ea7a4`, CUDA runtime 12.3, cuDNN 8.x.
- Deterministic algorithms and deterministic cuDNN are on;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; `cudnn_benchmark=False`.
- Host Python 3.13 / PyTorch 2.8 / CUDA 12.8 is not an authorized fallback.

## Fixed hardware mapping and execution

| Physical GPU | Seed | Required arm order |
| ---: | ---: | --- |
| 0 | 8 | A → B → C → D |
| 1 | 9 | A → B → C → D |
| 2 | 10 | A → B → C → D |
| 3 | 11 | A → B → C → D |
| 4 | 12 | A → B → C → D |

All four arms for one seed stay on the same physical GPU. There is one active
training process per GPU, no DDP, and all five seed queues run concurrently.
Run directories are unique, regular directories and are never reused.

## Decision, failure, and resume policy

All five seeds and all twenty arms must finish regardless of early outcomes.
There is no result-dependent arm deletion, seed deletion, or stopping based on
relative loss, samples, FID, or KID. Relative sample quality is not inspected.

A nonzero arm exit stops that seed queue fail-closed. Logs and partial artifacts
are preserved and the incident is classified. A clean restart is not allowed
to obtain a more favorable trajectory. Deterministic continuation is allowed
only for an infrastructure failure when source, runtime, assets, GPU UUID, and
self-contained state remain identical and model, EMA, optimizer, GradScaler,
RNG, and sampler state restore exactly. The exact-resume gate must pass.

Formal training may start only after the exact runtime, asset hashes,
training-core identity, preregistration freeze, focused correctness tests,
shared-memory safety, five-A100 exclusivity, cross-GPU 32-attempt parity, and
uninterrupted-versus-resumed exact-trajectory gates all pass.

After 20/20 training completion, only a checkpoint handoff manifest is frozen.
Formal quality evaluation remains a later, independent operation.

# q256 g=1.10 target geometry x loss weighting: preregistration

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-19
- Verification Status: UNVERIFIED
- Version Label: q256_target_weight_factorial_prereg_v1

Status: **frozen before any formal four-arm training**. Engineering tests,
paired correctness audits, and 32-attempt smoke runs are gates, not quality
observations. No new four-arm FID or KID may be generated or inspected until
every gate has a versioned PASS receipt.

## Scientific question and boundary

Scaling the ECT consistency gap changes both the stop-gradient target geometry
and the finite-difference loss denominator. This experiment asks which factor,
or their interaction, changes finite-budget generation quality at q=256 and
g=1.10.

This design does not estimate mediation and cannot establish that optimizer
history causes, explains, or mediates a FID/KID change. It does not decompose a
quality effect into percentages.

## Independent replication unit

The independent unit is the training seed: seeds 3, 4, and 5 (`n=3`).
Minibatches, optimizer attempts, checkpoints, generation blocks, metric
repeats, and NFE modes are repeated measurements, not additional independent
training replicates.

## Frozen training factors

For every sampled `t`, first compute the official q=256 sigmoid schedule and
its g=1.10 scaled counterpart with the production mapping and clamp:

`r_1 = r(g=1.00)`, `r_g = r(g=1.10)`,
`Delta_1 = t - r_1`, and `Delta_g = t - r_g`.

The per-sample objective is

`ell_(T,W) = ||D_theta(x_t,t) - sg D_theta(x_(r_T),r_T)||_2 / Delta_W`.

| Arm | Target scale | Denominator scale | Interpretation |
| --- | ---: | ---: | --- |
| A | 1.00 | 1.00 | native baseline |
| B | 1.10 | 1.10 | native g=1.10 |
| C | 1.10 | 1.00 | target-geometry-only |
| D | 1.00 | 1.10 | loss-weighting-only |

The denominator always uses the realized per-sample `t-r` after production
mapping and clamping. C and D are never implemented as a batch-level scalar
multiple.

## Frozen common training protocol

- CIFAR-10 32x32 unconditional.
- Canonical archive SHA256:
  `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
- Authoritative initial EDM transfer checkpoint SHA256:
  `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`.
- Fresh training from that same immutable transfer checkpoint to 256 kimg.
- `ddpmpp`, ECT, q=256, k=8, b=1, c=0.
- Batch 128, `batch_gpu=16`, RAdam, lr `1e-4`, betas `(0.9,0.999)`,
  epsilon `1e-8`, no weight decay.
- Dropout 0.2, augmentation 0, dataset xflip false.
- FP16 network, AMP/GradScaler enabled, TF32 false, loss scaling 1.0.
- EMA beta 0.9993; curriculum double interval 10,000 ticks.
- Tick 10 kimg, no numbered snapshots or state dumps, latest checkpoint every
  10 ticks, preview every 26 ticks, no metrics during training.
- Per seed, all four arms use identical initialization source and the same
  seed-derived RNG and `InfiniteSampler` construction. Different seeds may run
  concurrently; arms within one seed run on the same GPU model and are not
  selected or stopped using intermediate quality.

The exact clean execution commit, training-code content hash, resolved config,
dataset and transfer hashes, model/EMA/optimizer initialization hashes,
Python/PyTorch/CUDA/GPU/container identity, and RNG/sampler protocol are bound
in the launch manifest before formal training. A semantic code change after a
formal launch invalidates every affected run.

## Reuse decision

All 12 runs are fresh. The archived compatibility audit for the old q256 A/B
controls has blocking missing or mismatched source RNG/sampler, state/config,
checkpoint-cadence, and protocol-hash fields; old global110 also does not share
the authoritative fixed source. Unknown fields are not imputed. Historical A/B
remain contextual evidence only and are not cells in this factorial.

## Frozen endpoints and evaluation

Primary endpoint:

- FID-50k at NFE=1.

Secondary endpoints:

- KID-50k at NFE=1.
- FID-50k and KID-50k at NFE=2, with `mid_t=0.821`.

The existing formal evaluator is frozen: FP32 sampling, generation seeds
`0-49999`, metric seed `20260730`, the same CIFAR-10 reference statistics, and
the final 256 kimg checkpoint. Every arm uses identical generation seed blocks.
Checkpoint, NFE, sample count, interaction definition, and primary metric
cannot be changed after results are visible.

## Preregistered contrasts

For each seed and endpoint, with lower values better:

- target geometry at baseline weighting: `Y_C - Y_A`;
- target geometry at g weighting: `Y_B - Y_D`;
- loss weighting at baseline target: `Y_D - Y_A`;
- loss weighting at g target: `Y_B - Y_C`;
- interaction: `I = Y_B - Y_C - Y_D + Y_A`.

Negative values indicate an improvement for the first term relative to the
second under the frozen contrast. Report every seed-level raw value and
contrast, then cross-seed mean, median, and range. Sampling-block variation is
descriptive and never increases `n` above three. No minibatch- or block-level
significance test is permitted.

## Preregistered interpretation branches

- `C approximately B < A` and `D approximately A`: target geometry dominates.
- `D approximately B < A` and `C approximately A`: loss weighting dominates.
- B is best while C and D alone do not reproduce it: target x weighting
  interaction.
- C and D each partly improve: both pathways contribute descriptively.
- Fresh B is not better than A: the original q256 quality effect did not
  reproduce; stop a strong gap-benefit narrative.
- Direction differs materially across seeds: report heterogeneity only and do
  not claim that the endpoint mechanism is determined.

"Approximately" is a descriptive assessment against the complete seed-level
values and ranges; it is not a post-hoc equivalence margin or a license to
relabel the primary contrast.

## Correctness gates before formal training

The gates run in this order and fail closed:

1. Unit and synthetic tests.
2. Factorized A versus canonical sigmoid parity and factorized B versus
   canonical global-sigmoid g=1.10 parity for per-sample loss, reduced loss,
   and gradients under identical draws.
3. A/D same-target and B/C same-target identities, plus per-sample realized
   denominator checks including clamp cases.
4. Stop-gradient, immutable source, and arm-order invariance checks.
5. Deterministic same-batch gradient rerun.
6. Seed-3 A/B/C/D smoke, exactly 32 optimizer attempts each.
7. Checkpoint save/load/resume and repeated same-source smoke identity.
8. Finite loss, sanitized gradient, parameter update, EMA, positive realized
   denominators, and loadable final state.
9. Skipped-step behavior must match the paired canonical AMP warm-up pattern;
   any extra or arm-specific skip is unexpected and blocks formal training.
10. Full test suite and `git diff --check`.

A failed gate stops formal action. Engineering fixes require a new numbered
gate receipt; tolerances and scientific definitions are not relaxed.

## Scheduling, compute, and storage forecast

Historical q256 256-kimg A100 runs took 1,446-2,147 seconds per cell
(24.1-35.8 minutes, 0.40-0.60 GPU-hours). The frozen planning envelope is
0.65 GPU-hours per training run, or at most 7.8 GPU-hours for 12 runs before
retries. On the previously recorded two-A100-80GB server, use six sequential
two-run waves, pairing arms within seed on the same GPU model. Expected training
wall time is 3.0-4.5 hours including validation and checkpoint I/O.

The formal four-endpoint evaluation is provisioned separately at approximately
12-16 GPU-hours total. Training artifacts require about 0.90 GB per run based
on historical final snapshot/state packages (about 11 GB total). Allow at least
30 GB for training, evaluation images/features, receipts, and figures, plus a
20 GB safety reserve: formal launch requires at least 50 GB free on the target
filesystem.

Once server access and gates are resolved, training plus evaluation is expected
to finish within one day and therefore before 2026-09-05. This forecast is not
an authorization: live GPU identity, free disk, and competing processes must
be re-audited immediately before launch.

## Current formal stop

`formal_training_authorized=false`. As of 2026-08-19,
`region-9.autodl.pro:34360` returns TCP `Connection refused` before SSH key
authentication. Live source identity, GPU availability, disk capacity, active
processes, and server worktree cleanliness therefore remain unverified.


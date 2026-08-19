# q256 g=1.10 one-time RAdam moment transport: preregistration

Status: **frozen before any formal continuation training**. Held-out optimizer
replay, engineering smoke tests, and compatibility audits are gates, not formal
quality observations. No FID or KID may be generated or inspected before those
gates are resolved.

## Question and estimands

At the common 256 kimg q256 fixed-schedule branch source, compare:

- **F**: `sigmoid`, gap scale 1.00, inherited optimizer state;
- **G**: `global_sigmoid`, gap scale 1.10, inherited optimizer state;
- **T**: `global_sigmoid`, gap scale 1.10, with the single offline mapping
  `exp_avg *= a_s` and `exp_avg_sq *= a_s**2` (and
  `max_exp_avg_sq *= a_s**2` if present).

The primary paired contrast is T-G. The secondary paired contrast is T-F.
Training seed is the independent replication unit (`n=3`, seeds 3, 4, 5).
Checkpoint probes and sample blocks are repeated measurements, not replicates.

## Source and execution identity

- Recovered continuation implementation commit:
  `6b26c04d37789ee59a620df40a71f5eb76bd7d76`.
- Recovered training-code content hash:
  `3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43`.
- The experiment branch may add audit and orchestration files, but `ct_train.py`
  and `training/` must remain byte-identical to the recovered implementation.
- The exact clean Git commit used by the runner is recorded in the immutable
  launch receipt before the first formal process starts.
- The runtime execution manifest is written outside the Git worktree and binds
  that exact clean commit.  This avoids the impossible self-reference that
  would result from embedding a commit's own hash in a file in that commit;
  the manifest itself is immutable and its SHA256 is recorded in every plan
  and receipt.
- Formal data archive SHA256:
  `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
- The server audit archive has a different ZIP-container SHA256 but has exactly
  the same 50,001 member names, order, sizes, CRCs, and bytes. Its frozen
  name/content digest is
  `c96fdaa57f8bdb6b697664831085acc7fc42aa927b41e2bde0863ef26057f27f`.
  It is used only where old factorial receipts require its container hash; the
  formal continuation uses the canonical archive above.

The six authoritative fixed/global110 256 kimg checkpoint pairs and their
options, snapshots, optimizer contents, and receipts are audited in
`control_compatibility.json`. Directory names alone never establish identity.
An old arm is reusable only when every load-bearing field is PASS or when an
explicitly enumerated unavailable serialization field is proven equivalent by
the deterministic replay gate. A source mismatch forces a fresh arm.

The already discovered old global110 states do not share the fixed 256 kimg
branch source, so old G is not eligible. A fresh G is paired with T. If the
completed compatibility audit cannot establish old F, F is also rerun from the
same source; this contingency is fixed before held-out results are inspected.

## Scalar calibration and held-out split

Canonical factorial audit IDs are sorted before computing any aggregate:

- calibration ranks 0, 2, 4, 6: `2026081101`, `2026081103`, `2026081105`,
  `2026081107`;
- held-out ranks 1, 3, 5, 7: `2026081102`, `2026081104`, `2026081106`,
  `2026081108`.

For calibration batch `b`,

`a_s,b = dot(G_1.10, G_1.00) / dot(G_1.00, G_1.00)`.

The whole-model seed coefficient is the unweighted median of its four
calibration values. It must be finite and positive and is never clipped,
layerwise fit, or tuned using held-out replay, loss, FID, or KID. Frozen values:

| seed | `a_s` |
| ---: | ---: |
| 3 | 0.8370121196598016 |
| 4 | 0.8073491626309143 |
| 5 | 0.8233457134218897 |

## Fail-closed manipulation gate

For each held-out batch, replay disposable cloned state for F, G, T, and the
exact-scalar T control using the exact passed factorial definition of
`R_opt`, parameter support, weighting, epsilon, and optimal scalar alignment.
Formal continuation is authorized only when all of the following hold:

1. all values are finite and no batch is skipped;
2. source preservation, deterministic rerun, AMP, and branch-order checks pass;
3. every seed has positive suppression
   `1 - median(R_opt_T) / median(R_opt_G)`;
4. cross-seed median suppression is at least 0.50;
5. every seed's median exact-scalar transported residual is at most 0.01;
6. every seed's median `||U_T|| / ||U_F||` is in [0.90, 1.10];
7. the source checkpoint remains byte-identical.

A failed scientific gate is a formal **NO-GO**. Thresholds are not changed and
formal GPU training is not started. Engineering failures may be corrected and
the identical gate rerun with a receipt of each failed attempt.

## Transformation and replay gates

The offline transformer writes a new no-replace checkpoint and sidecar. It may
modify only supported RAdam moments and add a provenance marker. Model, EMA,
parameter-group hyperparameters, optimizer step/rectification state,
GradScaler, loss/schedule state, counters, and every other serialized field are
canonically preserved. Unknown tensor-valued optimizer fields, repeated
transport, source mutation, RNG mutation, missing state, or a hash mismatch are
fatal.

Before formal training:

- an `a=1` transform must be an exact in-memory moment no-op;
- direct g=1.10 resume and `a=1`-checkpoint resume must produce canonically
  identical training state and snapshot after exactly 32 optimizer attempts;
- seed-3 real-`a_3` smoke runs are repeated twice from the same immutable
  transformed source, for 32 attempts each, and must have identical final
  hashes, finite loss/state, and strict save/load success;
- smoke tests never generate or inspect FID/KID.

The legacy checkpoint format does not serialize Python, NumPy, CPU/CUDA RNG or
the `InfiniteSampler` cursor. This absence is recorded rather than imputed.
Each segment follows the recovered control protocol: startup deterministically
reseeds from the training seed and reconstructs the sampler. Identical derived
startup-stream hashes and the 32-step replay are required. The transformer
itself must leave the caller's Python, NumPy, CPU, and all-device CUDA RNG state
unchanged.

The recovered checkpoint was written with NumPy 2 module paths while the
frozen training container supplies NumPy 1.24.  A load-only launcher maps the
legacy `numpy._core` pickle lookup to `numpy.core` before invoking unchanged
training/audit code.  It never rewrites a source artifact, and the launcher is
part of the hashed execution code.  Its direct-load and no-op replay checks are
mandatory before formal training.

## Formal continuation protocol

All formal arms start from the per-seed fixed 256 kimg source. T starts from its
one-time transported copy; G (and fresh F if required) starts from an immutable
unmodified staged copy. Each arm uses three recovered restart segments:

`256 -> 512 -> 768 -> 1024 kimg`.

Frozen settings:

- CIFAR-10 32x32 unconditional; `ddpmpp`; ECT; q=256; k=8, b=1, c=0;
- batch 128, `batch_gpu=16`, one A100 per process;
- PyTorch RAdam, lr `1e-4`, betas `(0.9, 0.999)`, eps `1e-8`, no weight decay;
- dropout 0.2, augmentation 0, xflip false;
- FP16 network plus AMP/GradScaler; TF32 false;
- EMA beta 0.9993; curriculum double interval 10,000 ticks;
- 10 kimg tick, latest checkpoint every 10 ticks, preview sampling every 26
  ticks, evaluation callback cadence 50 ticks, no numbered snapshot/state
  dumps, and an empty metrics list during training;
- the same seed, segment boundaries, data ordering, startup RNG construction,
  logging, checkpoint format, and container/runtime as the recovered controls.

Within a seed, paired arms run on the same physical GPU and never concurrently.
Different seeds may run in parallel. Outputs are new paths below
`/data/raw/ECT/ect_runs/q256_g110_moment_transport`; existing source, fixed,
global110, audit, and evaluation paths are read-only. A dedicated tmux session,
exclusive lock, free-space/GPU-memory checks, command logs, environment
snapshots, timestamps, exit receipts, input/output hashes, and checkpoint
validation are mandatory. Code changes after any formal start invalidate every
affected seed.

The host `tmux` process launches the frozen Apptainer command because the
container intentionally has no tmux binary.  Each one-GPU worker receives
explicit `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`;
ports are a frozen base plus the physical GPU index and are recorded in the
environment receipt.

## Diagnostics and evaluation

At 256/512/768/1024 kimg, record loss, gradient/update/EMA norms, lr,
GradScaler, skipped steps, nonfinite counts, elapsed and GPU hours, T-G and T-F
parameter/EMA distances, and frozen paired gradient/optimizer probes. Probes
operate on copies with frozen batches/RNG and cannot mutate training state.

At 1024 kimg, use the exact existing formal evaluator, FP32 sampling,
generation seed blocks, and strict checkpoint loading for NFE=1 FID-50k/KID
and NFE=2 FID-50k/KID with the existing intermediate-time setting. At 512 and
768 kimg, match an existing formal protocol when available; otherwise use the
frozen 5k proxy and label it as such. No checkpoint, NFE, coefficient, duration,
or sample block is selected after viewing quality.

Report all seed-level F/G/T values, paired T-G and T-F differences, cross-seed
mean/median/range, within-seed sample-block spread, diagnostics, and compute
cost. No minibatch- or block-level p-value is interpreted as `n>3`.

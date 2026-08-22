# q256 target-geometry × denominator-weighting exact-budget replay

**Training status: 12/12 trajectories PASS**

**Checkpoint coverage: 72/72 replay milestones PASS**

**EMA snapshot coverage: 84/84 PASS**

**1024 replay parity: 12/12 bitwise-equivalent**

**Formal FID/KID evaluation: 168/168 jobs PASS**

## Scope

Seeds 3–5 and frozen arms A/B/C/D were replayed from their immutable formal 256 kimg full-states to a total budget of exactly 1024 kimg. No 0→256 training, seed extension, parameter sweep, or checkpoint selection was performed. FID/KID remained disabled throughout training and was executed only after the complete trajectory audit passed.

Immutable replay milestones are 384, 512, 640, 768, 896, and 1024 kimg. The 256 kimg rows reference the frozen source states. Every milestone is keyed by exact `cur_nimg`, not tick number.

## Training integrity

- All states contain online model, EMA, complete RAdam state, GradScaler, counters, loss/control state, rank-local RNG, and sampler state.
- Every cell reached `cur_nimg=1024000` and `attempted_iteration=8000`.
- Strict telemetry reported no non-finite loss/update/model/EMA/factor events and no non-positive denominator events.
- Every formal arm has exactly one launcher START and one matching END receipt; no formal trajectory crashed or required recovery. Two copied 256 kimg source logs contain their own earlier resume history, which is not counted as replay recovery. The separate seed3/armA saver smoke is archived as engineering evidence and is not part of the 12 formal trajectories.
- Training commit: `c8721a05227f3ff171f8dc1f559a64d58281c0ae`.

## Canonical 1024 parity

All 12 replay endpoints are canonically bitwise-equivalent to the corresponding PR #76 endpoints for online model, EMA, optimizer state, GradScaler, loss/control state, RNG/sampler state, trajectory config, counters, and factorial identity. File-level `.pt` SHA256 is reported separately and is not used as the parity criterion.

## Compute

Total replay compute across the 12 single-GPU trajectories was 27.376 A100 GPU-hours, including immutable checkpoint I/O in elapsed wall time.

## Archive

Server archive root: `/data/raw/ECT/ect_runs/q256-target-weight-replay-curve-v1-20260822`

The archive contains the 72 immutable replay states, 84 EMA snapshots, 12 frozen 256 kimg source states, deterministic runtime image, code bundles, resolved options, telemetry, logs, audits, and saver smoke evidence. `artifact_hashes.sha256` is generated after server-side transfer verification.

## Formal learning-curve evaluation

All 168 frozen jobs completed: FID-50k and KID-50k at NFE=1 and NFE=2 for every seed×arm×budget checkpoint. NFE=2 uses `mid_t=0.821`. Evaluation used FP32, 50,000 samples, generation seeds 0–49999, metric seed 20260730, and byte-identical generated features within every KID/FID pair. The compact results are under `fidkid50k-final-20260823/`; regenerable sample and feature arrays are intentionally excluded from the final archive.

## Limitations

This package establishes a deterministic model trajectory, replay identity, and formal FID/KID learning curve. It contains only three formal training seeds; paired contrasts must remain seed-level, and no mechanism should be inferred from endpoint FID/KID alone.

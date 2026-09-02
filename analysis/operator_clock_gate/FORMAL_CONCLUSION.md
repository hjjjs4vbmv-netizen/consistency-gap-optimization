# Operator-clock Jacobian audit: formal conclusion

Date: 2026-08-26
Status: **execution complete; smooth-Jacobian claims fail closed**

## Decision

The formal audit completed the frozen `4 arms x 4 audit minibatches x 8
directions` field and algorithmic matrix, plus matched counterfactual rollouts
at horizons `1/4/16/64`. The execution is complete and source-preserving, but
the evidence does **not** license treating the real FP16/GradScaler training
step as a conventional smooth Jacobian at this state.

For this source state:

- retain the squared-loss operator only as a theoretical baseline;
- reject the recompute-and-detach field estimate as a converged classical
  derivative under the frozen 5% gate;
- reject the full algorithmic linearization because the finite-difference
  branches cross AMP skip regimes;
- use the matched micro-rollout as the primary local counterfactual evidence.

## Frozen source and matrix

- Source-selection rule: lowest eligible seed in the archived verified q256
  fixed-baseline 256-kimg source manifest.
- Source: seed 3, q=256, 256 kimg, RAdam step 1990.
- Training-state SHA256:
  `fbda746805e6614319b96653563757f9e48670339e8f275f018194ebe19c9575`.
- Snapshot SHA256:
  `09a41e1e7c03dcdf5ffb93bb68687390278b4b190183dfff92bacc1bf79738d9`.
- Four frozen batch IDs: `2026082601..2026082604`.
- Eight frozen projection seeds: `2026082611..2026082618`.
- Arms: A native baseline; B native g=1.10; C target-geometry-only;
  D loss-weighting-only.
- Production accumulation: batch 128 as 8 microbatches of 16.
- Determinism: deterministic PyTorch algorithms, deterministic cuDNN,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, and reduced-precision matmul
  reduction off.

## Predictor outcomes

| Predictor | Formal coverage | Gate outcome | Interpretation |
| --- | ---: | --- | --- |
| Squared-loss simplified operator | 128/128 | 128 source-preserving executions | Theory baseline only; not real ECT |
| Recompute-and-detach field JVP | 128/128 | 0/128 pass the 5% convergence gate | No converged classical field Jacobian at the frozen FP16 scale |
| Full algorithmic state JVP | 128/128 | 0/128 valid overall | AMP skip discontinuity invalidates a single smooth transition Jacobian |

The field estimator recomputed both online and target forwards at every
`theta +/- epsilon*u` point and detached the newly recomputed target inside
that point. It therefore does not make the cached-detached-loss Hessian error.
Nevertheless, the finest adjacent relative change had median `0.16658` and
range `0.09345..0.56246`; no cell passed the frozen 5% threshold.

For the full algorithmic map, 12/128 cells passed the numerical adjacent-change
test in isolation, but 0/128 satisfied AMP plus/minus pairing across the full
epsilon sweep and 0/128 remained in one AMP regime across that sweep. The
discrete optimizer/scaler state pairing itself passed in 128/128 cells, and
all 128 source states were preserved. The failure is therefore an observed
algorithmic discontinuity, not optimizer-state pollution.

## Matched continuation

All four branches completed 64 counterfactual steps from clones of the same
complete state. Receipts exist at horizons `1, 4, 16, 64` for:

- model and EMA parameter projections;
- validation output and residual profile;
- fixed-latent output features;
- RAdam moment summaries;
- every-step raw gradient and update norms.

All 256 arm-steps had `step_skipped=0`; cross-arm AMP skip behavior was
identical. The matched receipt and source/file preservation gates passed.
These nonlinear trajectories remain valid even though a single smooth
algorithmic Jacobian is not licensed.

## Completeness and claim boundary

- Field receipts: 128 unique cells, no missing/extra/duplicate cells.
- Algorithmic receipts: 128 unique cells, no missing/extra/duplicate cells.
- Raw tensor pairs: 256 files, 342,534,144,896 bytes.
- Raw-tensor checksum entries: 256 unique SHA256 records.
- Test suite: 209 passed, 4 skipped.

This is a one-state, one-training-seed local audit. The four minibatches and
eight directions are paired repeated measurements, not independent training
replicates. No FID/KID or new schedule-family result was produced. The audit
does not establish a population-level absence of differentiability, a quality
effect, or a causal mechanism beyond this frozen state and numerical path.

## Artifacts

- Formal compact summary:
  `analysis/operator_clock_gate/results/raw_receipts/formal-20260826/results/formal_summary.json`.
- Matched receipt:
  `analysis/operator_clock_gate/results/raw_receipts/formal-20260826/results/matched/matched_micro_rollout.json`.
- Pre-formal calibration and retained failures:
  `analysis/operator_clock_gate/results/raw_receipts/formal-20260826/source/analysis/operator_clock_gate/CALIBRATION.md`.
- Remote raw-tensor location and checksums:
  `analysis/operator_clock_gate/results/raw_receipts/formal-20260826/REMOTE_STORAGE.md`.

The large `.pt` tensors remain on the Matpool execution node pending optional
durable migration to the ECT data server. Compact receipts, manifests, logs,
and the complete checksum list are committed in Git.

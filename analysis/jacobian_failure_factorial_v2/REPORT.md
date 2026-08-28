# Calibrated Jacobian failure factorial v2

**GO at the production-transition level; HOLD for attribution inside that transition.**

All 160 frozen cells completed with two valid manifests, 160 unique keys, matched
protocol hashes, and preserved source state and assets. The calibrated controls
separate cleanly from the production transition: B, C, and E pass in 32/32 cells,
whereas D fails in 32/32 cells. A passes in 30/32 cells; the two failures are the
same marginal batch/direction condition repeated in arms A and D, with relative
change 0.0517766 against the frozen 0.05 threshold.

| Regime | PASS | FAIL_CLOSED | Relative change, min / mean / max |
|---|---:|---:|---:|
| A: squared-GN FP32 | 30 | 2 | 0.0038 / 0.0187 / 0.0518 |
| B: real-loss GN FP32 | 32 | 0 | 0.0075 / 0.0247 / 0.0450 |
| C: full recompute-detach FP32 | 32 | 0 | 0.0029 / 0.0055 / 0.0103 |
| D: parameter-partial production transition | 0 | 32 | 0.5161 / 0.9399 / 1.3320 |
| E: pseudo-Huber FP32 field | 32 | 0 | 0.0035 / 0.0073 / 0.0149 |

The D failures are not process crashes or mismatched central-difference branches.
All D cells are finite, preserve the source state, pair AMP behavior across the
positive and negative branches, remain in one AMP regime across the sweep, and
pair the tracked discrete state. Their finest-scale relative changes remain
large (0.516--1.332; mean 0.940), while the complete FP32 field C remains below
0.011 in every cell.

## Interpretation

At this checkpoint, the smooth FP32 objective fields admit stable local
linearizations across the frozen factorial. The local parameter-to-augmented-state
production transition does not exhibit a numerically stable parameter-partial
Jacobian at the calibrated scales. This establishes separation at the transition
level. Regime D perturbs network parameters while holding the incoming optimizer,
EMA, scaler, buffers, and discrete state fixed; the output transition jointly
contains autocast/FP16, RAdam, EMA, and scaler updates. The factorial therefore
does not assign the instability to one internal component or characterize the
complete augmented-state derivative.

The result supports the bounded training-dynamics statement that instantaneous
objective structure need not survive the production optimizer transition. It
does not connect the Jacobian diagnostic to FID or identify an optimizer-mediated
quality mechanism.

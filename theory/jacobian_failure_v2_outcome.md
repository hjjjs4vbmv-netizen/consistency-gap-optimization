# Jacobian failure factorial v2 outcome

The calibrated formal audit gives **GO at the production-transition level and
HOLD for attribution inside that transition**.

The original correctness smoke on the coarse epsilon grid did not satisfy Gate
0. A separately frozen calibration against an autograd reference located a
stable squared-GN plateau and fixed the v2 epsilon grid at
`[0.00390625, 0.001953125, 0.0009765625]` before the formal matrix was opened.
The formal audit then completed all 160 cells with two complete manifests,
unique expected keys, matched protocol hashes, and preserved source state and
assets.

| Regime | PASS | FAIL_CLOSED | Finest relative change, min / mean / max |
|---|---:|---:|---:|
| A: squared-GN FP32 | 30 | 2 | 0.0038 / 0.0187 / 0.0518 |
| B: real-loss GN FP32 | 32 | 0 | 0.0075 / 0.0247 / 0.0450 |
| C: full recompute-detach FP32 | 32 | 0 | 0.0029 / 0.0055 / 0.0103 |
| D: production algorithmic transition | 0 | 32 | 0.5161 / 0.9399 / 1.3320 |
| E: pseudo-Huber FP32 field | 32 | 0 | 0.0035 / 0.0073 / 0.0149 |

The two A failures are one identical marginal batch/direction condition repeated
in arms A and D (0.0517766 against the frozen 0.05 gate). Every D cell is finite,
preserves the source state, pairs plus/minus AMP behavior, remains in one AMP
regime across the epsilon sweep, and pairs the tracked discrete state.

At this audited state and scale range, the FP32 objective-field controls admit
stable local linearizations while the complete production transition does not.
Attribution *within* that transition remains HOLD because D jointly contains the
internal FP16/autocast path, GradScaler, RAdam, and EMA. The audit does not
establish global nondifferentiability, optimizer causality, or a link to FID.

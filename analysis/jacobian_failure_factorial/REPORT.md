# Jacobian failure factorial report

**Historical status:** this coarse-grid correctness outcome is superseded for
field-level interpretation by the calibrated v2 audit. The calibrated full
recompute-and-detach FP32 field passes 32/32 formal cells. This report remains as
protocol provenance for the original stop decision, not as evidence that the
audited FP32 field lacks a local Jacobian.

**NO-GO: the squared-GN correctness baseline failed the frozen convergence gate; the formal factorial was not run by design.**

The preregistered correctness stage is complete. The 160-cell formal factorial was not run, as required by the frozen stop rule.

## Correctness-gate evidence

| Quantity | Value |
|---|---:|
| Finite | True |
| Source state preserved | True |
| Finest adjacent epsilon pair | 0.015 to 0.01 |
| Finest adjacent relative change | 0.106657124764 |
| Frozen relative-change tolerance | 0.05 |
| Finest-pair cosine | 0.994560831987 |

## Interpretation boundary

The baseline finite-difference field is finite and highly aligned across the finest pair, but it does not satisfy the frozen relative-change threshold. This audit therefore cannot localize the PR #87 failure to loss non-smoothness, network curvature, FP16/AMP, or the production transition.

This result is not evidence that the training process is globally non-differentiable. A new, separately frozen harness-calibration protocol is required before any source-localization factorial.

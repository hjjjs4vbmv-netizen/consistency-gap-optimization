# Claims licensed by the production-transition separation audit

Status: claim map for merged PR #89 at squash commit `d0229e3`

## 1. Audited mathematical objects

Regime C evaluates central finite differences of the full one-sided
recompute-and-detach FP32 objective-gradient field with target outputs freshly
recomputed and detached at each perturbed parameter state.

Regime D perturbs network parameters while holding the incoming optimizer
moments, EMA, scaler, floating buffers, and discrete state fixed. Its output is
the augmented post-transition state after internal FP16/autocast, GradScaler,
RAdam, and EMA updates. It therefore audits a local parameter-to-augmented-state
partial of the production transition, not the complete derivative
\(D_z\Phi\).

## 2. Empirical separation

Under the frozen calibrated epsilon grid, two audit minibatches, four directions,
and four factorial arms:

- the full FP32 recompute-and-detach field passes `32/32` cells, with maximum
  finest-scale adjacent relative change `0.0103`;
- the parameter-partial production transition fails closed in `32/32` cells,
  with finest-scale adjacent relative change `0.516--1.332`;
- all production cells remain finite, preserve the source state, pair the
  positive and negative AMP branches, remain in one AMP regime across the
  epsilon sweep, and pair the tracked discrete state.

The old coarse-grid field-level failure is superseded by the calibrated result.
It identifies a finite-difference scale failure in the earlier field audit, not
an absence of local linear structure in the evaluated FP32 field.

## 3. Main-text claim

The strongest licensed sentence is:

> At the audited state and calibrated scales, the full one-sided FP32 objective
> field admits a stable local finite-difference linearization, whereas the local
> parameter-to-augmented-state production transition does not.

The corresponding interpretation is:

> Exact instantaneous objective structure can survive into the smooth FP32
> objective field without surviving the stateful numerical optimizer transition
> at the same audited scales.

This is a numerical local-separation result. A finite collection of directions
and scales does not prove nonexistence of a classical derivative at the limiting
point.

## 4. Internal attribution

Regime D jointly contains internal FP16/autocast arithmetic, gradient scaling,
RAdam moment and parameter updates, EMA, and serialization/state-transition
logic. The current factorial localizes the instability to this combined
production layer. Attribution to any one internal component remains `HOLD`.

A component-level claim requires a new frozen intervention that changes one
internal layer while preserving the same parameter directions, states,
minibatches, epsilon grid, and decision rule.

## 5. Connection to carryover-corrected dynamics

The production separation explains why an objective-level factorization does
not determine a trained trajectory. Once two trajectories differ, persistent
state blocks contain both mechanical carryover and state-dependent update
feedback. The exact carryover-corrected recursion separates these terms:

\[
\Delta x_{k+1}
=b_k^x+C_k\Delta x_k+\widetilde R_k^x.
\]

Merged PR #89 v2 measures exact uncorrected forcing/feedback closure over one
matched 64-step replay from a frozen source state. For every state block and
observable it records the pre-transition separation \(\Delta_k\), propagation
gain, and alignment with the incoming separation. For \(\theta\), EMA, and
RAdam \(m\) and \(v\), it additionally reports
\(\widetilde R_k^x=R_k^x-C_k\Delta x_k\) using the implemented block-specific
carryover rule, together with corrected norm ratios and directional
alignments.

These measurements separate mechanical retention from non-trivial incremental
feedback at the audited frozen state. The stronger expansion claim remains
withheld because no independent state replication establishes directionally
consistent expansion. Residual and fixed-latent feature diagnostics remain
mixed and do not close a residual, feature, FID, or time-to-quality mediation
claim.

## 6. Claim table

| Statement | Status |
|---|---|
| The calibrated FP32 objective field is locally linearizable in all audited C cells. | Supported |
| The production parameter partial separates sharply from the FP32 field on the audited grid. | Supported |
| Carryover-corrected incremental feedback is measured for \(\theta\), EMA, \(m\), and \(v\) in one frozen-state replay. | Supported at the audited state |
| Incremental feedback expands separation across independent states. | Withheld pending replication |
| PR #87 proved that the real FP32 field lacks a stable local Jacobian. | Retracted by v2 calibration |
| The complete augmented-state derivative \(D_z\Phi\) was audited. | Not evaluated |
| One of FP16, GradScaler, RAdam, or EMA is the isolated source. | Hold |
| The production transition is globally nondifferentiable. | Not established |
| The numerical separation explains arm-level FID. | Not established |
| CIFAR transition behavior explains ImageNet dynamics. | Not established |

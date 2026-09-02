# Jacobian failure source: frozen factorial decision tree

Status: frozen before new factorial results
Base evidence: PR #87 operator-clock audit at seed 3, q256, 256 kimg

## 1. Question

PR #87 found that 0/128 recompute-and-detach field cells passed the frozen 5%
adjacent finite-difference gate and that 0/128 full algorithmic cells stayed in
a valid paired AMP regime.  The present audit asks which modeling layer first
destroys a numerically stable local directional derivative.

The audit uses one common parameter direction convention across all regimes.
Regime D perturbs network parameters while holding the baseline optimizer, EMA,
and scaler coordinates fixed; it evaluates the resulting full state transition.
Thus D is a partial directional derivative of the algorithmic map with respect
to parameters, not a derivative along arbitrary optimizer-state directions.

The production path in this repository uses the network's internal FP16 path
and GradScaler.  It does not wrap the loss in `torch.autocast`; the audit follows
the implemented transition rather than an imagined autocast configuration.

## 2. Frozen regimes

### A. Squared-loss Gauss--Newton action

For direction \(u\), estimate

\[
J_i^\top(J_i-J_j)u
\]

with paired finite differences of network outputs and a VJP through the online
branch.  This removes residual-norm curvature and network second derivatives.
It is the harness correctness baseline, not the production ECT field.

### B. Real-loss Gauss--Newton action

Let \(r=f_i-f_j\) and
\(\rho_c(r)=\sqrt{\|r\|^2+c^2}-c\), with the production value \(c=0\).
Estimate

\[
J_i^\top H_{\rho_0}(r)(J_i-J_j)u.
\]

For nonzero \(r\),

\[
H_{\rho_0}(r)v
=
\frac{v}{\|r\|}
-
\frac{r\langle r,v\rangle}{\|r\|^3}.
\]

This preserves the local residual-norm geometry and target recomputation
direction while omitting derivatives of the online Jacobian itself.  It is not
the complete gradient-field derivative.

### C. Full recompute-and-detach field, forced FP32

At every \(\theta\pm\epsilon u\), rerun online and target forwards, detach the
fresh target inside that evaluation, and differentiate the actual \(c=0\) ECT
loss.  Force the network's FP32 path and use no GradScaler or optimizer step.
This includes network curvature and the exact one-sided stop-gradient semantics.

### D. Production algorithmic transition

At every \(\theta\pm\epsilon u\), clone the same complete state and execute the
implemented internal-FP16, GradScaler, RAdam, and EMA transition.  Record
plus/minus scales, overflow/skip decisions, optimizer-step signatures, and
whether the entire epsilon sweep remains in one discrete regime.

### E. Full FP32 field with fixed pseudo-Huber smoothing

Repeat C with the diagnostic value \(c=0.06\), frozen from the existing ECT
configuration family.  This changes the diagnostic objective and is used only
to test whether smoothing restores finite-difference stability.

## 3. Common matrix and gate

- Arms: A/B/C/D from the q256 target×denominator factorial.
- Audit minibatches: `2026082601`, `2026082602`.
- Direction seeds: `2026082611..2026082614`.
- Epsilon grid for every regime: `[0.03, 0.02, 0.015, 0.01]`.
- Convergence gate: finest adjacent relative change at most 0.05.
- Report at every epsilon: JVP norm, adjacent relative error, adjacent cosine,
  norm ratio, finiteness, and production discrete-regime metadata.

No arm, batch, direction, or epsilon may be removed after results.

## 4. Decision tree

### Gate 0: A fails the correctness smoke

Verdict: **NO-GO**.  Inspect direction construction, perturbation application,
RNG replay, graph construction, finite-difference scale, and state restoration.
No production differentiability conclusion is licensed.

### A passes; B fails

The first observed instability enters with residual-norm geometry.  If E passes,
smoothing restores local numerical identifiability under the audited field and
supports `loss-geometry source` for this state.  If E also fails, the source is
not isolated.

### A and B pass; C fails

The Gauss--Newton actions are stable while the complete FP32 field is not.  The
remaining difference is the derivative of the online Jacobian and the complete
recompute-and-detach nonlinear field.  Verdict: `network-curvature/full-field
source` at the audited state.

### C passes; D fails with plus/minus AMP mismatch

The FP32 field passes the local finite-difference stability criterion, while
the production transition crosses a discrete AMP regime.  Verdict:
`production discrete-transition source`.
This does not imply absence of local structure away from the boundary.

### C passes; D fails without an AMP mismatch

The failure lies in the combined internal-FP16/stateful-transition layer, but
this five-regime design does not separate quantization from optimizer-state
curvature.  Verdict: **HOLD** with that bounded localization.

### C fails; E passes

Fixed smoothing restores local finite-difference stability despite retaining
network curvature.  Verdict: `c=0 loss geometry materially contributes` at the
audited state.  No training-quality claim follows.

### Mixed outcomes across batches or directions

Verdict: **HOLD**.  Report the complete cell matrix and the heterogeneity; do not
select successful directions or average away failures.

### Stable A--E

The PR #87 failure is not reproduced under the reduced paired-parameter audit.
This localizes the earlier failure to the broader state-direction convention or
other removed production coordinates, but does not validate a global clock.

## 5. Paper placement

Main text may contain one bounded conclusion if a source is stable across both
batches and all four directions.  Cell-level convergence tables, pseudo-Huber
diagnostics, and AMP signatures belong in the appendix.  HOLD and NO-GO outcomes
remain appendix evidence and narrow the smooth-operator narrative.

The audit cannot support claims about FID, optimizer causality, improved training
from smoothing, population differentiability, or a universal schedule operator.

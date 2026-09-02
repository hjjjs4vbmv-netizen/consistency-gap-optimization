# Independent theory verdict for PR #89

Reviewed merged evidence: PR #89 at squash commit `d0229e3`

Repository status: **MERGED**

Independent theory verdict: **MERGE**. The repository followed the recommended
order and squash-merged PR #89 after PR #87.

## Basis for the verdict

### Exact algebra

The forcing--feedback identity is correct for declared vector-valued readouts:

\[
\delta_{k+1}^{\psi}=b_k^{\psi}+R_k^{\psi}.
\]

It is algebraic and remains valid without differentiability. The conditional
Lipschitz propagation bound is also correct under its stated common-chart and
shared-discrete-regime assumptions.

### Calibrated field/transition separation

The v2 audit corrects the earlier field-level interpretation. The full FP32
recompute-and-detach field passes `32/32` formal cells, while the local
parameter-to-augmented-state production transition fails closed in `32/32`
cells. The production result is bounded to the audited state, directions,
batches, and scales.

### Carryover-corrected state propagation

The merged PR uses `persistent_state_feedback_dominance` rather than
`trajectory_feedback_amplification`. This matches the evidence: a large
\(R/b\) ratio shows that accumulated state differences dominate current
common-state forcing. The v2 replay also records \(\Delta_k\), propagation
gain, carryover-corrected \(\widetilde R_k\), and corrected alignments for
\(\theta\), EMA, and RAdam \(m\) and \(v\). These diagnostics identify
non-trivial incremental feedback separately from declared mechanical
carryover at one frozen state.

Residual and fixed-latent feature diagnostics remain
`mixed_or_inconclusive`. PR #89 therefore stops before a quality-mediation
claim. Its expansion gate is also withheld because a second independent state
replication is unavailable.

### Artifact shape

The final tree retains compact summaries, CSVs, reports, manifests, correctness
receipts, and five regime exemplars. Full telemetry is stored in a SHA256-verified
release artifact. This is reviewable in squash-merge form.

## Claims admitted after merge

1. Changing the schedule creates measurable common-state forcing.
2. Finite-horizon separation combines repeated forcing and trajectory-dependent
   state propagation.
3. Persistent model and optimizer state retains schedule-induced differences.
4. The calibrated FP32 field and the production parameter partial exhibit a
   sharp local numerical separation.
5. Carryover-corrected diagnostics separate mechanical retention from
   incremental feedback for \(\theta\), EMA, \(m\), and \(v\) at the audited
   frozen state.
6. Exact objective-level identities are insufficient by themselves to
   determine finite-training outcomes.

## Claims requiring new evidence

1. Incremental feedback expands existing trajectory separation across states.
   Replicate the corrected directional diagnostics from a second independently
   frozen state and apply the expansion criteria in the carryover-corrected
   proposition.
2. A specific internal component causes the production-transition instability.
   Run a frozen component factorial.
3. State feedback mediates residual, feature, FID, or time-to-quality changes.
   Align corrected state diagnostics with seed- and budget-resolved outcomes.

## Claims outside the current evidence

- incremental feedback causes FID improvement;
- RAdam is the unique trajectory mechanism;
- the production transition is globally nondifferentiable;
- CIFAR FP16 transition behavior explains ImageNet dynamics;
- state-block norm ratios are causal contribution percentages.

The carryover-corrected recursion strengthens the mechanism vocabulary without
changing PR #89's empirical verdict. It is an appropriate follow-up theory
result rather than a prerequisite for the bounded claims already in PR #89.

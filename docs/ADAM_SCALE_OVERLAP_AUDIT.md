# Adam scale-invariance overlap audit

Date: 2026-08-17

## Audit target and decision rule

This audit compares Fernández-Hernández et al., *Why Adam Works Better with
β1 = β2: The Missing Gradient Scale Invariance Principle*, arXiv:2601.21739v2
(11 May 2026), against the narrow GFCT/ECT theorem and evidence claims in this
repository.

The decision rule is deliberately conservative:

- `COVERED` means the 2026 paper already states or demonstrates the broad
  mechanism, even if our notation, optimizer variant, or application differs.
- `PARTIAL OVERLAP` means novelty survives only after naming the additional
  conjunction that the 2026 paper does not study.
- `NOT COVERED` means the object is outside the 2026 paper's stated theory and
  experiments. It is not, by itself, a claim that no other prior work covers it.

## Claim-by-claim overlap matrix

| Candidate claim | Fernández-Hernández et al. (2026) | Novelty decision for this paper | Required manuscript wording |
|---|---|---|---|
| Unequal first/second-moment memories create gradient-scale sensitivity | `COVERED`. Theorem 3.3 isolates the explicit scale-lag term `(τ2-τ1)δ(t)`; Section 3.4 interprets the unequal-memory case as first-order sensitivity. Appendix B also measures a large off-diagonal lag budget on stored real probe gradients. | **No broad novelty claim.** | Cite the 2026 paper as the general Adam-scale mechanism; present our unequal-memory observation only as an ECT/RAdam instantiation. |
| `β1 = β2` removes the explicit first-order scale-lag channel | `COVERED`. Corollary 3.6 states first-order gradient scale invariance iff `τ1=τ2`, corresponding to `β1=β2`. | **No novelty claim.** | Do not state or imply that GFCT first discovered the balanced-memory cancellation. |
| An ECT discretization-gap intervention produces a near-scalar real gradient perturbation | `NOT COVERED`. The paper studies synthetic rescaling and Adam/AdamW training diagnostics, not consistency training, ECT gaps, or a measured gap-induced scalar quotient/residual. | **Novel within the bounded comparison.** | Say that we *measure* near-scalarity for the paired ECT gap intervention at specified checkpoints/minibatches; do not universalize it to all gaps, models, or training stages. |
| Exact discrete finite-history characterization of ECT-conditioned RAdam updates | `PARTIAL OVERLAP`. Appendix B already implements numerically exact discrete Adam moment decompositions on a stored real probe-gradient sequence, so “exact discrete replay” is not new. It does not analyze RAdam's unrectified/rectified branch, the support-aware coordinate history gauge, off-support energy, or an ECT gap intervention. | **Defensible only in the narrowed form.** | Claim an **ECT-conditioned, rectified-RAdam, support-aware finite-history gauge identity/diagnostic**, with explicit assumptions (`eps=0`, no weight decay, shared step index) and separate treatment of the early unrectified branch. Never claim the general idea of exact discrete moment replay. |
| Real `(θ,m,v,step)` same-state counterfactual replay | `PARTIAL OVERLAP`. Definition 3.1 already fixes the internal state when rescaling the current gradient, and Appendix B replays a common stored probe-gradient trajectory across a `β1,β2` grid. However, that replay uses held-out probe gradients and does not fork different ECT gaps from the same nonzero saved training parameter/optimizer state. | **Defensible only as a conjunction.** | Claim a **paired ECT-gap intervention from the same nontrivial saved RAdam state**, with the same stochastic audit minibatch and explicit state components. Do not claim novelty for fixed-state scale tests or gradient-sequence replay in general. |
| Endpoint FID/KID boundary for the mechanism | `NOT COVERED`. The paper evaluates update-norm smoothness and task performance, not consistency-model FID/KID or the boundary between optimizer-update evidence and generation-quality evidence. | **Study-specific contribution, not a theorem.** | State that the optimizer diagnostic does not itself imply FID/KID; report endpoint metrics as a separate empirical boundary. Do not call the finite-history lemma an FID/KID theorem or a causal mediation proof. |

## Source-grounded comparison

The 2026 paper's strongest overlapping result is explicit: Theorem 3.3 writes
the continuous-time normalized Adam update as a sign term plus
`(τ2-τ1)δ(t)` and a remainder, and Corollary 3.6 identifies `β1=β2` as the
balanced case (paper pp. 5-6, Sections 3.3-3.4). That fully occupies the broad
claims “unequal moments cause first-order scale sensitivity” and “equal moments
cancel it.”

The overlap is stronger than a main-text-only reading suggests. Appendix B
stores a `7040 × 24458` CIFAR-10 probe-gradient matrix and replays a `3 × 3`
Adam memory grid on the common sequence. It reports numerically exact discrete
moment identities and a theorem-level `R=S+L+E` budget (paper pp. 23-24).
Accordingly, our paper must not advertise “exact discrete EMA replay” by itself
as new.

The remaining distinction is the intervention and state being conditioned on.
The 2026 probe gradients come from a fixed held-out batch, are not used for the
training update, and are replayed to isolate EMA parameters (paper p. 23 and
Appendix D, p. 31). Our narrow object is instead a paired consistency-gap fork
from the same saved, nonzero RAdam training state, with an ECT-specific gap
intervention and a support-aware update quotient/residual. The 2026 paper also
analyzes Adam rather than RAdam: it does not characterize RAdam's early
unrectified branch or its later rectified branch.

Two caveats from the 2026 paper should be carried into our Related Work and
limitations. First, bias correction is treated there as a finite-time,
non-autonomous perturbation of the continuous flow, so its main theorem is not
an exact finite-time discrete characterization near initialization (Remark
A.4, pp. 21-22). Second, nonzero `ε` introduces a zeroth-order breaking of exact
scale invariance when it is non-negligible relative to coordinate gradient
magnitudes (Remark A.5, p. 22). These points strengthen, rather than weaken,
the need to state our own RAdam assumptions and measure real-state residuals.

## Reviewer answer

> Beyond Fernández-Hernández et al. (2026), who establish the general Adam
> scale-lag mechanism and the first-order `β1=β2` cancellation, we instantiate
> and test a narrower ECT-specific question: whether a discretization-gap-induced
> near-scalar gradient change remains scalar-equivalent after conditioning on
> the same nontrivial discrete RAdam `(θ,m,v,step)` state, using a support-aware
> finite-history gauge and separating that optimizer evidence from the endpoint
> FID/KID boundary.

## Claim guardrails

Allowed:

- “The 2026 paper supplies the general Adam gradient-scale framework; our work
  studies an ECT/RAdam intervention not analyzed there.”
- “Our exact statement is support-aware and conditional on the idealized
  rectified-RAdam assumptions.”
- “The real-state counterfactual is paired at the saved state; it is not a
  bitwise continuation of the uninterrupted training trajectory.”
- “Endpoint FID/KID is separate empirical evidence, not implied by the update
  theorem.”

Forbidden:

- “We are the first to show unequal Adam moments cause scale sensitivity.”
- “We are the first to prove `β1=β2` scale invariance.”
- “Prior work has no discrete or real-gradient replay.”
- “Near-scalar gradients imply scalar-equivalent RAdam updates.”
- “The finite-history optimizer result proves the FID/KID outcome.”

## Local evidence anchors

- General RAdam null/history-gauge theorem: `theory/radam_gap_equivalence.md`.
- Exact finite-history moment identity and first-order expansion:
  `theory/radam_moment_memory.md`.
- Same-state audit protocol: `docs/RADAM_STATEFUL_UPDATE_AUDIT_PROTOCOL.md`.
- Bounded prior-work audit and paired-intervention definition:
  `docs/gfct_novelty_audit_0807.md`.
- Evidence-chain boundary: `docs/GAP_EVIDENCE_ASSET_AUDIT_2026_08_14.md`.

Primary external source: Fernández-Hernández et al. (2026),
https://arxiv.org/abs/2601.21739 (v2).

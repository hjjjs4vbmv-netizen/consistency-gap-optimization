# Gap × Learning-Rate Matched Experiment Protocol

## Status

PREPARED — FORMAL TRAINING BLOCKED UNTIL THE FRESH-STATE RADAM UPDATE AUDIT PASSES.

## Scientific question

This experiment tests whether the finite-budget quality effect of global gap
scaling can be explained by an initial-state RAdam learning-rate rescaling.

All formal trajectories start from the same pretrained EDM network and fresh
optimizer state:

\[
m_0=0,\qquad v_0=0,\qquad \mathrm{step}=0.
\]

They also share the same fresh GradScaler state, training seed, minibatch
order, timestep/noise/dropout randomness, dataset, source commit and budget.

## Frozen arms

| Arm | Gap | RAdam learning rate |
|---|---:|---:|
| A | 1.0 | 1.0e-4 |
| B | 1.3 | 1.0e-4 |
| C | 1.3 | c0_star × 1.0e-4 |

Arm C is an `initial-state one-step RAdam-update matched control`.

It is not an entire-trajectory optimizer-matched control.

## Update-matching definition

At the common fresh initialization, Role D must clone the complete state and
perform virtual, non-committing RAdam steps using completely paired inputs:

\[
\Delta\theta_{1.0},\qquad \Delta\theta_{1.3}.
\]

The learning-rate multiplier is defined directly as

\[
c_0^\star =
\frac{
\langle\Delta\theta_{1.3},\Delta\theta_{1.0}\rangle
}{
\|\Delta\theta_{1.3}\|^2
}.
\]

The matched residual is

\[
R_{\mathrm{update}} =
\frac{
\|c_0^\star\Delta\theta_{1.3}-\Delta\theta_{1.0}\|
}{
\|\Delta\theta_{1.0}\|
}.
\]

The audit must also report update cosine, both update norms, layerwise
multipliers and residuals, AMP unscale verification, and pre/post hashes of
parameters, optimizer state and GradScaler state.

The virtual diagnostic must not modify the formal training initialization.

## Prior raw-gradient evidence

The existing q128 256-kimg EMA result

\[
a_{1.3}^\star \approx 0.7700
\]

is retained only as FP32 raw mean-gradient geometry evidence. It is not a
fresh-state RAdam update fit and must not determine Arm C's learning rate.

## Checkpoint contract

Every formal trajectory must retain numbered network snapshots and training
states near 32, 64, 128 and 256 kimg. The exact `processed_kimg` in
`train_summary.csv` is authoritative. A `latest` checkpoint alone is invalid.

At Arm A checkpoints, the same virtual update audit will estimate

\[
c_K^\star,\qquad K\in\{32,64,128,256\},
\]

without adding training trajectories. This measures drift of the initial
one-step match.

## Evaluation contract

The primary endpoint is NFE=1 with identical evaluation seeds:

- FID-5k;
- KID-5k.

NFE=2 may be appended from retained checkpoints without retraining. Additional
training seeds, gap values and FID-50k are outside the frozen scope.

## Launch gate

Formal training may start only after:

1. the fresh network, RAdam and GradScaler initialization is verified;
2. the real repository ECT loss is used;
3. gap inputs are completely paired;
4. gradients are correctly AMP-unscaled;
5. virtual RAdam steps leave the source state unchanged;
6. `c0_star` and update residual diagnostics are finite;
7. collaborator returns PASS;
8. Arm C learning rate is resolved as `c0_star × 1.0e-4`;
9. the numbered-checkpoint smoke test passes.

## Claim boundary

A difference between Arms A and C after initialization-level matching can
support a non-scalar or trajectory-dependent deep-optimizer effect. It does
not by itself prove that gap geometry is the unique cause. If `cK_star`
changes materially over training, the result must be described as
initialization-matched but trajectory-divergent.

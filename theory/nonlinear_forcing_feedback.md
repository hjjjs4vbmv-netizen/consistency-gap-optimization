# Nonlinear schedule forcing and trajectory feedback

Status: theory updated after the calibrated Jacobian-source factorial
Scope: one-step coupled training transitions and finite-horizon propagation

## 1. Algorithmic state and coordinate scope

Let the complete training state be

\[
z_k=(x_k,d_k)\in\mathcal Z,
\]

where the continuous component \(x_k\) contains network parameters, floating
optimizer moments, floating model buffers, EMA parameters and buffers, and the
GradScaler scale.  The discrete signature \(d_k\) contains optimizer step
counters, scaler growth trackers, overflow/skip decisions, schedule stage, and
any other integer or Boolean state that changes the next transition.

For schedule \(S\in\{A,B\}\), one coupled training step is

\[
z_{k+1}^{S}=\Phi_{S,k}(z_k^S,\xi_k),
\]

where \(\xi_k\) contains the shared minibatch, diffusion noise, dropout mask,
augmentation draw, and all other exogenous randomness.  Sharing \(\xi_k\) is a
coupling choice: it isolates the schedule intervention from stochastic-input
differences.

The state space is not globally a vector space because \(d_k\) is discrete.
Let \(\psi:\mathcal Z\rightarrow V\) be a declared vector-valued readout.  It
may select model parameters, the continuous coordinates of a fixed discrete
regime, or a downstream output feature.  All differences below are differences
in \(V\), not undefined subtraction of arbitrary algorithmic states.

## 2. Exact forcing--feedback identity

Define the readout separation

\[
\delta_{k+1}^{\psi}
=
\psi\!\left(\Phi_{B,k}(z_k^B,\xi_k)\right)
-
\psi\!\left(\Phi_{A,k}(z_k^A,\xi_k)\right).
\]

The common-state schedule forcing is

\[
b_k^{\psi}
=
\psi\!\left(\Phi_{B,k}(z_k^A,\xi_k)\right)
-
\psi\!\left(\Phi_{A,k}(z_k^A,\xi_k)\right),
\]

and the schedule-\(B\) trajectory feedback is

\[
R_k^{\psi}
=
\psi\!\left(\Phi_{B,k}(z_k^B,\xi_k)\right)
-
\psi\!\left(\Phi_{B,k}(z_k^A,\xi_k)\right).
\]

Adding and subtracting
\(\psi(\Phi_{B,k}(z_k^A,\xi_k))\) gives the exact identity

\[
\boxed{\delta_{k+1}^{\psi}=b_k^{\psi}+R_k^{\psi}.}
\]

This identity is algebraic.  It does not require differentiability, a small
perturbation, a linear optimizer, continuous AMP behavior, or a fixed global
schedule operator.  Its content is operational: both terms are counterfactual
one-step differences that can be measured from cloned states under paired
randomness.

When \(\psi\) is a coordinate chart for the complete continuous state and the
two branches share a discrete signature, we write
\(\Delta_k=\psi(z_k^B)-\psi(z_k^A)\) and obtain

\[
\Delta_{k+1}=b_k+R_k.
\]

If the branches differ in optimizer-step, overflow, scaler-growth, or schedule
stage signatures, the exact readout identity remains valid, but a smooth
continuous-state recurrence is not asserted.  The discrete mismatch is then a
reported outcome of the transition.

## 3. Local propagation bound

Assume that both states lie in a region with a common discrete signature and
that \(\psi\) is an injective continuous-state chart.  Define the charted map

\[
F_{B,k}(x;\xi_k)
=
\psi\!\left(
\Phi_{B,k}(\psi^{-1}(x),\xi_k)
\right).
\]

If, on the segment containing the two coupled states,

\[
\|F_{B,k}(x;\xi_k)-F_{B,k}(x';\xi_k)\|
\le L_k\|x-x'\|,
\]

then

\[
\|R_k\|\le L_k\|\Delta_k\|
\]

and therefore

\[
\boxed{
\|\Delta_{k+1}\|
\le
\|b_k\|+L_k\|\Delta_k\|.
}
\]

For \(\Delta_0=0\), recursive substitution yields

\[
\boxed{
\|\Delta_T\|
\le
\sum_{\tau=0}^{T-1}
\left(
\prod_{s=\tau+1}^{T-1}L_s
\right)
\|b_\tau\|,
}
\]

with an empty product equal to one.  This is a worst-case propagation bound.
It is not an observed gain: it discards direction, cancellation, stochastic
dependence, and the looseness of the local Lipschitz constants.

## 4. Directional interaction

The propagation diagnostic must use the separation that exists before the
transition,

\[
\Delta_k^\psi=\psi(z_k^B)-\psi(z_k^A),
\]

not the post-transition separation.  We therefore report

\[
G_k^\psi=
\frac{\|R_k^\psi\|}
{\max(\|\Delta_k^\psi\|,\epsilon)},
\qquad
a_k^\psi=\cos(R_k^\psi,\Delta_k^\psi).
\]

`G` is inseparable from its alignment: `G≈1` and `a≈1` indicates state
persistence, `G<1` indicates contractive propagation, and `G>1` with `a≈1`
is possible same-direction expansion.  Low alignment instead indicates
rotation or more complex deformation.  None of these regimes identifies a
scientifically preferable schedule.

The exact identity gives

\[
\|\Delta_{k+1}\|^2
=
\|b_k\|^2+\|R_k\|^2+2\langle b_k,R_k\rangle.
\]

The pair

\[
\frac{\|R_k\|}{\max(\|b_k\|,\epsilon)},
\qquad
\cos(b_k,R_k)
=
\frac{\langle b_k,R_k\rangle}
{\max(\|b_k\|\|R_k\|,\epsilon)}
\]

distinguishes three descriptive regimes:

1. **Forcing dominated:** \(\|R_k\|\ll\|b_k\|\).
2. **Reinforcing feedback:** \(\|R_k\|\) is non-negligible and
   \(\langle b_k,R_k\rangle>0\).
3. **Cancellation:** both norms are non-negligible and
   \(\langle b_k,R_k\rangle<0\).

The ratio \(\|b_k\|/\|\Delta_{k+1}\|\) is not a contribution percentage.
Vector terms can reinforce or cancel, and both terms depend on the chosen state
readout and norm.

### 4.1 Carryover-corrected incremental feedback

Raw \(R_k\) includes the part of the incoming separation retained by a
persistent state coordinate.  For parameters,

\[
\widetilde R_k^\theta=R_k^\theta-\Delta_k^\theta.
\]

For RAdam moments, the implemented parameter-group retentions give

\[
\widetilde R_k^m=R_k^m-\beta_1\Delta_k^m,
\qquad
\widetilde R_k^v=R_k^v-\beta_2\Delta_k^v.
\]

The implementation reads each parameter group's actual `betas`; it does not
assume that all groups share one scalar.  EMA is also derived from the actual
transition.  Its parameter update is

\[
e_{k+1}=\beta_{\rm EMA}e_k+(1-\beta_{\rm EMA})\theta_{k+1},
\]

while the audited transition leaves EMA buffers unchanged.  The EMA
carryover-only counterfactual map consequently applies \(\beta_{\rm EMA}\) to
parameter separation and identity retention to buffer separation.  This map,
including its observed retention values and implementation source, is recorded
in the output schema.

For every corrected state block the audit reports

\[
\frac{\|\widetilde R_k\|}{\max(\|\Delta_k\|,\epsilon)},
\qquad
\frac{\|\widetilde R_k\|}{\max(\|b_k\|,\epsilon)},
\]

together with alignment against \(\Delta_k\), \(b_k\), and raw \(R_k\).
This measures the new effect of trajectory separation on the current optimizer
update after declared carryover is removed.

A stronger causal expansion claim is withheld unless all of the following are
observed: \(G_k>1\), strong alignment of \(R_k\) with \(\Delta_k\),
non-trivial \(\widetilde R_k\), and directionally consistent evidence in a
second independent state replication.  Otherwise the permitted descriptions
are propagation and persistence.

## 5. Clock equivalence

### 5.1 Continuous-time condition

For autonomous systems

\[
\dot z=G_A(z),\qquad \dot z=G_B(z),
\]

if \(G_B(z)=sG_A(z)\) for a constant \(s>0\), uniqueness gives
\(z_B(t)=z_A(st)\) from the same initial state.  More generally,
\(G_B(z)=\alpha(z)G_A(z)\) with \(\alpha(z)>0\) preserves oriented integral
curves but induces a state-dependent time reparameterization.

### 5.2 Discrete-time condition

Two discrete maps have an exact clock interpretation when they are time slices
of a common semiflow: \(\Phi_A=\varphi_h\) and
\(\Phi_B=\varphi_{\tilde h}\).  The special relation
\(\Phi_B=\Phi_A^{\circ n}\) is meaningful only for an integer \(n\) unless a
continuous semigroup embedding has been supplied.

For one-step increments

\[
v_A(z)=\Phi_A(z)-z,
\qquad
v_B(z)=\Phi_B(z)-z,
\]

the coefficient and transverse residual

\[
a^*(z)=\frac{\langle v_B(z),v_A(z)\rangle}{\|v_A(z)\|^2},
\qquad
v_B^\perp(z)=v_B(z)-a^*(z)v_A(z)
\]

provide a local clock-like diagnostic.  Small \(v_B^\perp\) is local evidence
of aligned increments; it is not a global time-reparameterization theorem.

### 5.3 Failure boundaries

Clock equivalence can fail through:

- transverse common-state forcing;
- path-dependent optimizer moments and preconditioning;
- AMP overflow, skipped steps, scaler updates, or other discrete regime changes;
- unpaired stochastic inputs;
- schedule-stage changes;
- readout-dependent deformation, including different NFE sampling paths.

ReLU activation changes and the \(c=0\) residual norm can also invalidate a
classical Jacobian at specific points.  These failures affect smooth
linearization, not the exact forcing--feedback identity.

In the calibrated audit, the full recompute-and-detach FP32 field passes all 32
formal cells. The earlier coarse-grid field failure is therefore a harness-scale
failure, not evidence that the audited FP32 field lacks a local linearization.
The production audit perturbs parameters while holding incoming optimizer, EMA,
scaler, buffer, and discrete coordinates fixed; it measures a local
parameter-to-augmented-state transition rather than the full derivative
\(D_z\Phi\).

## 6. Claim boundary

The theory establishes an exact, measurable decomposition of paired one-step
schedule effects and a conditional worst-case propagation bound.  It licenses
the statement:

> Schedule changes generate common-state forcing.  Finite-horizon separation
> combines repeated forcing, state-dependent feedback, and their directional
> interaction. At the audited state, the smooth FP32 objective field admits a
> stable local linearization, while the parameter-partial production transition
> does not at the calibrated scales.

It does not reduce all schedule effects to the spectrum of one global smooth
operator.

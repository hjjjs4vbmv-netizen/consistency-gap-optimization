# Carryover-corrected forcing and feedback

Status: exact blockwise result

Scope: continuous persistent state blocks with a declared, shared linear
carryover operator at one coupled training step

## 1. Block update model

Let \(S\in\{A,B\}\) index two schedules, and let \(z_k^S\) be the complete
algorithmic state before step \(k\). Both trajectories consume the same frozen
exogenous input \(\xi_k\). For a continuous state block
\(x_k^S=\pi_x(z_k^S)\in V_x\), assume that the implemented transition admits

\[
x_{k+1}^S
=
C_k x_k^S+U_{S,k}(z_k^S,\xi_k).
\tag{1}
\]

Here \(C_k:V_x\to V_x\) is a known linear carryover operator shared by the
counterfactual evaluations in this step. It may depend on the step index,
frozen optimizer hyperparameters, and a shared discrete regime. It cannot vary
between \((A,z_k^A)\), \((B,z_k^A)\), and \((B,z_k^B)\) without adding a
carryover-mismatch term.

The increment \(U_{S,k}\) contains the part newly written during the current
transition. It may be nonlinear in the full state and may include gradients,
preconditioning, weight decay, clipping, numerical precision, and schedule
dependence.

The correction is exact relative to the declared block and operator. If another
persistent coordinate enters the update linearly, it should be included in a
joint block with the corresponding off-diagonal entry in \(C_k\). Otherwise
\(\widetilde R_k^x\) is corrected only for the chosen block's self-carryover and
can still contain mechanical carryover imported from another block.

## 2. Proposition: exact carryover-corrected recursion

Define the block separation

\[
\Delta x_k=x_k^B-x_k^A,
\]

the common-state schedule forcing

\[
b_k^x
=
U_{B,k}(z_k^A,\xi_k)-U_{A,k}(z_k^A,\xi_k),
\tag{2}
\]

and the state-dependent incremental feedback under schedule \(B\)

\[
\widetilde R_k^x
=
U_{B,k}(z_k^B,\xi_k)-U_{B,k}(z_k^A,\xi_k).
\tag{3}
\]

Then

\[
\boxed{
\Delta x_{k+1}
=
b_k^x+C_k\Delta x_k+\widetilde R_k^x.
}
\tag{4}
\]

### Proof

Subtract (1) for schedules (A) and (B), and add and subtract
\(U_{B,k}(z_k^A,\xi_k)\):

\[
\begin{aligned}
\Delta x_{k+1}
&=C_k(x_k^B-x_k^A)
  +U_{B,k}(z_k^B,\xi_k)-U_{A,k}(z_k^A,\xi_k)\\
&=C_k\Delta x_k
  +\left[U_{B,k}(z_k^A,\xi_k)-U_{A,k}(z_k^A,\xi_k)\right]\\
&\quad
  +\left[U_{B,k}(z_k^B,\xi_k)-U_{B,k}(z_k^A,\xi_k)\right].
\end{aligned}
\]

The bracketed terms are (2) and (3). No differentiability or small-perturbation
assumption is used. \(\square\)

Equation (4) separates three distinct sources of next-step separation:

1. \(b_k^x\): schedule forcing at a common state;
2. \(C_k\Delta x_k\): mechanical retention of existing state differences;
3. \(\widetilde R_k^x\): a difference in newly written updates caused by the
   trajectories already occupying different states.

## 3. Relation to the uncorrected exact decomposition

The earlier trajectory-feedback term for block \(x\) is

\[
R_k^x
=
\pi_x\!\left(\Phi_{B,k}(z_k^B,\xi_k)\right)
-
\pi_x\!\left(\Phi_{B,k}(z_k^A,\xi_k)\right).
\]

Under (1), it satisfies the exact identity

\[
\boxed{
R_k^x=C_k\Delta x_k+\widetilde R_k^x,
\qquad
\widetilde R_k^x=R_k^x-C_k\Delta x_k.
}
\tag{5}
\]

Consequently, a large \(\|R_k^x\|/\|b_k^x\|\) can be produced by ordinary
state retention even when incremental feedback is small. The ratio remains a
history-dominance diagnostic; it is not an amplification factor.

## 4. Exact finite-horizon unrolling

For \(j>i\), define

\[
C_{j:i}=C_{j-1}C_{j-2}\cdots C_i,
\]

with \(C_{i:i}=I\). Repeated substitution in (4) gives

\[
\boxed{
\Delta x_T
=
C_{T:0}\Delta x_0
+
\sum_{\tau=0}^{T-1}
C_{T:\tau+1}
\left(b_\tau^x+\widetilde R_\tau^x\right).
}
\tag{6}
\]

For matched initialization, \(\Delta x_0=0\). Equation (6) is exact but not a
closed causal attribution: \(\widetilde R_\tau^x\) depends on the realized
coupled trajectories, and vector terms can reinforce or cancel.

## 5. Implemented persistent blocks

### Network parameters

Write the optimizer parameter increment as

\[
\theta_{k+1}^S=\theta_k^S+u_{S,k}(z_k^S,\xi_k).
\]

Then

\[
C_k^\theta=I,
\qquad
\widetilde R_k^\theta
=u_{B,k}(z_k^B,\xi_k)-u_{B,k}(z_k^A,\xi_k).
\]

All RAdam preconditioning, bias correction, clipping, precision effects, and
weight decay belong to \(u_{S,k}\).

### RAdam first and second moments

On an executed optimizer step with one parameter group,

\[
m_{k+1}=\beta_1m_k+(1-\beta_1)g_k,
\qquad
v_{k+1}=\beta_2v_k+(1-\beta_2)g_k^{\odot2}.
\]

Therefore

\[
C_k^m=\beta_1I,
\qquad
C_k^v=\beta_2I.
\]

The formal ECT configuration uses \(\beta_1=0.9\) and \(\beta_2=0.999\).
With a shared execution indicator \(a_k\in\{0,1\}\), where \(a_k=0\) denotes
an AMP-skipped optimizer step, the exact operators are

\[
C_k^m=\left[1-a_k(1-\beta_1)\right]I,
\qquad
C_k^v=\left[1-a_k(1-\beta_2)\right]I.
\tag{7}
\]

Different skip decisions across the paired evaluations violate the shared-
\(C_k\) premise and must be reported as a discrete-regime mismatch.

### EMA parameters

The implemented update is

\[
e_{k+1}^S
=
\beta_{\mathrm{ema},k}e_k^S
+(1-\beta_{\mathrm{ema},k})\theta_{k+1}^S,
\]

where \(\beta_{\mathrm{ema},k}\) is either the configured decay or the actual
step-dependent half-life/ramp-up value. Thus

\[
C_k^{\mathrm{EMA}}=\beta_{\mathrm{ema},k}I,
\qquad
U_{S,k}^{\mathrm{EMA}}
=(1-\beta_{\mathrm{ema},k})\theta_{k+1}^S.
\tag{8}
\]

EMA buffers require separate block definitions. In the PR #89 audit transition,
EMA parameters are averaged and EMA buffers are mechanically retained, giving
\(C_k=I\) for those retained buffer coordinates.

For mechanism attribution, the preferred state is the joint
\((\theta,e)\) block. Substituting
\(\theta_{k+1}=\theta_k+u_{S,k}\) into (8) gives

\[
\begin{bmatrix}
\theta_{k+1}\\ e_{k+1}
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
I & 0\\
(1-\beta_{\mathrm{ema},k})I & \beta_{\mathrm{ema},k}I
\end{bmatrix}}_{C_k^{(\theta,\mathrm{EMA})}}
\begin{bmatrix}
\theta_k\\ e_k
\end{bmatrix}
+
\begin{bmatrix}
u_{S,k}\\
(1-\beta_{\mathrm{ema},k})u_{S,k}
\end{bmatrix}.
\tag{9}
\]

This block operator removes both EMA self-retention and the mechanical transfer
of the existing parameter difference into EMA. The EMA-only convention in (8)
is exact, but its incremental term still contains that cross-block parameter
history.

## 6. When does incremental feedback expand separation?

Let \(q_k=C_k\Delta x_k\). Incremental feedback expands the inherited block
separation precisely when

\[
\|q_k+\widetilde R_k^x\|>\|q_k\|,
\]

or equivalently

\[
2\langle q_k,\widetilde R_k^x\rangle
+\|\widetilde R_k^x\|^2>0.
\tag{10}
\]

Its effect on the complete next-step separation is a different comparison:

\[
\|b_k^x+q_k+\widetilde R_k^x\|
>
\|b_k^x+q_k\|.
\tag{11}
\]

Equations (10) and (11), together with
\(\cos(q_k,\widetilde R_k^x)\), are the appropriate expansion diagnostics.
Neither follows from \(\|R_k^x\|/\|b_k^x\|>1\).

## 7. Claim boundary

The exact result licenses:

> Schedule-induced differences are retained in persistent optimizer and model
> state, while state-dependent update feedback can be measured separately from
> mechanical carryover.

An expansion claim requires measurements satisfying (10) or (11) across a
declared set of blocks, steps, seeds, and schedules. The recursion itself does
not connect incremental feedback to FID, identify RAdam as the unique mechanism,
establish global differentiability of the production transition, or transfer a
CIFAR transition audit to ImageNet dynamics.

## 8. Reproducible algebra check

`tests/test_carryover_corrected_recursion_theory.py` verifies (4), the joint
parameter--EMA specialization (9), and the finite-horizon unrolling (6) over
200 deterministic generated cases each. It uses exact rational arithmetic, so
the checks introduce no floating-point tolerance. Run:

```text
python tests/test_carryover_corrected_recursion_theory.py
```

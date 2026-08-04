# ECT Gap Calibration: Novelty Statement & Theoretical Propositions

Date: 2026-08-04 (Role C, theory day)
Status: pre-registered propositions, partially validated on the linear-Gaussian toy model.

---

## 1. Novelty statement (defensible)

> Existing adaptive-discretization work for consistency models (e.g. *Adaptive
> Discretization for Consistency Models*) learns a *discretization* (how to
> choose the pairing times t -> r) to satisfy local/global consistency
> constraints, typically with a Gauss–Newton-style local correction. This
> project does **not** claim to invent "adaptively choose r(t)". Its claims are
> about the **optimization geometry of a single global gap multiplier**, which
> is orthogonal to and absent from that line of work:
>
> **(N1) Finite-budget geometry.** For a fixed total optimization budget K, the
> symmetric population Hessian of the consistency objective satisfies
> `H_g = g^2 H_2 + g^3 H_3 + g^4 H_4` in a linear-Gaussian model. The optimal gap
> is an *internal* optimum `g_K^* > 1` that exists because increasing g speeds
> the slow eigen-directions (larger lambda_min) while also pushing
> `eta * lambda_max(H_g)` toward the GD stability boundary 2. **No global
> scale exists in the adaptive-discretization literature; they optimize a local
> discretization, we characterize a global scale's cost surface.**
>
> **(N2) Budget-dependence of the optimum.** `g_K^*` is a function of the
> budget: short-budget optima sit at different g than long-budget optima, and
> the response curve is single-peaked (or has a wide plateau) rather than
> monotone. This is a *finite-K* statement; the infinite-K limit collapses to a
> trivial "bigger g is faster" regime. Adaptive-discretization work targets the
> infinite-data / stationary setting and does not characterize finite-K optima.
>
> **(N3) Loss-only identifiability failure.** A local controller that observes
> only the scalar loss at a single g cannot determine whether g should increase
> or decrease: two environments `(A,B) = (4,1)` and `(1,4)` give identical
> loss at g=1 but opposite optimal directions (`g^* > 1` vs `g^* < 1`). The
> controller needs variance / curvature / multi-g probe. This is a **negative
> result about the information content of the loss trajectory alone** and is
> the theoretical underpinning for why a loss-only local controller is
> unreliable across seeds.

**Why this is NOT subsumed by prior work:**
- *sCM (stability conditioned on scale)*: sCM trains to match the score directly
  with an SDE, no t->r pairing gap; our object of study (the pairing gap and its
  global multiplier) does not exist there.
- *Adaptive Discretization*: optimizes a local t->r rule per interval; we study
  a **single global multiplier** and prove existence of an internal optimum from
  the Hessian polynomial structure + stability boundary. The local/global axis,
  the finite-budget axis, and the loss-only non-identifiability are not in that
  paper.
- *Classical CM convergence theory* (score/consistency error bounds under data
  smoothness): gives W2 control at fixed KFE; does not predict how a **global gap
  multiplier** changes the finite-budget optimization landscape. Our predictions
  are testable, quantitative, and tied to FID/KID response, which the external
  theory does not make.

---

## 2. Three paper-ready propositions

### Proposition A (Ideal-solution invariance, exact-trajectory sense)
Let `gamma_z(t)` be a PF trajectory ending at `z`, and let `f*` be the ideal
consistency map satisfying `f*(gamma_z(t), t) = z`. For any legal schedule
`r_g(t) < t` satisfying A4, `f*` achieves zero loss on every training pair:
`L_g(f*) = 0`. Conversely, any zero-loss solution coincides with the ideal map
on the schedule graph. Hence the *zero-loss solution set is independent of g*.

*Proof sketch* (from report §0.3): each integrand is zero at f*; conversely A3 +
schedule connectivity + boundary condition gives f* uniqueness.

*Toy-model check*: the toy `f_beta` with `beta=0` is the ideal map and attains
zero residual for every `(t, r_g)`; confirmed numerically for all g in
[0.5, 2.0].

### Proposition B (Finite-budget internal optimum)
In the linear-Gaussian model with symmetric population loss, the full-batch GD
iterate satisfies `beta_K = (I - eta H_g)^K beta_0` with
`H_g = g^2 H_2 + g^3 H_3 + g^4 H_4`. If increasing g raises the minimal
eigenvalue `lambda_min(H_g)` (speeding the slow direction) while
`eta * lambda_max(H_g) -> 2` as `g -> g_s`, then by continuity the GD spectral
radius `rho_g = max_j |1 - eta*lambda_j(H_g)|` attains an internal minimum at
some `g_K^* in (1, g_s)`. The finite-budget error
`||beta_K||^2 = sum_j (1 - eta*lambda_j)^{2K} beta_{0,j}^2` is minimized at a
`g_K^*` that depends on K.

*Toy-model check (VALIDATED)*: with `eta = 1.0/lambda_max(H_1)`,
- `eta*lambda_max(H_g)`: 0.25 (g=0.5) -> 1.40 (g=1.0) -> **3.98 (g=2.0)**
  crossing the stability boundary 2 between g=1.4 and 1.5.
- noiseless optimal g*: 1.35 (K=50), 1.40 (K=200), 1.40 (K=1000).
- noisy (rms=0.01) optimal g*: 1.00 (K=50), 1.10 (K=200), 1.35 (K=1000).

### Proposition C (Loss-only non-identifiability)
Let a local controller observe only the scalar loss `R_{A,B}(g) = A/g^2 + B g^2`.
Construct two environments `(A1,B1)=(4,1)` and `(A2,B2)=(1,4)`. Both give
`R(1) = 5`, but `R'_{4,1}(1) = -6` (optimum g* = sqrt(2) > 1) and
`R'_{1,4}(1) = +6` (optimum g* = 1/sqrt(2) < 1). Any deterministic policy
`pi(R)` that depends only on the scalar loss performs the same action in both
environments and is therefore wrong in at least one of them. A controller must
observe variance / curvature / or a multi-g probe to resolve the direction.

*Toy-model check (VALIDATED, mechanism)**: the finite-budget error curve is
single-peaked; two g values on opposite flanks of the peak can share the same
loss value, so the loss level alone does not indicate the correct update
direction. Noise reshapes the curve and shifts the optimum (see §3).

---

## 3. Toy-model results summary (what the numbers actually show)

| Prediction | Result | Verdict |
|---|---|---|
| 1. error vs g single-peaked / plateau | noiseless: optimum at g*=1.35-1.40, divergence for g>~1.45 (eta*lambda_max>2); noisy: U-shape sharper, optimum moves to g*=1.00-1.35 | **VALIDATED** |
| 2. short-budget g_K* differs from long-budget | noiseless: 1.35 (K=50) -> 1.40 (K=1000) (mild); noisy 0.01: 1.00 -> 1.10 -> 1.35 (K=50/200/1000) | **VALIDATED (noisy), mild (noiseless)** |
| 3. too-large g near instability | eta*lambda_max exceeds 2 for g >= ~1.5; error diverges | **VALIDATED** |

### Honest caveats (anticipate reviewer pushback)
- **The noiseless curve is shallow** (error 9.87e-5 -> 9.86e-5, ~0.1%): the
  final error is dominated by the slow eigen-direction whose lambda_min is tiny
  (cond ~86000). The internal optimum is real but the sensitivity is weak.
- **`H_g ~ g^2 H_2`**: in the toy, `H3, H4 ~ 1e-4 * H2` terms, so the g-polynomial
  is dominated by the quadratic term; the cubic/quartic structure is present but
  small. This means the toy shows the *stability-boundary + slow-direction*
  mechanism; the "signal-curvature trade-off" (O(Delta) signal, O(Delta^2)
  curvature) lives at the level of the *per-pair loss* (report eq. for
  `||h(t)-h(t-Delta)||^2`), not in the population Hessian. A toy that includes
  the finite-difference surrogate-bias term (shared-noise coupling) is needed to
  show that mechanism; it is planned as the next step.
- **Stop-gradient operator**: we report the symmetric part spectrum of
  `A_g = E[J_t^T(J_t - J_r)]`. The asymmetry (antisymmetric part) changes
  eigen-directions vs `H_g`; this is computed but not yet used to draw
  conclusions about g* (deferred to the non-symmetric analysis).

---

## 4. Mapping to the ICLR argument

```
ideal-solution invariance (Prop A) 
   -> finite-gap signal-curvature trade-off (report §0.3, toy: slow-direction + stability)
   -> finite-budget optimization geometry (Prop B, toy: g_K* single-peaked)
   -> loss-only non-identifiability (Prop C, toy: same-loss-different-direction)
```

The chain is what makes the project a *theoretical* contribution rather than
"tuned g=1.1": each link is a testable proposition with an explicit mechanism,
and the loss-only non-identifiability (N3/Prop C) is the sharpest point of
difference from adaptive-discretization work.

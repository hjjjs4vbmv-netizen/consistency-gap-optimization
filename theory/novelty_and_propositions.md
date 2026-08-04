# ECT Gap Calibration: Novelty Statement & Theoretical Propositions (rev.2)

Date: 2026-08-04; revised 2026-08-05 after the PR #33 review (REQUEST_CHANGES).
Status: stop-gradient operator corrected (finite-difference verified);
LR-matched controls added; ADCM separation counterexample constructed.

---

## 1. Novelty statement (instantaneous vs finite-horizon)

> **ADCM optimizes a state-conditioned *instantaneous* consistency criterion.**
> We do NOT claim novelty from "global multiplier vs local discretization" —
> that distinction is insufficient, since ADCM already studies finite image
> budget, training efficiency, and convergence. The candidate novelty is:
>
> **Does the instantaneous criterion determine the finite-horizon optimum
> under a specified optimizer, remaining budget, parameter-space curvature,
> and gradient-noise covariance?**
>
> Our claim is that it does **not**: two settings with the *same* instantaneous
> criterion (same population curvature $H_g$) can have *different* finite-step
> optima $g_K^\star$ because their gradient-noise covariance $\Sigma_g$ differs.
> This is a gap-dependent finite-horizon geometry that an instantaneous-only
> method cannot resolve.

**What we explicitly do NOT claim anymore** (removed per review):
- ~~ADCM targets an infinite-data / stationary setting~~
- ~~no global scale exists in ADCM, therefore our problem is orthogonal~~
- ~~the U-shape establishes a gap-specific internal optimum~~

The honest scope, supported by the LR-matched controls (§3):
- Under a **fixed** learning rate, increasing $g$ raises the effective rate
  $\eta\lambda_{\max}(H_g)\propto g^2$ until the GD stability boundary; the
  observed "U-shape" is largely this effective-learning-rate effect.
- Under **LR-matched** $\eta$ (so $\eta\lambda_{\max}$ is constant across $g$),
  the U-shape **collapses** in the noiseless case and becomes **monotone** in
  the noisy case: the remaining $g$-dependence is the *noise-amplification*
  term $\eta\nu/(\lambda(2-\eta\lambda))$, which is gap-specific but does **not**
  by itself yield an internal optimum.
- The genuine finite-horizon separation comes from **anisotropic** $\Sigma_g$
  differing between loss types with identical $H_g$ (§4).

---

## 2. Three propositions (revised)

### Proposition A (Ideal-solution invariance, exact-trajectory)
Unchanged: for exact PF pairs, zero-loss solution set is independent of $g$.
(Toy: $f_\beta$ with $\beta=0$ is the ideal map, zero residual for all $g$.)

### Proposition B (finite-horizon optimum depends on optimizer/budget/noise)
Let $H_g = \sigma_d^2\mathbb{E}_t[v_g v_g^\top]$ (curvature) and
$\Sigma_g = \sigma_d^2\mathbb{E}_t[u_g u_g^\top]$ (gradient-noise covariance per
sample), where $u_g$ is the per-sample gradient-noise feature:
- symmetric loss: $u_g = v_g$
- stop-gradient loss: $u_g = [t, t^2]$

The finite-step iterate satisfies
$$\mathbb{E}[\beta_{K+1}\beta_{K+1}^\top] = (I-\eta H_g)\,\mathbb{E}[\beta_K\beta_K^\top]\,(I-\eta H_g)^\top + \eta^2 \Sigma_g.$$
The finite-horizon error $\|\beta_K\|^2$ depends on $K$, $\beta_0$, the
eigenvectors of $H_g$, and $\Sigma_g$. Define
$$g_K^\star = \arg\min_g \mathbb{E}\|\beta_K\|^2,\qquad
g_\rho^\star = \arg\min_g \max_j|1-\eta\lambda_j(H_g)|.$$
**$g_\rho^\star$ is budget-independent; $g_K^\star$ is budget-dependent and
also depends on $\Sigma_g$.** The previous draft conflated the two; they are
distinct and only $g_K^\star$ is the finite-horizon object. The internal-min
existence argument via $\rho_g$ alone is **not** a proof of an internal $g_K^\star$;
that requires the noise term (see Prop C / §4).

### Proposition C (instantaneous criterion does not determine $g_K^\star$)
Two environments with **identical** $H_g$ (hence identical instantaneous
criterion / ADCM recommendation) but **different** $\Sigma_g$ yield
**different** $g_K^\star$. Counterexample constructed in §4.

---

## 3. LR-matched controls (resolves review P0-2)

With $\eta_g = \eta_1\,\lambda_{\max}(H_1)/\lambda_{\max}(H_g)$ (so
$\eta_g\lambda_{\max}(H_g)\equiv$ const, removing the effective-rate effect):

| setting | error vs $g$ | verdict |
|---|---|---|
| noiseless | flat (spread ≈ 0) | original "U" was an LR artifact |
| noisy (isotropic) | **monotone** decreasing, no internal min | gap effect = noise amplification, not internal optimum |

**Implication**: we withdraw the claim "an internal $g_K^\star>1$ exists from
Hessian geometry alone". The toy's honest content is:
1. fixed-LR U-shape $\approx$ effective-rate × stability boundary;
2. LR-matched monotone noise amplification;
3. true finite-horizon separation requires *anisotropic* $\Sigma_g$ (next).

---

## 4. ADCM separation counterexample (resolves review P0/separation, rev.2)

**Exact second-moment recursion (no Monte-Carlo):**
$M_{k+1}=B_g M_k B_g^\top+\eta^2\Sigma_g^{(e)},\ B_g=I-\eta H_g,\ M_0=\beta_0\beta_0^\top,\ E_K=\mathrm{Tr}(M_K).$

Two environments share the **same curvature $H_g$** (hence identical instantaneous
criterion $J_\text{inst}=\mathrm{Tr}(H_g)$) but differ in gradient-noise covariance:
- **env1** (symmetric loss): $\Sigma^{(1)}=H_g$ — the noise feature $v_g=[t{-}r,t^2{-}r^2]$ contains $\Delta\sim g$, so $\Sigma\propto g^2$.
- **env2** (stop-gradient loss): $\Sigma^{(2)}=\sigma_d^2\mathbb{E}[[t,t^2][t,t^2]^\top]$ — the online Jacobian $J_t=[t,t^2]$ has **no $g$ dependence**, so $\Sigma$ is $g$-independent.

This is the real ECT physics: the stop-gradient loss shares curvature $H_g$ with
the symmetric loss (verified: both have population loss $\tfrac12\beta^\top H_g\beta$,
finite-diff rel-err $8\!\times\!10^{-2}$ MC) but its gradient-noise covariance does
**not** scale with $g$, because $g$ enters only the residual, not the online Jacobian.

### Result (fixed $\eta$, exact recursion)
| setting | env1 $g^\star$ | env2 $g^\star$ | differ? |
|---|---|---|---|
| **realistic** (g-dep vs g-indep $\Sigma$) | 0.5 | 1.0–1.45 | **yes** |
| trace-matched (pure direction, equal Tr) | 0.5 | 0.5 | no |

**Two honest conclusions:**
1. **Realistic separation appears**: same $H_g$, same $\eta$, exact recursion, but
   $g^\star$ differs because $\Sigma^{(1)}$ grows with $g$ while $\Sigma^{(2)}$ does
   not. This is gap-dependent finite-horizon geometry the instantaneous criterion
   cannot predict.
2. **Pure direction (equal trace) is insufficient**: when we rescale $\Sigma^{(2)}$
   to equal trace with $\Sigma^{(1)}$, the optima coincide. So the separation does
   *not* come from "different noise direction" alone — it requires the genuine
   difference in *how $\Sigma_g$ depends on $g$*, which is exactly what
   stop-gradient vs symmetric provides.

Caveat: in the **LR-matched** regime ($\eta\propto 1/\lambda_{\max}(H_g)\propto 1/g^2$),
the drift $B_g$ becomes $g$-independent (since $H_g\approx g^2 H_2$), and the
realistic separation above also collapses — there $g^\star$ is driven to the
stability boundary for both. The counterexample is therefore stated for the
**fixed-$\eta$** (real-training) regime, not the LR-matched abstraction.

---

## 5. Stop-gradient operator correction (review P0-3)

**Corrected** $A_g = \sigma_d^2\,\mathbb{E}_t\big[[t,t^2]^\top\,v_g(t)^\top\big]$
(online-branch Jacobian $J_t = [t,t^2]$, NOT $v_g$). Verified two ways:
- per-sample gradient $\mathbb{E}_z[z^2(v_g^\top\beta)[t,t^2]]$ matches $A_g\beta$
  to **rel-err $7\!\times\!10^{-12}$** (analytic $\mathbb{E}_z$);
- population-loss finite-difference gradient matches $H_g\beta$ (curvature),
  confirming population loss is $\tfrac12\beta^\top H_g\beta$ — *same* as
  symmetric loss. The asymmetry of $A_g$ vs $H_g$ is the source of the
  different $\Sigma_g$ in §4.

Previous CSV columns `asym_min/asym_max/asym_norm2` are recomputed from the
corrected $A_g$; old values must not be cited.

---

## 6. Honest caveats & open items (anticipate reviewers)

- The separation effect at $K{=}200$ puts sym's $g^\star$ at the boundary 1.45;
  needs finer $g$ grid + more MC averaging to confirm it is interior.
- All claims are in a **linear** model with 2 parameters; not a deep-net proof
  (explicitly disclaimed, matching the report's A5).
- The "signal–curvature trade-off" $O(\Delta)$/$O(\Delta^2)$ lives at the
  per-pair loss level, not the population Hessian; still requires the
  shared-noise surrogate-bias toy (open).
- Clipping ($\Delta=\min(g\delta_0, t-t_{\min})$) means $H_g$ is **not** exactly
  a degree-4 polynomial in $g$; report polynomial-reconstruction residual
  (open, small).
- Unit tests pending (review minor item).

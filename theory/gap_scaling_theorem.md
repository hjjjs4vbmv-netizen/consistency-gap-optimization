# Gradient Scaling Law for Path-Consistency Objectives

**Role C theory note · 2026-08-19** (rev. P0/P1, single source of truth)

Goal: a theory of *when and why* a consistency objective's gradient becomes
(approximately) a scalar rescaling under a target-gap change — independent of
ECT, quantitative, and falsifiable. The chain:

> objective → small-gap expansion → gradient scaling → non-scalar residual →
> batch aggregation → ECT specialization → testable predictions.

---

## 1. Problem setup

### 1.1 Path-consistency objective

A general path-consistency objective at gap $\delta$ is

$$\ell_\delta(\theta; z) \;=\; w(\delta,z)\,\rho\!\big(|e_\delta(\theta;z)|\big), \qquad \delta = t - r,$$

where $z$ collects the random variables (noise level $t$, noise $\varepsilon$,
dropout mask, image, ...), $w(\delta,z)$ is the loss weighting, $\rho$ is the
outer loss, and $e_\delta$ is the consistency residual. For ECT the residual is

$$e_\delta(\theta;z) \;=\; F_\theta(z,\delta) - T(z,\delta),$$

with $F_\theta$ the student and $T$ a **stop-gradient** target (ECT: $T =
\mathrm{sg}[f_\theta(x_r,r)]$, $\nabla_\theta T = 0$; note ECT uses the same
$\theta$ for both branches, with stop-gradient on the target via `torch.no_grad()`).

### 1.2 Notation (ECT values grounded in code)

ECT's actual objective (`training/loss.py`, `CONFIRMATORY_COMMANDS.sh`):

| symbol | ECT value | meaning |
|--------|-----------|---------|
| $\delta$ | $t - r$ | gap |
| $w(\delta)$ | $\delta^{-1}$ | weighting → **$\alpha = 1$** |
| $\rho(r)$ | $\|e\|$ | **degree-1 homogeneous norm loss** (radial/Euclidean norm; not elementwise L1 — the code computes $\sqrt{\sum_i (D_{y_t}-D_{y_r})_i^2}$ then outer-weights) → **$p = 1$** |
| $\rho_c(r)$ | $\sqrt{r^2+c^2}-c$ | Pseudo-Huber (only if $c>0$; **not used** in the gap runs, `-c 0`) |
| $\Delta_g(t)$ | realized gap under global_sigmoid | see §1.3 |
| schedule | $r/t = 1{-}\frac{1+8\sigma(-t)}{q^{s+1}}$ | $q$ is the budget ($q{=}128$ for the cross-K validation runs; $q{=}256$ in `CONFIRMATORY_COMMANDS.sh`) |

The exponent symbol below is $\nu$ (residual growth order), **not** $q$ (budget).

### 1.3 Realized gap under global_sigmoid (NOT globally "exact")

`_apply_global_gap_scale` (`schedules.py`) multiplies the base gap by $g$, then
clamps $r\ge 0$, so the **realized** gap is

$$\boxed{\; \Delta_g(t) \;=\; \min\!\big(g\,\Delta_1(t),\; t\big). \;}$$

For **unclipped** samples ($g\Delta_1 < t$, the typical case since $\Delta_1$ is a
few percent of $t$), $\Delta_g = g\Delta_1$ exactly. But the clamp triggers when
$g\Delta_1 \ge t$, so predictions must use the **realized ratio** $\Delta_g/\Delta_1$
per sample, not a blanket $1/g$. The clamp rate is a measured runtime metric
(`lower/upper_gap_clip_rate` in `ECMLoss`).

---

## 2. Local gap expansion (residual-order assumption)

### Assumption A (residual-order at zero gap)
The residual vanishes at zero gap ($e_0 = F_\theta(z,0) - \mathrm{sg}[F_\theta(z,0)] = 0$, since student and target coincide at $t=r$) and is sufficiently smooth in $\delta$ with low-order derivatives vanishing at $0$, so that there exist an integer $\nu\ge 1$ and a non-zero vector field $v(z)$ with

$$\boxed{\; e_\delta \;=\; \delta^{\nu}\, v \;+\; O(\delta^{\nu+1}), \qquad v \neq 0. \;}$$

This is stated as an **assumption on the residual order**, not derived from generic
smoothness alone — the integer $\nu$ is the order of the first non-vanishing term,
which depends on the objective's structure (generic $\nu=1$; $\nu=2$ if the first-order
term cancels).

### Assumption B (student Jacobian regularity)
$J_\delta := \nabla_\theta e_\delta$ satisfies $\|J_\delta - J_0\| \le C\delta$.

### Assumption C (stop-gradient)
$\nabla_\theta T = 0$ (the target branch carries no $\theta$-gradient).

### Assumption D (non-degeneracy)
$J_0^\top v \neq 0$ (the leading gradient term does not vanish).

### Assumption E (weighting asymptotics)
For sufficiently small positive gaps, the weighting has the form

$$
 w(\delta,z) = w_0(z)\,\delta^{-\alpha}\bigl(1+O(\delta)\bigr),
 \qquad w_0(z)\neq 0.
$$

An exact power law $w(\delta,z)=w_0(z)\delta^{-\alpha}$ is the stronger special
case. The exponent formula below relies on this assumption; it does not follow
from an otherwise arbitrary weighting $w(\delta,z)$.

> **Theorem scope (what this covers / does not).**
> Covers: stop-gradient target; local path displacement $\delta=t-r$; smooth residual path; radial/degree-$p$-homogeneous outer loss; explicit displacement-dependent weighting; nondegenerate leading gradient.
> Does **not** cover: non-stop-gradient teachers ($\nabla_\theta T\neq 0$); arbitrary RL/Bellman targets; discontinuous schedules (e.g. hard $r=0$ clamp at the boundary); large-gap regime (Taylor invalid).

---

## 3. Gradient scaling law

### Theorem 1 (local gradient scaling)
For the objective in §1.1 under Assumptions A–E, the **sample-level** gradient is

$$g_i' \;=\; \nabla_\theta \ell_{\delta_i} \;=\; w(\delta_i,z_i)\,\rho'\!\big(|e_{\delta_i}|\big)\,\frac{J_{\delta_i}^{\top} e_{\delta_i}}{|e_{\delta_i}|} \;=\; C(z_i)\,\delta_i^{\kappa} \;+\; R_{\delta_i},$$

with

$$\frac{\|R_{\delta_i}\|}{\|g_i'\|} = O(\delta_i)$$

on the non-degenerate local support, and

$$\boxed{\;\kappa \;=\; \nu(p-1) \;-\; \alpha.\;}$$

The exponent $\kappa$ is fixed by the leading terms under Assumptions A and E;
the displayed gradient relation remains asymptotic because the residual path and
Jacobian contribute remainders. When $\rho$ is **exactly** homogeneous of degree $p$
and the weighting is exact power-law, the relative $O(\delta)$ remainder follows from
(i) the Taylor remainder in $e_\delta$ and (ii) the Jacobian drift
$J_\delta-J_0=O(\delta)$. For the degree-1 Euclidean norm used by ECT
($\rho(r)=r$, $p=1$), $\rho$ is exactly homogeneous, so this local bound applies.

**Sample-level prediction:** for two gaps $\delta_1, \delta_g$ on the **same** sample,

$$\boxed{\; a_{{\rm pred},i} \;=\; \left(\frac{\Delta_{g,i}}{\Delta_{1,i}}\right)^{\kappa}, \;}$$

using the **realized** gap ratio (§1.3; $=g^\kappa$ only when unclipped).

### Corollary 1 (homogeneous losses)
If $\rho$ is exactly degree-$p$ homogeneous and $w(\delta)\propto\delta^{-\alpha}$
exactly, the leading-order term carries the exponent $\kappa$ exactly; the full
gradient relation $g_i' = C\,\delta^\kappa + R$ remains asymptotic (the remainder
$R$ and Jacobian drift do not vanish). $a_{\rm pred}$ has **no fitted coefficient**
once $(\nu,p,\alpha)$ are specified — "parameter-free" only in the weak sense
(no fit to data); the general exponent still depends on $\nu$.

---

## 4. Pseudo-Huber specialization (regime change)

ECT's Pseudo-Huber $\rho_c(r) = \sqrt{r^2+c^2}-c$ (matching `loss.py`) has two regimes:

| regime | condition | asymptotic form | effective $p$ | $\kappa$ |
|--------|-----------|-----------------|---------------|----------|
| small-residual | $r \ll c$ | $\rho_c(r)\approx r^2/(2c)$ | $2$ | $\nu - \alpha$ |
| large-residual | $r \gg c$ | $\rho_c(r)\approx r - c$ | $1$ | $-\alpha$ |

$$\boxed{\;\kappa_{\rm small} = \nu - \alpha, \qquad \kappa_{\rm large} = -\alpha.\;}$$

### Corollary 2 (ECT special case)
ECT's gap runs use $c=0$ (degree-1 Euclidean norm, $p=1$, $\alpha=1$):

$$\kappa_{\rm ECT} = \nu(1-1) - 1 = -\alpha = \boxed{-1} \;\Rightarrow\; g_\delta \propto \delta^{-1} \propto 1/g.$$

With $c=0$ the small/large distinction collapses (degree-1 Euclidean norm everywhere). The $1/g$ law
is the $p{=}1,\alpha{=}1$ point of the general law. **Crucially, at $p=1$ the exponent
is independent of $\nu$** — this is a near-zero-parameter special case, but it also
means the residual-expansion parameter $\nu$ is **not** probed by this configuration.

---

## 5. Validation status (honest, post-review)

**What is tested.** The aggregate/minibatch-gradient scalar

$$a_j^\star = \frac{\langle G_j^g,G_j^1\rangle}{\|G_j^1\|^2}$$

matches $a_{\rm pred}=g^{-1}=1/1.3=0.769$ to mean relative error 0.18% across
four checkpoints of **one** $q{=}128$ seed-3 trajectory. This is a **retrospective
consistency check along one trajectory**, not a sample-level measurement, multi-seed
validation, or multi-dataset validation. A sample-level $a_i^\star$ would require a
separate measurement and is outside this theory note.

**What it establishes (weak).** At $p=1$, $\nu$ drops out, so the 0.18% aggregate
match supports (i) the weight-ratio prediction ($1/g$) and (ii) consistency with local
aggregate gradient geometry. It does **not** constrain $\nu$, $v$, or $J_0$ — the
residual-expansion machinery (Assumption A) is idle at $p=1$.

**Stage-invariance is a schedule-construction fact.** global_sigmoid makes the
realized ratio $\Delta_g/\Delta_1=g$ (unclipped), so $a_{\rm pred}=g^\kappa$ is
$t$-independent by construction; the observed flatness confirms the schedule is
well-implemented, not a novel prediction.

**Gradient scaling ≠ optimizer-update scaling.** The theorem predicts
**sample-level raw-gradient** scaling $g_i'\approx a\,g_i$. The cross-K/balanced-β
experiments measured two distinct objects: (a) $a^\star$ — raw-gradient (directly
tests the theory, above); (b) $R^2$/`R_opt` — the **optimizer-update** ratio
$h=U_g/U_1$. Finite-history, state-mismatched, time-varying, or non-scalar gradient
transformations can become non-equivalent under adaptive optimization (Adam/RAdam):
note that **ideal** rectified RAdam with a full constant-scale history and matched
moment scaling has an *exact* null (the scale cancels), so the optimizer does not
*inherently* break scalarity — it breaks it under the truncated/state-mismatched
conditions of real training. Object (b) is governed by this optimizer dynamics, not
by Theorem 1; an $R^2$ mismatch cannot falsify the gradient theory.

---

## 6. Minibatch aggregation

### Proposition 2 (exact batch decomposition)
Let per-sample gradients be $g_i' = a_i g_i + r_i$ ($a_i$ sample-level scalar, $r_i$
local non-scalar residual). With $G=\sum_i g_i$, $G'=\sum_i g_i'$, batch-average
scalar $\bar a$:

$$\boxed{\; G' \;=\; \bar a\, G \;+\; \underbrace{\sum_i (a_i - \bar a)\, g_i}_{\text{heterogeneity term}} \;+\; \underbrace{\sum_i r_i}_{\text{local non-scalar residual}}. \;}$$

This is an **exact algebraic identity**. The batch best-fit scalar
$\bar a_{\rm batch}^\star = \langle G',G\rangle/\|G\|^2$ is the aggregate
least-squares scalar, distinct from the sample-level $a_i$; the two are related by
this decomposition.

### Heuristic 3 (emergent scalarity — NOT a theorem)
If $g_i \approx \mu + \xi_i$ with a non-zero common mean direction $\mu$ and the
heterogeneity term is approximately centered / weakly correlated across the batch,
then the batch signal accumulates as $O(n)$ while the heterogeneity fluctuation
grows as $\approx O(\sqrt{n})$, so the **relative** heterogeneity may decay with
batch size. Under such conditions, aggregate $G',G$ can show higher scalarity
(cosine) than individual $g_i',g_i$ — scalarity emerging through cancellation.

> **Status: sufficient-condition sketch, not a theorem.** The "cancellation" is a
> high-dimensional concentration heuristic (CLT-like), requiring iid + high-dim +
> weak-alignment assumptions that are not fully stated here. It can fail for small
> batches or strongly-aligned gradients. Counterexamples exist; treat as a
> hypothesis to test, not an established result.

### Bound (corollary of Prop 2)
$$\|G' - \bar a G\| \;\le\; \sum_i |a_i - \bar a|\,\|g_i\| \;+\; \sum_i \|r_i\|.$$

---

## 7. ECT specialization (realized gap)

Using the realized ratio (§1.3), the per-sample prediction is

$$\boxed{\; a_{{\rm pred},i} \;=\; \left(\frac{\Delta_{g,i}}{\Delta_{1,i}}\right)^{\kappa}, \;}$$

which $=g^\kappa=g^{-1}$ for unclipped samples. For ECT ($\kappa=-1$): $a_{\rm pred}=g^{-1}=1/g$ (unclipped), independent of $t$ and stage — but Alicia should test the **realized** ratio per sample, not a fixed $1/g$, to account for clipping.

```python
def a_pred(t, r_baseline, r_probe, p=1.0, alpha=1.0, nu=1):
    """Predicted gap-to-gap scalar (realized). ECT (c=0): p=1, alpha=1 => kappa=-1."""
    d1 = max(t - r_baseline, 0)   # realized baseline gap (clamped, matches loss.py)
    dg = max(t - r_probe, 0)      # realized probe gap
    if d1 <= 0: return float('nan')   # clipping boundary (failure regime)
    kappa = nu*(p - 1) - alpha
    return (dg / d1) ** kappa
```

---

## 8. Target × denominator factorization

Four-cell factorial (factors: target geometry $r$, denominator/weighting $\Delta$):

| cell | target | gap | gradient |
|------|--------|-----|----------|
| A | $r_1$ | $\Delta_1$ | $G_A$ |
| B | $r_g$ | $\Delta_g$ | $G_B$ |
| C | $r_g$ | $\Delta_1$ | $G_C$ |
| D | $r_1$ | $\Delta_g$ | $G_D$ |

Factorize $g(r,\Delta) \approx s(\Delta)\,h(r) + \varepsilon(r,\Delta)$ ($s$=scalar
magnitude, $h$=target-geometry direction). Then

$$\boxed{\; G_{\rm int} \;=\; G_B - G_C - G_D + G_A \;=\; (s_g - s_1)(h_g - h_1) \;+\; (\varepsilon_B - \varepsilon_C - \varepsilon_D + \varepsilon_A). \;}$$

$G_{\rm int}$ is a **factorization-adequacy diagnostic**: a large value challenges a
simple weak-coupling factorization $g(r,\Delta)\approx s(\Delta)h(r)$; a small value is
**consistent with**, but does not prove, approximate separability (even under exact
factorization, $G_{\rm int}=(s_g-s_1)(h_g-h_1)$ need not vanish). It is **not** asserted
to be small a priori — its magnitude is the empirical question.

**Conditional hypotheses (not consequences of Theorem 1).** Under the factorization
ansatz, the denominator-only move A→D changes only $s(\Delta)$ and should show
near-pure scalar rescaling, $\cos(G_D,G_A)\approx 1$; the target-only move A→C changes
$h(r)$ and may show a larger directional residual, $\cos(G_C,G_A)<\cos(G_D,G_A)$. These
are testable hypotheses for the existing factorial pipeline, not derived predictions;
the gradient geometry is not symmetric in the two factors.

---

## 9. Failure regimes

The theorem is local ($\delta\to 0$, Assumption A); it fails when:
1. **Large gap** — Taylor expansion invalid.
2. **Pseudo-Huber crossover** ($c>0$) — effective $p$ shifts (§4).
3. **Clipping/boundary** — $r=\max(0,\cdot)$ makes $\delta$ non-smooth (§1.3).
4. **Degenerate leading term** — $J_0^\top v\approx 0$ (Assumption D violated).
5. **Target Jacobian drift** — $J_{\delta_g}\not\approx J_{\delta_1}$.
6. **Teacher not stop-gradient** — $\nabla_\theta T\neq 0$ (Assumption C violated; §1.1 scope).

---

## 10. Falsifiable predictions

**Non-trivial (test the $\nu$-dependent content — currently UNTESTED):**

| # | Prediction | Observable | Expected |
|---|-----------|-----------|----------|
| N1 | Pseudo-Huber regime change ($c>0$) | $a^\star$ small vs large residual | different $\kappa$ ($\nu{-}\alpha$ vs $-\alpha$) |
| N2 | denominator scalarity | $\cos(G_D, G_A)$ | high (near-pure rescale, conditional on factorization) |
| N3 | target directional effect | $\cos(G_C, G_A)$ | $< \cos(G_D, G_A)$ (conditional) |
| N4 | emergent batch scalarity (Heuristic 3) | sample vs batch $\epsilon_{\rm ns}$ | batch < sample (if conditions hold) |
| N5 | factorization diagnostic | $\|G_{\rm int}\|/\|G_B\|$ | empirical (small = consistent with separability, not a proof) |

**Weakly supported (do NOT test the $\nu$-content):**

| # | Prediction | Status |
|---|-----------|--------|
| W1 | aggregate gap scaling $a_j^\star/a_{\rm pred}\approx 1$ | weakly checked (0.18%, one q=128 seed-3 trajectory; $\nu$ idle at $p=1$) |
| W2 | stage invariance | schedule-construction fact |

The genuinely theory-specific content is in **N1–N5**; W1–W2 are consistency checks
that would hold under almost any model with the ECT loss schedule. A reviewer should
**not** read W1 as validating the residual-expansion theorem.

---

## Claim boundaries (must preserve)

**CAN say:**
- "Under a degree-$p$-homogeneous outer loss and power-law weighting $\delta^{-\alpha}$, the sample-level gradient scales as $g_\delta \sim \delta^{\nu(p-1)-\alpha}$ to leading order; $a_{\rm pred}=(\delta_2/\delta_1)^\kappa$ has no fitted coefficient once $(\nu,p,\alpha)$ are specified."
- "ECT's $1/g$ law is the $p=1,\alpha=1$ point. The raw-gradient $a^\star\approx0.77$ matches $a_{\rm pred}=g^{-1}=0.769$ (0.18%) — a direct but **weak** test (at $p=1$, $\nu$ drops out; confirms weight ratio + Jacobian smoothness, not the residual-expansion content)."
- "The batch decomposition (Proposition 2) is exact; emergent scalarity (Heuristic 3) is a sufficient-condition sketch."

**CANNOT say:**
- "The 0.18% match validates the residual-expansion theorem / its non-trivial content" — $\nu$ is idle at $p=1$; N1–N5 are untested.
- "The gradient scaling theorem implies identical optimization trajectories" — adaptive optimization can make them non-equivalent (§5).
- "Scale-drift dynamics don't matter" — the experiments had near-constant $a^\star$ (std 1–2%), so cannot distinguish scalar-mean from per-step-history.
- "The theorem is proven for ECT" — weak consistency with one trajectory; failure regimes and N1–N5 are falsifiable, not established.
- "Gradient scaling causes the FID improvement" — **no FID-causality claim.**

---

## Open / next (no new experiment authorized by this note)

- Empirical validation of N1–N5 is **deferred to the existing Role D / w10800
  pipeline** (factorial; $c>0$ regime). This theory note authorizes no new
  full-training experiment and does not re-assign work; the current four-arm run
  is owned by the existing pipeline.
- Explicit finite-gap remainder bound $|R_\delta|\le C_1\delta^{\kappa+1}$: needs the
  second-order term; deferred.

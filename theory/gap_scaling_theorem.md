# Gradient Scaling Law for Path-Consistency Objectives

**Role C theory note · 2026-08-19**

Goal: build a theory of *when and why* a consistency objective's gradient
becomes (approximately) a scalar rescaling under a target-gap change — one that
is (i) independent of ECT, (ii) produces a quantitative prediction, and (iii) is
falsifiable. The chain is:

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
\mathrm{sg}[f_{\bar\theta}(x_r,r)]$, $\nabla_\theta T = 0$). We keep the form
abstract: the theorems depend only on $T$ being $\theta$-independent in the
gradient.

### 1.2 Notation (ECT values grounded in code)

ECT's actual objective (`training/loss.py`, `CONFIRMATORY_COMMANDS.sh`):

| symbol | ECT value | meaning |
|--------|-----------|---------|
| $\delta$ | $t - r$ | gap |
| $w(\delta)$ | $\delta^{-1}$ | weighting → **$\alpha = 1$** |
| $\rho(r)$ | $r$ (L1, since `-c 0`) | outer loss → **$p = 1$** |
| $\rho_c(r)$ | $\sqrt{r^2+c^2}-c$ | Pseudo-Huber (only if $c>0$; **not used** in the gap runs) |
| $\Delta_g(t)$ | $g\,\Delta_1(t)$ | global_sigmoid gap, **exact** (schedules.py `_apply_global_gap_scale`) |
| schedule | $r/t = 1{-}\frac{1+8\sigma(-t)}{q^{s+1}}$, $q{=}256$ | |

The exponent symbol below is $\nu$ (residual growth order), **not** $q$ (budget).

---

## 2. Local gap expansion

### Proposition 1 (residual expansion)

**Assumption A (target-path smoothness).** $T(z,\delta)$ is $C^{2}$ in $\delta$ along the target path near $\delta=0$.

**Assumption B (student Jacobian regularity).** $J_\delta := \nabla_\theta e_\delta$ satisfies $J_\delta = J_0 + O(\delta)$, i.e. $\|J_\delta - J_0\| \le C\delta$.

**Assumption C (stop-gradient).** $\nabla_\theta T = 0$ (the target branch carries no $\theta$-gradient).

Then there exist a vector field $v(z)$ and an integer $\nu \ge 1$ such that

$$\boxed{\; e_\delta \;=\; \delta^{\nu}\, v \;+\; O(\delta^{\nu+1}). \;}$$

*Sketch.* Under A, $T(z,\delta) = T(z,0) + \partial_\delta T|_0\,\delta + \tfrac12 \partial_\delta^2 T|_0\,\delta^2 + \cdots$. With C, $\nabla_\theta e_\delta = J_\delta = \nabla_\theta F_\theta(z,\delta)$, and B gives $F_\theta(z,\delta) = F_\theta(z,0) + O(\delta)$. Subtracting, $e_\delta = [F_\theta(z,\delta)-F_\theta(z,0)] - [T(z,\delta)-T(z,0)]$; the leading non-vanishing term is $\delta^\nu v$. ∎

$\nu$ is the **residual growth order**: $\nu=1$ when the first-order term is non-zero (generic), $\nu=2$ if it cancels.

### Assumption D (non-degeneracy)
The leading term is non-vanishing: $v \neq 0$ and $J_0^\top v \neq 0$. Otherwise the gradient's leading order jumps to $\nu+1$.

---

## 3. Gradient scaling law

### Theorem 1 (local gradient scaling)

For the objective in §1.1, the per-sample gradient is (chain rule, using C so only $F_\theta$ contributes)

$$g_\delta \;=\; \nabla_\theta \ell_\delta \;=\; w(\delta,z)\,\rho'\!\big(|e_\delta|\big)\,\frac{J_\delta^{\top} e_\delta}{|e_\delta|}.$$

Substituting Proposition 1 and Assumption B:

$$\boxed{\; g_\delta \;=\; C(z)\,\delta^{\kappa} \;+\; R_\delta, \qquad \frac{|R_\delta|}{|g_\delta|} = O(\delta), \;}$$

where $C(z) = w_1\,\rho'_1\,J_0^{\top}v$ (the leading prefactor evaluated at the leading residual) and the exponent is

$$\boxed{\;\kappa \;=\; \nu(p-1) \;-\; \alpha.\;}$$

Here $\alpha$ is the weighting exponent ($w(\delta) \propto \delta^{-\alpha}$) and $p$ is the homogeneity degree of $\rho$ ($\rho(cr) \approx c^p \rho(r)$). **Both gaps satisfy the same scaling**, so for two gaps $\delta_1, \delta_2$:

$$\boxed{\; a_{\rm pred}(z) \;=\; \frac{g_{\delta_2}}{g_{\delta_1}} \;\approx\; \left(\frac{\delta_2}{\delta_1}\right)^{\kappa}, \qquad |a_{\rm pred} - a^\star| = O(\delta), \;}$$

with $a^\star = \langle G_{\delta_2}, G_{\delta_1}\rangle / \|G_{\delta_1}\|^2$ the empirical best scalar.

*Sketch.* $|e_\delta| \sim \delta^\nu \Rightarrow \rho'(|e_\delta|) \sim \delta^{\nu(p-1)}$ under homogeneity $p$; $w(\delta)\sim\delta^{-\alpha}$; $J_\delta^\top e_\delta/|e_\delta| \to J_0^\top v$ (unit vector, $O(1)$). The product scales as $\delta^{\nu(p-1)-\alpha} = \delta^\kappa$. The remainder $R_\delta$ collects the $O(\delta^{\nu+1})$ residual, the $O(\delta)$ Jacobian drift $J_\delta-J_0$, and loss non-homogeneity. ∎

**This is the central theorem.** $\kappa = \nu(p-1)-\alpha$ is a closed-form prediction from the loss geometry, with no fit parameter.

### Corollary 1 (homogeneous losses)
If $\rho$ is exactly homogeneous of degree $p$ and $w(\delta)\propto\delta^{-\alpha}$ exactly, the scaling $g_\delta \sim \delta^\kappa$ is exact to leading order, and $a_{\rm pred}$ is parameter-free.

---

## 4. Pseudo-Huber specialization (regime change)

ECT's Pseudo-Huber $\rho_c(r) = c^2(\sqrt{1+(r/c)^2}-1)$ has two regimes:

| regime | condition | effective $p$ | $\kappa$ |
|--------|-----------|---------------|----------|
| small-residual | $r \ll c$ | $2$ (since $\rho_c \approx \tfrac12 r^2$) | $\nu - \alpha$ |
| large-residual | $r \gg c$ | $1$ (since $\rho_c \approx c r$) | $-\alpha$ |

$$\boxed{\;\kappa_{\rm small} = \nu - \alpha, \qquad \kappa_{\rm large} = -\alpha.\;}$$

### Corollary 2 (inverse-gap regime)
For ECT's actual config ($c=0 \Rightarrow p=1$, $\alpha=1$):

$$\kappa_{\rm ECT} = \nu(1-1) - 1 = -\alpha = \boxed{-1} \;\;\Rightarrow\;\; g_\delta \propto \delta^{-1} \propto 1/g.$$

**The $1/g$ law is not an ECT empirical accident — it is the $p=1,\alpha=1$ point of the general scaling law, and it holds for any $\nu$.** Note: with $c=0$ the small/large distinction collapses (pure L1 everywhere), so there is no regime change for the gap experiments; the regime change is a *prediction* for a c>0 run (§9).

**Zero-parameter validation:** $a_{\rm pred} = g^{-1} = 1/1.3 = 0.769 \approx$ observed $a^\star \approx 0.77$ (cross-K, all four stages). The theorem predicts the scalar with no fit parameter.

---

## 5. Beyond exact scalarity (non-scalar residual)

Exact scalarity $G_B = aG_A$ is too strong; write

$$G_B \;=\; a_{\rm pred}\,G_A \;+\; R, \qquad \epsilon_{\rm ns} := \frac{\|G_B - a_{\rm pred}G_A\|}{\|G_B\|}.$$

Sources of $R$ (each is a falsifiable mechanism):

1. **Finite-gap Taylor remainder** $O(\delta^{\nu+1})$ — dominant for large gaps.
2. **Jacobian drift** $J_{\delta_2} \neq J_{\delta_1}$ — changes gradient *direction*, not just magnitude.
3. **Loss non-homogeneity** — Pseudo-Huber crossover (only if $c>0$).
4. **Sample-dependent gap** $\delta_i$ — $t$ is random, so $\delta_i$ varies per sample.
5. **Clipping/boundary** — $r=\max(0,\cdot)$ makes the schedule non-smooth at $r=0$ (loss.py:251 `mask * D_yr`).
6. **Target geometry change** — changing $r$ moves the target *point*, not just gap magnitude (connects to §7 factorial).

---

## 6. Minibatch aggregation (emergent scalarity)

### Proposition 2 (heterogeneous scaling decomposition)

Let per-sample gradients be $g_i' = a_i g_i + r_i$ (sample-level scalar $a_i$ plus local non-scalar residual $r_i$). The batch gradient $G' = \sum_i g_i'$, $G = \sum_i g_i$, with batch-average scalar $\bar a$:

$$\boxed{\; G' \;=\; \bar a\, G \;+\; \underbrace{\sum_i (a_i - \bar a)\, g_i}_{\text{heterogeneity residual}} \;+\; \underbrace{\sum_i r_i}_{\text{local non-scalar residual}}. \;}$$

### Corollary 3 (residual bound)

$$\frac{\|G' - \bar a G\|}{\|G'\|} \;\lesssim\; \underbrace{\text{scaling heterogeneity}}_{\propto \mathrm{Var}(a_i)} \;+\; \underbrace{\text{local remainder}}_{\propto \mathbb{E}\|r_i\|}.$$

### Proposition 3 (emergent scalarity)

**Approximate gradient scalarity need not hold sample-wise; it can emerge after minibatch aggregation through cancellation of the heterogeneous residual.** If $g_i$ are weakly aligned across the batch (typical: high-dim gradients are near-orthogonal), then $\sum_i(a_i-\bar a)g_i$ partially cancels and $\|G'-\bar a G\|/|G'| < \mathbb{E}_i[\|g_i'-a_ig_i\|/\|g_i'\|]$.

*Why this matters:* it explains the empirical observation that **aggregate** $G',G$ have high cosine while **individual** $g_i',g_i$ look noisy — scalarity is an emergent batch property, not a per-sample one.

---

## 7. ECT specialization (global sigmoid schedule)

For global_sigmoid the gap is **exact**: $\Delta_g(t) = g\,\Delta_1(t)$ (schedules.py). Theorem 1 therefore predicts, per sample/time,

$$\boxed{\; a_{\rm pred}(t) \;=\; \left(\frac{\Delta_g(t)}{\Delta_1(t)}\right)^{\kappa} = g^{\kappa}. \;}$$

For ECT ($\kappa=-1$): $a_{\rm pred}(t) = g^{-1} = 1/g$, **independent of $t$ and stage** — matching the cross-K finding that $a^\star \approx 0.77$ is stage-invariant. (If the schedule were not an exact global multiplier, $a_{\rm pred}(t)$ would be $t$-dependent; the global_sigmoid design makes it constant, which is why the cross-K $a^\star$ is flat across $K$.)

### Computable prediction (spec for Alicia)

```python
def a_pred(t, r_baseline, r_probe, p=1.0, alpha=1.0):
    """Predicted gap-to-gap scalar. For ECT (c=0): p=1, alpha=1 => kappa=-1."""
    d1 = max(t - r_baseline, 0)   # baseline gap (clamped, matches loss.py)
    dg = max(t - r_probe, 0)     # probe gap
    if d1 <= 0: return float('nan')   # clipping boundary (failure regime)
    nu = 1  # generic first-order residual
    kappa = nu*(p - 1) - alpha
    return (dg / d1) ** kappa
```

For the gap experiments ($r_{\rm probe}=r_g$, $r_{\rm baseline}=r_1$, $g=1.3$): `a_pred = g**-1 = 0.769`. Compare to `a_star = <G_g, G_1>/||G_1||^2`.

---

## 8. Target × denominator factorization

The four-cell factorial (factors: target geometry $r$, denominator/weighting $\Delta$):

| cell | target | gap | gradient |
|------|--------|-----|----------|
| A | $r_1$ | $\Delta_1$ | $G_A$ |
| B | $r_g$ | $\Delta_g$ | $G_B$ |
| C | $r_g$ | $\Delta_1$ | $G_C$ |
| D | $r_1$ | $\Delta_g$ | $G_D$ |

Factorize the gradient as

$$g(r,\Delta) \;\approx\; s(\Delta)\,h(r) \;+\; \varepsilon(r,\Delta),$$

where $s(\Delta)$ controls scalar magnitude (denominator effect, Theorem 1) and $h(r)$ is the target-geometry direction. Then $G_A=s_1 h_1$, $G_B=s_g h_g$, $G_C=s_1 h_g$, $G_D=s_g h_1$ (ignoring $\varepsilon$). The interaction

$$\boxed{\; G_{\rm int} \;=\; G_B - G_C - G_D + G_A \;}$$

measures departure from factorization: $G_{\rm int} = \varepsilon_B - \varepsilon_C - \varepsilon_D + \varepsilon_A$, the **local target–denominator coupling term**. It is not an arbitrary metric — it is the leading non-factorized correction.

**Prediction (denominator vs target):** the denominator-only move A→D changes only $s(\Delta)$ ⇒ **near-pure scalar rescaling** $\cos(G_D,G_A)\approx 1$. The target-only move A→C changes $h(r)$ ⇒ **directional residual** $\cos(G_C,G_A) < \cos(G_D,G_A)$. The gradient geometry is *not* symmetric in the two factors.

---

## 9. Failure regimes

The theorem is local ($\delta\to 0$); it fails when:

1. **Large gap** — Taylor expansion $e_\delta = \delta^\nu v + O(\delta^{\nu+1})$ invalid.
2. **Pseudo-Huber crossover** ($c>0$ only) — effective $p$ changes between small/large residual; $\kappa$ shifts from $\nu{-}\alpha$ to $-\alpha$. **Prediction:** a $c>0$ run should show a *different* $a^\star$ in the two residual regimes.
3. **Clipping/boundary** — $r=\max(0,\cdot)$ (loss.py:251) makes $\delta$ non-smooth; $a_{\rm pred}=$ nan.
4. **Degenerate leading term** — $J_0^\top v \approx 0 \Rightarrow$ leading gradient vanishes, $\nu$ jumps.
5. **Target Jacobian drift** — $J_{\delta_2}\not\approx J_{\delta_1}$ over a large gap.
6. **Teacher not stop-gradient** — $\nabla_\theta T \neq 0$ changes the gradient structure entirely (Assumption C violated).

### Optimizer caveat (critical)

$$\boxed{\;\text{gradient scaling theorem} \;\neq\; \text{optimizer update scaling theorem}.\;}$$

Adam/RAdam transforms the gradient through $m_t, v_t$: a scalar gradient relation $G_B \approx a G_A$ does **not** imply $U_B \approx a U_A$. The optimizer can break scalarity (this is the RAdam moment-memory mechanism of the cross-K/balanced-β experiments). Gradient scaling must be tested *and* optimizer-update scaling separately — the two answer different questions.

---

## 10. Falsifiable predictions (for Alicia / empirical)

| # | Prediction | Observable | Expected |
|---|-----------|-----------|----------|
| 1 | gap scaling | $a^\star / a_{\rm pred}$ | $\approx 1$ (ECT: $a_{\rm pred}=0.769$, $a^\star\approx0.77$) |
| 2 | denominator scalarity | $\cos(G_D, G_A)$ | high (near-pure rescale) |
| 3 | target directional effect | $\cos(G_C, G_A)$ | $< \cos(G_D, G_A)$ |
| 4 | emergent batch scalarity | sample vs batch residual | batch $\epsilon_{\rm ns}$ < sample $\epsilon_{\rm ns}$ |
| 5 | interaction higher-order | $\|G_{\rm int}\| / \|G_B\|$ | small (higher-order coupling) |
| 6 | stage invariance | $a^\star$ across $K$ | flat (global_sigmoid ⇒ $g^\kappa$ is $t$-independent) |
| 7 | regime change (if $c>0$) | $a^\star$ small vs large residual | different $\kappa$ |

Predictions 1 and 6 are already supported by the cross-K experiment (#58): $a^\star\approx0.77$ across all four stages, matching $a_{\rm pred}=g^{-1}$. Predictions 2–5 require the four-cell factorial (Role D / w10800). Prediction 7 requires a $c>0$ run (not yet done).

---

## Claim boundaries (must preserve)

**CAN say:**
- "Under a homogeneous outer loss of degree $p$ and power-law weighting $\delta^{-\alpha}$, the per-sample gradient scales as $g_\delta \sim \delta^{\nu(p-1)-\alpha}$ to leading order; the gap-to-gap scalar $a_{\rm pred} = (\delta_2/\delta_1)^\kappa$ is parameter-free."
- "ECT's $1/g$ law is the $p=1,\alpha=1$ point of this theorem; it predicts $a_{\rm pred}=g^{-1}=0.769$, matching the observed $a^\star\approx0.77$ with no fit parameter, across training stages."
- "Approximate scalarity can emerge at the batch level through cancellation of heterogeneous per-sample scalars."

**CANNOT say:**
- "The gradient scaling theorem implies identical optimization trajectories" — the optimizer breaks scalarity (§9 caveat).
- "Scale-drift dynamics don't matter" — the cross-K/predictor-comparison experiments had a near-constant $a^\star$ (std 1–2%), so they *cannot distinguish* scalar-mean from per-step-history; this theorem makes no claim about time-varying-scale regimes.
- "The theorem is proven for ECT" — it is a leading-order prediction validated by one parameter-free match; the non-scalar residual $R$ and failure regimes are falsifiable, not established.
- "Gradient scaling causes the FID improvement" — **no FID-causality claim.** This is a characterization of the gradient geometry, not a causal identification of ECT's quality outcomes.

---

## Open / next (not Role C today)

- Falsify predictions 2–5 (factorial): Role D / w10800.
- Prediction 7 (regime change under $c>0$): requires a Pseudo-Huber run.
- Finite-gap remainder bound $|R_\delta| \le C_1\delta^{\kappa+1}$: needs the explicit second-order term; deferred.

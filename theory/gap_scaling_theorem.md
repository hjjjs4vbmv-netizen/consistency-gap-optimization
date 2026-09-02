# Gradient Scaling Law for One-Sided Stop-Gradient Consistency Objectives

**Role C theory note · 2026-08-23** (one-sided SG derivative + exact ECT decomposition; single source of truth)

Goal: a theory of *when and why* the **one-sided stop-gradient training
derivative** of a consistency objective becomes approximately scalar under a
target-gap change, together with an exact decomposition for the ECT objective.
The chain:

> lifted objective → one-sided SG derivative → exact ECT decomposition →
> small-gap scaling → batch aggregation → testable predictions.

---

## 1. Problem setup

### 1.1 Lifted objective and the stop-gradient training derivative

Stop-gradient is a rule for differentiation, not a statement that the target's
forward value is independent of the current parameters. To make that distinction
explicit, introduce two parameter arguments: $\theta$ for the online/student
branch and $\bar\theta$ for the target branch. At gap $\delta=t-r$, define

$$
\widetilde e_\delta(\theta,\bar\theta;z)
  := F^+_\theta(z,\delta)-F^-_{\bar\theta}(z,\delta),
\qquad
\widetilde\ell_\delta(\theta,\bar\theta;z)
  := w(\delta,z)\,\Phi\!\left(\widetilde e_\delta(\theta,\bar\theta;z)\right),
$$

where $z$ contains all quantities held fixed by one backward pass (data, $t$,
shared noise, augmentation, dropout realization, schedule state, and so on).
The forward residual used in training is the diagonal value

$$e_\delta(\theta;z):=\widetilde e_\delta(\theta,\theta;z),$$

but its **one-sided stop-gradient training derivative** is

$$
\boxed{
\mathcal D^{\rm sg}_\theta\ell_\delta(\theta;z)
  :=\left.
  \partial_\theta\widetilde\ell_\delta(\theta,\bar\theta;z)
  \right|_{\bar\theta=\theta}.
}
$$

Thus the partial derivative is taken first, with $\bar\theta$ frozen, and only
then are the two parameter values identified. This is exactly the autograd rule
implemented by `sg[...]`, `detach()`, or `torch.no_grad()`. It is generally **not**
the ordinary derivative of the diagonal scalar
$\widetilde\ell_\delta(\theta,\theta;z)$, which would contain both branch terms:

$$
\frac{d}{d\theta}\widetilde\ell_\delta(\theta,\theta;z)
=\left.(\partial_\theta+\partial_{\bar\theta})
\widetilde\ell_\delta(\theta,\bar\theta;z)\right|_{\bar\theta=\theta}.
$$

Accordingly, $\mathcal D^{\rm sg}_\theta\ell$ is the vector used by the training
algorithm; it need not be a conservative gradient field of any scalar function
on the diagonal. This note deliberately proves results only for this **one-sided
SG derivative**. It does not claim a unified theorem for arbitrary mixtures of
online- and target-branch derivatives.

Let

$$
J^+_\delta(\theta;z)
  :=\left.\partial_\theta F^+_\theta(z,\delta)\right|_\theta .
$$

When $w$ and the sampled schedule values are held fixed during backpropagation,
the exact one-sided chain rule is

$$
\boxed{
g^{\rm sg}_\delta
:=\mathcal D^{\rm sg}_\theta\ell_\delta
=w(\delta,z)\,{J^+_\delta}^{\!\top}
  \nabla_e\Phi(e_\delta).
}
$$

For ECT, $F^+_\theta=f_\theta(x_t,t)$ and
$F^-_{\bar\theta}=f_{\bar\theta}(x_r,r)$; the latter is evaluated with the same
numerical parameter value but under `torch.no_grad()`.

### 1.2 Notation (ECT values grounded in code)

ECT's actual objective (`training/loss.py`, `CONFIRMATORY_COMMANDS.sh`):

| symbol | ECT value | meaning |
|--------|-----------|---------|
| $\delta$ | $t - r$ | gap |
| $w(\delta)$ | $\delta^{-1}$ | weighting → **$\alpha = 1$** |
| $\Phi_0(e)$ | $\|e\|_2$ | **degree-1 homogeneous Euclidean norm** (not elementwise L1 — the code computes $\sqrt{\sum_i (D_{y_t}-D_{y_r})_i^2}$ before weighting) → **$p = 1$** |
| $\Phi_c(e)$ | $\sqrt{\|e\|_2^2+c^2}-c$ | Pseudo-Huber (only if $c>0$; **not used** in the gap runs, `-c 0`) |
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

### Assumption A (forward residual order at zero gap)
At fixed $\theta$ and $z$, the **forward** residual on the parameter diagonal
vanishes at zero gap because the two path endpoints coincide. It admits

$$
e_\delta(\theta;z)
=\delta^\nu v(\theta,z)+O(\delta^{\nu+1}),
\qquad \nu\ge 1,\quad v\neq0.
$$

This is an assumption about forward values, not about the diagonal derivative
with respect to $\theta$. The integer $\nu$ is the order of the first
non-vanishing displacement term (generically $\nu=1$; $\nu=2$ only if the
first-order term cancels).

### Assumption B (online-branch Jacobian regularity)
The one-sided Jacobian in §1.1 satisfies

$$\|J^+_\delta-J^+_0\|=O(\delta).$$

For a coupled ECT gap comparison at fixed $(x_t,t)$, the online branch is
unchanged and hence $J^+_\delta=J_t$ exactly; Assumption B is then automatic.

### Assumption C (outer-loss homogeneity and regularity)
$\Phi:\mathbb R^d\to\mathbb R$ is positively homogeneous of degree $p>0$,

$$\Phi(ae)=a^p\Phi(e),\qquad a>0,$$

and $\nabla\Phi$ is locally Lipschitz in a neighbourhood of $v$. Equivalently,
$\nabla\Phi(ae)=a^{p-1}\nabla\Phi(e)$ away from the origin. The Euclidean norm
satisfies this condition around every $v\neq0$. Pseudo-Huber is not globally
homogeneous and is treated by its two local regimes in §4.

### Assumption D (non-degeneracy)
The leading one-sided SG coefficient does not vanish:

$$\boxed{\;{J^+_0}^{\!\top}\nabla\Phi(v)\neq0.\;}$$

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
> Covers: the one-sided SG derivative defined in §1.1; local positive path
> displacement $\delta=t-r$; a smooth forward residual path; a differentiable
> degree-$p$-homogeneous outer loss away from zero; explicit
> displacement-dependent weighting; and a nondegenerate leading SG term.
> Does **not** cover: the ordinary diagonal derivative, symmetric/two-sided
> consistency gradients, arbitrary mixtures of online and target derivatives,
> discontinuous schedule boundaries, or the large-gap regime.

---

## 3. Gradient scaling law

### Theorem 1 (local scaling of the one-sided SG training derivative)
Under Assumptions A–E, the exact chain rule in §1.1 gives the sample-level
expansion

$$
\boxed{
g^{\rm sg}_\delta
=C(\theta,z)\,\delta^\kappa+O(\delta^{\kappa+1}),
\qquad
C(\theta,z)
=w_0(z){J^+_0}^{\!\top}\nabla\Phi(v),
}
$$

where

$$\boxed{\;\kappa \;=\; \nu(p-1) \;-\; \alpha.\;}$$

On the nondegenerate support from Assumption D,

$$
\frac{\|g^{\rm sg}_\delta-C\delta^\kappa\|}
     {\|g^{\rm sg}_\delta\|}=O(\delta).
$$

**Proof.** Assumption A and positive homogeneity give

$$
\nabla\Phi(e_\delta)
=\delta^{\nu(p-1)}\bigl[\nabla\Phi(v)+O(\delta)\bigr].
$$

Multiplying this expression by
$w_0\delta^{-\alpha}[1+O(\delta)]$ and
${J^+_\delta}^{\!\top}={J^+_0}^{\!\top}+O(\delta)$ in the exact one-sided
chain rule proves the displayed expansion. Assumption D makes the leading
coefficient nonzero and therefore yields the relative bound. $\square$

**Sample-level prediction:** for two gaps $\delta_1, \delta_g$ on the **same** sample,

$$\boxed{\; a_{{\rm pred},i} \;=\; \left(\frac{\Delta_{g,i}}{\Delta_{1,i}}\right)^{\kappa}, \;}$$

using the **realized** gap ratio (§1.3; $=g^\kappa$ only when unclipped).

### Corollary 1 (homogeneous losses)
If $\Phi$ is exactly degree-$p$ homogeneous and
$w(\delta)\propto\delta^{-\alpha}$ exactly, the leading term carries exponent
$\kappa$ exactly; the full relation remains asymptotic because the forward
residual direction and online Jacobian may drift with $\delta$.
$a_{\rm pred}$ has no fitted coefficient once $(\nu,p,\alpha)$ are specified,
but the general exponent still depends on $\nu$.

---

## 4. Pseudo-Huber specialization (regime change)

ECT's Pseudo-Huber
$\Phi_c(e)=\sqrt{\|e\|_2^2+c^2}-c$ (matching `loss.py`) has two regimes:

| regime | condition | asymptotic form | effective $p$ | $\kappa$ |
|--------|-----------|-----------------|---------------|----------|
| small-residual | $\|e\|_2 \ll c$ | $\Phi_c(e)\approx \|e\|_2^2/(2c)$ | $2$ | $\nu - \alpha$ |
| large-residual | $\|e\|_2 \gg c$ | $\Phi_c(e)\approx \|e\|_2-c$ | $1$ | $-\alpha$ |

$$\boxed{\;\kappa_{\rm small} = \nu - \alpha, \qquad \kappa_{\rm large} = -\alpha.\;}$$

### Corollary 2 (ECT special case)
ECT's gap runs use $c=0$ (degree-1 Euclidean norm, $p=1$, $\alpha=1$):

$$\kappa_{\rm ECT} = \nu(1-1) - 1 = -\alpha = \boxed{-1}
\;\Rightarrow\; g^{\rm sg}_\delta \propto \delta^{-1}.$$

With $c=0$ the small/large distinction collapses (degree-1 Euclidean norm everywhere). The $1/g$ law
is the $p{=}1,\alpha{=}1$ point of the general law. **Crucially, at $p=1$ the exponent
is independent of $\nu$** — this is a near-zero-parameter special case, but it also
means the residual-expansion parameter $\nu$ is **not** probed by this configuration.

---

## 5. Existing evidence boundary (no new audit in this note)

**What is tested.** The aggregate/minibatch-gradient scalar

$$a_j^\star = \frac{\langle G_j^g,G_j^1\rangle}{\|G_j^1\|^2}$$

matches $a_{\rm pred}=g^{-1}=1/1.3=0.769$ to mean relative error 0.18% across
four checkpoints of **one** $q{=}128$ seed-3 trajectory. This is a **retrospective
consistency check along one trajectory**, not a sample-level measurement, multi-seed
validation, or multi-dataset validation. A sample-level $a_i^\star$ would require a
separate measurement and is outside this theory note.

**What it establishes (weak).** At $p=1$, $\nu$ drops out, so the existing 0.18% aggregate
match supports (i) the weight-ratio prediction ($1/g$) and (ii) consistency with local
aggregate training-derivative geometry. It does **not** constrain $\nu$, $v$, or $J^+_0$ — the
residual-expansion machinery (Assumption A) is idle at $p=1$.

**Stage-invariance is a schedule-construction fact.** global_sigmoid makes the
realized ratio $\Delta_g/\Delta_1=g$ (unclipped), so $a_{\rm pred}=g^\kappa$ is
$t$-independent by construction; the observed flatness confirms the schedule is
well-implemented, not a novel prediction.

**Training derivative ≠ optimizer update.** Theorem 1 and Proposition 3 concern
$g^{\rm sg}$ before optimizer state is applied. The existing $a_j^\star$ is an
aggregate/minibatch raw-gradient scalar, so it is not a sample-level test of
Theorem 1. No optimizer-history predictor, trajectory equivalence, or quality
mechanism follows from these statements.

---

## 6. Minibatch aggregation

### Proposition 2 (exact batch decomposition)
Let per-sample SG derivatives satisfy
$g^{\rm sg}_{g,i}=a_i g^{\rm sg}_{1,i}+r_i$, where $a_i$ is a sample-level
scalar and $r_i$ is the remaining non-scalar term. With
$G_1=\sum_i g^{\rm sg}_{1,i}$, $G_g=\sum_i g^{\rm sg}_{g,i}$, and any reference
scalar $\bar a$,

$$
\boxed{
G_g
=\bar a G_1
+\underbrace{\sum_i(a_i-\bar a)g^{\rm sg}_{1,i}}_{\text{heterogeneity term}}
+\underbrace{\sum_i r_i}_{\text{local non-scalar residual}}.
}
$$

This is an **exact algebraic identity**. The batch best-fit scalar
$a_{\rm batch}^\star = \langle G_g,G_1\rangle/\|G_1\|^2$ is the aggregate
least-squares scalar, distinct from the sample-level $a_i$; the two are related by
this decomposition.

### Heuristic 3 (emergent scalarity — NOT a theorem)
If $g^{\rm sg}_{1,i} \approx \mu + \xi_i$ with a non-zero common mean direction $\mu$ and the
heterogeneity term is approximately centered / weakly correlated across the batch,
then the batch signal accumulates as $O(n)$ while the heterogeneity fluctuation
grows as $\approx O(\sqrt{n})$, so the **relative** heterogeneity may decay with
batch size. Under such conditions, aggregate $G_g,G_1$ can show higher
scalarity (cosine) than individual
$g^{\rm sg}_{g,i},g^{\rm sg}_{1,i}$ — scalarity emerging through cancellation.

> **Status: sufficient-condition sketch, not a theorem.** The "cancellation" is a
> high-dimensional concentration heuristic (CLT-like), requiring iid + high-dim +
> weak-alignment assumptions that are not fully stated here. It can fail for small
> batches or strongly-aligned gradients. Counterexamples exist; treat as a
> hypothesis to test, not an established result.

### Bound (corollary of Prop 2)
$$
\|G_g-\bar aG_1\|
\le\sum_i|a_i-\bar a|\,\|g^{\rm sg}_{1,i}\|+\sum_i\|r_i\|.
$$

---

## 7. Exact ECT decomposition

The local theorem above is asymptotic. The implemented ECT loss also admits a
separate **exact**, finite-gap decomposition under its one-sided SG backward
rule. For one coupled sample, hold $(y,t,\varepsilon)$ and the online dropout
realization fixed, define

$$
D_t:=f_\theta(y+\varepsilon t,t),
\qquad
T_r:=
\begin{cases}
\operatorname{sg}\!\left[f_\theta(y+\varepsilon r,r)\right],&r>0,\\
y,&r=0,
\end{cases}
\qquad
e_r:=D_t-T_r,
\qquad
\Delta:=t-r>0,
$$

and let

$$
\psi_c(e):=\nabla_e\Phi_c(e)
=\frac{e}{\sqrt{\|e\|_2^2+c^2}}.
$$

For $c=0$, this is $\psi_0(e)=e/\|e\|_2$ on $e\neq0$. With
$J_t:=\partial_\theta D_t$, the implemented per-sample ECT loss and its
one-sided SG derivative are exactly

$$
\boxed{
\ell^{\rm ECT}(r,\Delta)=\frac{\Phi_c(e_r)}{\Delta},
\qquad
g^{\rm sg}(r,\Delta)
=\frac{1}{\Delta}J_t^\top\psi_c(e_r).
}
$$

This identity requires no small-gap expansion. It uses only the one-sided SG
backward rule, the external ECT denominator, and a matched comparison in which
the online branch is unchanged. All cross-cell identities below assume the same
current parameter value, data/noise realization, and coupled $D_t,J_t$; they do
not compare gradients from separately trained checkpoints.

### Proposition 3 (exact two-gap decomposition)

For baseline and probe targets $r_1=t-\Delta_1$ and
$r_g=t-\Delta_g$, set

$$
s_i:=\frac{\Delta_{1,i}}{\Delta_{g,i}},
\qquad
u_{1,i}:=\psi_c(e_{r_1,i}),
\qquad
u_{g,i}:=\psi_c(e_{r_g,i}).
$$

For each matched sample $i$,

$$
\boxed{
g^{\rm sg}_{g,i}
=s_i g^{\rm sg}_{1,i}
+\underbrace{\frac{1}{\Delta_{g,i}}J_{t,i}^\top
  (u_{g,i}-u_{1,i})}_{\text{target-induced correction}}.
}
$$

The first term is the **exact denominator rescaling**. The second contains the
entire change in target value and residual direction; it is not assumed to be
orthogonal to the first term. In the local $c=0$ regime with
$\Delta_g/\Delta_1$ fixed, Assumptions A–D imply
$u_g-u_1=O(\Delta_1)$, so the target-induced correction is lower by one
relative order than the $O(\Delta_1^{-1})$ leading term. This recovers the
$1/g$ law asymptotically while retaining the exact finite-gap remainder.

### Corollary 3 (exact target × denominator identities)

For the four gradient-level cells

| cell | target | denominator | per-sample SG derivative |
|------|--------|-------------|--------------------------|
| A | $r_1$ | $\Delta_1$ | $g_{A,i}=\Delta_{1,i}^{-1}J_{t,i}^\top u_{1,i}$ |
| B | $r_g$ | $\Delta_g$ | $g_{B,i}=\Delta_{g,i}^{-1}J_{t,i}^\top u_{g,i}$ |
| C | $r_g$ | $\Delta_1$ | $g_{C,i}=\Delta_{1,i}^{-1}J_{t,i}^\top u_{g,i}$ |
| D | $r_1$ | $\Delta_g$ | $g_{D,i}=\Delta_{g,i}^{-1}J_{t,i}^\top u_{1,i}$ |

the following identities hold exactly:

$$
\boxed{
g_{B,i}=s_i g_{C,i},
\qquad
g_{D,i}=s_i g_{A,i},
\qquad
g_{B,i}-g_{D,i}=s_i(g_{C,i}-g_{A,i}).
}
$$

Consequently, the per-sample factorial interaction is not a free residual:

$$
\boxed{
g_{{\rm int},i}
:=g_{B,i}-g_{C,i}-g_{D,i}+g_{A,i}
=(s_i-1)(g_{C,i}-g_{A,i}).
}
$$

For an unclipped global gap multiplier, $\Delta_{g,i}=g\Delta_{1,i}$ for every
sample and $s_i=1/g$ is constant. The same identities then hold after summing
over the minibatch: $G_B=g^{-1}G_C$ and $G_D=g^{-1}G_A$. Under clipping,
$s_i$ varies by sample; the per-sample identities remain exact, whereas a
single batch scalar cannot in general be pulled outside the sum.

These are identities for matched **training derivatives**, not for parameter
updates after an adaptive optimizer and not for finite-budget quality outcomes.

---

## 8. Failure regimes

The local theorem (not the exact identities in §7) loses its stated conclusion
when:

1. **Large gap** — the residual expansion is not accurate.
2. **Pseudo-Huber crossover** ($c>0$) — no single homogeneous degree $p$ applies
   across the crossover (§4).
3. **Clipping/boundary** — the path in $\delta$ is non-smooth. The per-sample
   §7 identities remain exact, but the constant-$1/g$ batch corollary does not.
4. **Degenerate leading SG term** —
   ${J^+_0}^{\!\top}\nabla\Phi(v)=0$ (Assumption D).
5. **Irregular online Jacobian** — $J^+_\delta-J^+_0$ is not $O(\delta)$.
6. **Two-sided differentiation** — allowing target-branch derivatives adds a
   term involving $\partial_{\bar\theta}F^-_{\bar\theta}$; Theorem 1 and §7 are
   not stated for that training rule.

---

## 9. Consequences and falsifiable checks

No empirical audit is performed or authorized by this revision. The statements
below classify what a future, separately scoped check would test.

| # | Statement | Status |
|---|-----------|--------|
| T1 | Pseudo-Huber crosses from $\kappa=\nu-\alpha$ to $\kappa=-\alpha$ between its small- and large-residual regimes | non-trivial local-theorem prediction |
| T2 | For $c=0$ and a fixed gap ratio, the exact target-induced correction in Proposition 3 is one relative order below the $O(\Delta^{-1})$ term | non-trivial local-theorem prediction |
| E1 | $g_{B,i}=s_i g_{C,i}$ and $g_{D,i}=s_i g_{A,i}$ in a matched one-sided SG derivative calculation | exact ECT identity, not an empirical hypothesis |
| E2 | $g_{{\rm int},i}=(s_i-1)(g_{C,i}-g_{A,i})$ | exact ECT identity, not an independent interaction model |
| H1 | Aggregate scalarity may exceed sample scalarity under the centering/concentration conditions in Heuristic 3 | heuristic only |

**Weakly supported (do NOT test the $\nu$-content):**

| # | Prediction | Status |
|---|-----------|--------|
| W1 | aggregate gap scaling $a_j^\star/a_{\rm pred}\approx 1$ | weakly checked (0.18%, one q=128 seed-3 trajectory; $\nu$ idle at $p=1$) |
| W2 | stage invariance | schedule-construction fact |

Only T1–T2 probe the non-trivial local asymptotic content. E1–E2 are algebraic
consequences of the implemented loss and one-sided backward rule. W1–W2 are weak
consistency checks and do not validate the residual-order assumption.

---

## Claim boundaries (must preserve)

**CAN say:**
- "For the one-sided SG training derivative, a degree-$p$-homogeneous outer loss and power-law weighting $\delta^{-\alpha}$ give $g^{\rm sg}_\delta\sim\delta^{\nu(p-1)-\alpha}$ under Assumptions A–E."
- "ECT admits the exact finite-gap decomposition $g_g^{\rm sg}=s g_1^{\rm sg}+\Delta_g^{-1}J_t^\top(u_g-u_1)$; only the first term is exact denominator rescaling."
- "ECT's local $1/g$ law is the $p=1,\alpha=1$ point. The target-induced correction is lower order under the stated local assumptions, not identically zero at finite gap."
- "ECT's aggregate/minibatch raw-gradient scalar $a_j^\star\approx0.77$ matches the implied $a_{\rm pred}=g^{-1}=0.769$ (0.18%) along one q=128 seed-3 trajectory. This is a **weak aggregate consistency check**, consistent with the weight-ratio prediction and small aggregate directional deviation; it is not a sample-level validation of the residual-order or Jacobian assumptions."
- "The batch decomposition (Proposition 2) is exact; emergent scalarity (Heuristic 3) is a sufficient-condition sketch."

**CANNOT say:**
- "The theorem covers the ordinary gradient of $\widetilde\ell(\theta,\theta)$ or a symmetric/two-sided training rule."
- "ECT has the exact finite-gap identity $g_g^{\rm sg}=g^{-1}g_1^{\rm sg}$" — this requires the target-induced correction to vanish; generally only the denominator component has that exact scaling.
- "The 0.18% match validates the residual-expansion theorem / its non-trivial content" — $\nu$ is idle at $p=1$; T1–T2 remain untested.
- "The gradient scaling theorem implies identical optimization trajectories" — adaptive optimization can make them non-equivalent (§5).
- "The exact ECT decomposition proves the conditional local assumptions A–E" — the decomposition is algebraic; the residual-order statement remains conditional.
- "Gradient scaling causes the FID improvement" — **no FID-causality claim.**

---

## Open / next (theory only)

- An explicit numerical constant in
  $\|g^{\rm sg}_\delta-C\delta^\kappa\|
  \le C_1\delta^{\kappa+1}$ requires quantitative remainder and Jacobian bounds;
  the present theorem states only the asymptotic order.
- Empirical audits, long-budget behaviour, optimizer-history predictors, and
  quality causality are outside this revision.

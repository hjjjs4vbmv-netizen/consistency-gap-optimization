# True stop-gradient ECT toy: exact dynamics and honest verdict

Date: 2026-08-05. Branch: `theory/true-sg-operator`.
This extends (does not modify) the PR #33 abstract-stochastic-oracle story to the
**real stop-gradient ECT toy**: the exact second-moment dynamics, the heavy-tail
instability, and an honest assessment of whether the finite-horizon main line is
supported by the true operator.

---

## 1. Derivation

Model: $f_\beta(x_t,t)=z(1+\beta_1 t+\beta_2 t^2)$, $x_t=m(t)z$, $z\sim\mathcal N(0,\sigma_d^2)$.
Stop-gradient loss $L=\tfrac12(f_t-\mathrm{sg}\,f_r)^2$, $r=t-\Delta$, $\Delta=\min(g\,\delta_0,\,t-t_{\min})$.

- residual: $f_t-f_r = z\,v_g(t)^\top\beta$, $v_g=[\Delta,\,t^2-r^2]^\top$
- online Jacobian: $J_t=[t,\,t^2]^\top$
- per-sample gradient: $g_s=z^2(v_g^\top\beta)\,J_t$
- random update matrix: $Q_g(z,t)=I-\eta z^2 J_t v_g(t)^\top$

**Three operators, distinguished explicitly:**
| operator | definition | g-scaling (measured) | role |
|---|---|---|---|
| $H_g$ | $\sigma_d^2\mathbb{E}[v_g v_g^\top]$ | $\|H_g\|\propto g^2$ (1.3→8.9) | forward-loss curvature |
| $A_g$ | $\sigma_d^2\mathbb{E}[J_t v_g^\top]$ (asymmetric) | $\|A_g\|\propto g$ (337→877) | mean stop-grad update |
| $\mathcal T_g$ | $\mathbb{E}[Q_g\otimes Q_g]$ (3×3 on symmetric basis) | $\rho(\mathcal T_g)\gg1$ | second-moment recursion |

**Exact second-moment recursion** (exact for resampled $(z,t)$ SGD):
$$M_{k+1}=M_k-\eta(A_gM_k+M_kA_g^\top)+3\eta^2\sigma_d^4\,\mathbb{E}_t\big[(v_g^\top M_k v_g)\,J_t J_t^\top\big],\quad E_K=\mathrm{Tr}(M_K).$$
Verified single-step against MC (theory 6.67e-4 vs MC 6.21e-4, <5%, 50k traj).

---

## 2. Stability: mean-square instability (corrected language)

In the report's parameterization ($t\sim\mathrm{LogNormal}(-1.1,2)$, clip $t_{\max}=100$,
un-normalized $t^2$ Jacobian), all moments are finite (t is bounded, z is Gaussian), so
$\mathbb{E}\|\beta_K\|^2<\infty$ for every finite $K$. The correct statement is:

> **The bare polynomial toy becomes asymptotically mean-square unstable for part of the
> tested gap range despite a contractive mean update.**

Measured: $\rho(\mathcal T_g)$ grows from $1.0000$ (g=0.5) to $1.0221$ (g=1.5) at the
reference $\eta$. Since $\rho(\mathcal T_g)>1$, the second moment grows exponentially in
$K$ (e.g. $E_{200}(g{=}1.5)\approx8\times10^{-3}$ vs $E_{20}\approx2.6\times10^{-4}$), not
because "no finite second moment exists" but because the mean operator $A_g$ can be
contractive while the second-moment operator $\mathcal T_g$ is not.

This is a real physical finding: the stop-gradient update's noise variance scales with
the quartic moment of the online Jacobian, which the un-normalized $t^2$ parameterization
makes heavy-tailed. Real ECT normalizes $t$ via the $\sigma_d^2+t^2$ preconditioner; a
faithful toy should too (deferred to a separate study, per review).

---

## 3. Finite-horizon scan — EXACT recursion (review: main result must be exact, not MC)

`theory/true_sg_horizon.csv`, `figures/true_sg_error_vs_g_budget.pdf`, generated with the
exact 3×3 recursion $M_{K+1}=\mathcal T_g M_K$ (no MC; MC used only as a single-step
sanity check, §6).

Three modes (all with tiny base $\eta$ so the second moment stays finite at these $K$):

| mode | g* (all K) | spread (K=200) | error vs g |
|---|---|---|---|
| **fixed** $\eta=\eta_1$ | 0.5 | 51 | **monotone ↑** |
| **lr_match_H** $\eta_g\propto 1/\lambda_{\max}(H_g)\propto 1/g^2$ | 1.5 | 5013 | **monotone ↓** |
| **lr_match_A** $\eta_g=\eta_1/a(g)$, $a(g)=\langle A_g,A_1\rangle_F/\|A_1\|_F^2$ | (flat) | **≈0 (1e-6)** | **flat** |

**The A-matched control is the decisive one.** The mean update is governed by $A_g$
($\mathbb{E}[\beta_{k+1}]=(I-\eta A_g)\mathbb{E}[\beta_k]$), and $\|A_g\|\approx g\|A_1\|$
while $\|H_g\|\approx g^2\|H_1\|$. Matching the *correct* operator ($A_g$, not $H_g$)
via the Frobenius-optimal scalar makes error-vs-g **flat to ~1e-6**.

> **In the bare polynomial stop-gradient toy, gap is almost exactly equivalent to an
> optimizer-step rescaling.** The fixed/H-matched "opposite directions" are artifacts of
> *not* matching the true mean operator.

---

## 4. Honest verdict on the finite-horizon main line

> The true 2D stop-gradient toy does **not** support an independent finite-horizon gap
> geometry: after the correct optimizer matching ($A$-matched), the gap dependence
> essentially vanishes. The toy-level claim is the **negative theorem**:
> **gap ≈ optimizer-step rescaling in the simplest stop-gradient linear model.**

**What this means for ICLR (trivial-rescaling-equivalence vs deep residual effects):**
1. theory: prove the equivalence in the simple model (this PR);
2. measure in a real deep network whether gap produces directional / layer / noise-structure
   changes that scalar matching cannot remove;
3. only if a real residual effect exists, continue the optimizer-aware-selector story.

The PR #33 abstract-separation required a *constructed* $\Sigma_g(g)$; the true
operator's $\Sigma_g$ is $\beta$-coupled and heavy-tailed — it drives instability
(mean-square) and, under correct matching, flatness, not an interior optimum.

---

## 5. Deliverables
- `theory/true_sg_operator.py` — derivation, exact recursion, A-matched control, scan.
- `theory/true_sg_horizon.csv` — exact $E_K(g)$ for fixed / lr_match_H / lr_match_A, $K\in\{20,50,100,200\}$.
- `theory/true_sg_operators.csv` — $\|A_g\|,\|H_g\|,\mathrm{Tr}(\cdot)$ g-scaling diagnostics.
- `figures/true_sg_error_vs_g_budget.pdf` — exact curves: monotone ↑/↓/flat.
- `theory/test_true_sg.py` — 4 unit tests (single-step recursion vs MC, matrix-power vs
  iterative, no cumulative-step bug, A-matched flatness), all pass.

## 6. MC role and sanity check
MC is used only as a single-step / short-K sanity check (heavy-tail draws make long-K MC
unreliable — 1500 trajectories under-sample the tail). Single-step analytic recursion vs
200k-trajectory MC agrees within ~17% (high MC variance under the heavy tail); the exact
recursion is the primary quantity everywhere else.

---

## 5. Deliverables
- `theory/true_sg_operator.py` — derivation + exact recursion + MC + scan.
- `theory/true_sg_horizon.csv` — MC $E_K(g)$ for fixed / lr_match_H, $K\in\{20,50,100,200\}$.
- `theory/true_sg_operators.csv` — $\|A_g\|,\|H_g\|,\mathrm{Tr}(\cdot)$ g-scaling diagnostics.
- `figures/true_sg_error_vs_g_budget.pdf` — monotone curves, opposite directions.

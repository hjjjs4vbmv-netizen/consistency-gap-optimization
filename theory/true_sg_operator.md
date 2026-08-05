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

## 2. Heavy-tail instability (the honest headline)

In the report's parameterization ($t\sim\mathrm{LogNormal}(-1.1,2)$, clip $t_{\max}=100$,
un-normalized $t^2$ Jacobian):

- $J_t=[t,t^2]$ gives $\mathbb{E}[J_2^2]\approx3.5\times10^5$ and joint quartic moments
  $\mathbb{E}[v_2^2J_2^2]\approx1.7\times10^9$; $\rho(\mathcal T_g)\gg1$ for any reasonable $\eta$.
- The **exact second moment diverges** at moderate $K$ (MC confirms: trajectories that
  hit a large-$t$ sample explode). The mean operator $A_g$ can be contractive while the
  second moment still blows up (variance instability).
- **Implication**: in the bare-$t^2$ parameterization the true stop-gradient toy has
  **no finite second moment at finite $K$** for practical $\eta$. Only a tiny
  $\eta\lesssim 10^{-2}/\rho(A_1)$ keeps $E_K$ finite at short $K$.

This is a **real physical finding**, not a bug: the stop-gradient update's noise
variance scales with the quartic moment of the online Jacobian, which the un-normalized
$t^2$ parameterization makes heavy-tailed. Real ECT normalizes $t$ via the
$\sigma_d^2+t^2$ preconditioner; a faithful toy should too (open item).

---

## 3. Finite-horizon scan at stable $\eta$ (MC, $K\in\{20,50,100,200\}$)

`theory/true_sg_horizon.csv`, `figures/true_sg_error_vs_g_budget.pdf`.

Two modes (both with tiny base $\eta$ to stay finite):

| mode | g* (K=200) | error vs g | direction |
|---|---|---|---|
| **fixed** $\eta=\eta_1$ | 0.50 | 1.77e-4 → 8.10e-4 | **monotone ↑** (small g best) |
| **lr_match_H** $\eta_g\propto 1/\lambda_{\max}(H_g)\propto 1/g^2$ | 1.50 | 1.80e-3 → 1.99e-4 | **monotone ↓** (large g best) |

**Key observation**: the two normalizations give **opposite** optimal directions for g.
- fixed: bigger g → bigger effective rate AND bigger heavy-tail noise → noise dominates → small g.
- lr_match_H: bigger g → much smaller $\eta^2\Sigma$ ($\propto 1/g^4$) → noise suppressed → large g.

The curves are **monotone** in both modes; there is **no internal optimum** and no
`g_{K1}* != g_{K2}*` crossover (within the finite, stable regime).

---

## 4. Honest verdict on the finite-horizon main line

> **In the true stop-gradient ECT toy, the effect of g is entirely an
> effective-learning-rate × effective-noise-amplitude rescaling.**
> Error-vs-g is monotone; its *direction* flips under learning-rate normalization;
> there is no gap-specific geometry decoupled from $(\eta,\Sigma)$ rescaling.
> The finite-horizon separation shown in PR #33's abstract counterexample required
> a *constructed* $g$-dependence of $\Sigma_g$; the true operator's $\Sigma_g$ is
> $\beta$-coupled and heavy-tailed, which drives instability rather than an interior
> optimum.

**What this means for the main line:**
- The "ADCM-insufficiency" story (same instantaneous criterion, different finite-step
  optimum) **does not survive in the real 2D stop-gradient toy** — the honest toy-level
  claim is negative.
- The finite-horizon main line, if it is to be supported, needs either (a) a toy whose
  $\Sigma_g$ genuinely differs from $H_g$-scaling in a way that survives $\eta$ matching
  (not the bare-$t^2$ parameterization), or (b) a deep-network diagnostic where the
  per-layer/preconditioned $\Sigma_g$ structure matters (open).
- A constructive next step: parameterize with the ECT preconditioner
  ($\tilde J_t=[t/(\sigma_d^2+t^2),\,t^2/(\sigma_d^2+t^2)^2]$-style) so the second moment
  is finite, then re-run the same scan. Prediction: with the heavy tail removed, the
  residual g-effect may still be a pure rescaling — in which case the toy line is closed.

---

## 5. Deliverables
- `theory/true_sg_operator.py` — derivation + exact recursion + MC + scan.
- `theory/true_sg_horizon.csv` — MC $E_K(g)$ for fixed / lr_match_H, $K\in\{20,50,100,200\}$.
- `theory/true_sg_operators.csv` — $\|A_g\|,\|H_g\|,\mathrm{Tr}(\cdot)$ g-scaling diagnostics.
- `figures/true_sg_error_vs_g_budget.pdf` — monotone curves, opposite directions.

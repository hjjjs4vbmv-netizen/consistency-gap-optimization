# Same-Information-Budget Predictor Comparison (P1, revised) — Report

**Role C (theory) · P1 · 2026-08-19**
**Question:** under a uniform information budget (G^1 + the scalar projection
a\*_j only, never the full G^g), which optimizer-response model best explains
the observed update divergence?

---

## 0. Protocol (fair comparison)

**Ground truth:** the real update ratio h_{t,i} = U^g_{t,i}/U_{t,i}, where U^g =
replay(G^g) and U = replay(G^1) through the real RAdam recursion. The real G^g
is used ONLY to compute the target h and the scalar a\*_j — no predictor sees
the full G^g.

**Three predictors, same information budget {G^1, a\*_j}_{j≤t}:**

| Predictor | Information used | Optimizer-response model |
|-----------|------------------|--------------------------|
| A. global scalar | G^1 + causal mean ā_t = mean(a\*_{j≤t}) | Ĝ = ā_t·G^1, replay RAdam |
| B. local/continuous | G^1 + {a\*_j}_{j≤t} | first-order: ĥ = 1 + A^(1) − A^(2) |
| C. discrete replay | G^1 + {a\*_j}_{j≤t} | Ĝ_j = a\*_j·G^1, replay RAdam |

A uses a single causal scale; B uses a closed-form first-order approximation
(Fernández-Hernández 2026, arXiv:2601.21739); C uses the full per-step scale
history through the real finite-history RAdam recursion. **No oracle**: the
old `h_replay = h_actual` identity is removed.

**Two regimes:** fresh (m0=0, v0=0, step0=0) and real (m0/v0/step0 from the frozen
checkpoint). **Metrics:** weighted R², Corr, wRMSE, plus Var_w(h_actual).

---

## 1. Results — R² / Corr at h=20, both regimes

| K | regime | global R² | local R² | discrete R² | Var_w(h) |
|---|--------|-----------|----------|-------------|----------|
| 32  | fresh | −0.420 | −0.375 | −0.410 | 2.9e-5 |
| 32  | real  | **0.849** | −17.84 | **0.863** | 1.2e-3 |
| 64  | fresh | −0.890 | −0.685 | −0.631 | 1.3e-4 |
| 64  | real  | **0.907** | −9.67 | **0.913** | 2.5e-3 |
| 128 | fresh | −0.051 | −0.072 | −0.108 | 2.9e-4 |
| 128 | real  | **0.923** | −94.07 | **0.918** | 3.0e-3 |
| 256 | fresh | −0.123 | −0.100 | −0.101 | 7.8e-4 |
| 256 | real  | **0.691** | −21.95 | **0.722** | 3.2e-3 |

Corr at h=20, real regime: global 0.93–0.83, discrete 0.94–0.85, local ≈0.

---

## 2. Findings (honest)

**F1 — In the real-history regime, the global scalar and the discrete replay are
STATISTICALLY TIED; both explain most of the divergence (R² 0.69–0.92).** The
per-step scale history adds at most ΔR² ≤ 0.03 over a single causal mean, and the
ordering is NOT robust: discrete wins on R² at k32/64/256 but **loses at k128**
(0.918 vs 0.923), while Corr has discrete marginally higher at all four (but by
≤0.02, and contradicting the R² ranking at k128). So under the (near-constant)
scale history of this gap, the discrete per-step history provides **no robust
additional explanatory power** over a single causal scalar. The two are equivalent
for practical purposes; the discrete replay's only robust advantage is not
blowing up under incomplete history (F2).

**F2 — The local/continuous first-order predictor CATASTROPHICALLY fails in the
real-history regime (R² = −18 to −94).** This is not "slightly worse" — it is
divergent. Cause: the first-order formula is a perturbative expansion valid only
when the full history is available; with only 20 steps of gradient history and a
converged optimizer state (m0/v0 carry pre-checkpoint memory the formula cannot
see), the A^(1)−A^(2) terms blow up. (Verified separately: the first-order
formula is correct for exact-scalar gaps from a fresh start, Corr~0.9998 — so
the failure is the truncated-history regime, not a bug.)

**F3 — The fresh-start regime is NOT informative for predictor comparison.**
Var_w(h) is 2.9e-5 to 7.8e-4 — the target is near-constant (h≈1, gradient-scale-
invariant), so R² is meaningless (all predictors negative regardless of quality).
This is exactly why Var_w(h) is reported: it shows the fresh-start R² values are
an artifact of a near-constant target, not a predictor failure. **The real-history
regime is the regime that matters.**

**F4 — The discrete replay and global scalar are tied because a\* is approximately
stable (std ≈ 0.009–0.023, from the cross-K experiment). When the scale history is
nearly constant, a single causal mean captures almost as much as the full per-step
history — so the per-step scale history carries little independent information
beyond its mean. The discrete replay would gain a robust edge only under a more
time-varying scale history (e.g. a stronger gap); for this gap it does not.

---

## 3. Answer to the core question

**Under a uniform information budget, does the discrete scalar-history replay
explain more of the optimizer-response variation than simpler scalar
approximations?** **No robust advantage for this gap.** In the real-history regime
the discrete replay and the global scalar are statistically tied (both R² 0.69–0.92;
the per-step history's ΔR² ≤ 0.03 and the R² ordering is not robust — it flips at
k128). Both substantially beat the local/continuous first-order theory (which
fails catastrophically, R² ≪ 0). The discrete replay's only robust advantage over
the alternatives is **robustness**: unlike the first-order formula, it does not
diverge when the gradient history is truncated relative to the optimizer's memory.
For this near-constant scale history, the per-step scale history carries no
independent information beyond its mean.

---

## 4. Claim boundaries (must be preserved)

**We CAN say:**
- "Under a uniform information budget (G^1 + a\*), the discrete scalar-history
  replay and the global scalar both explain most of the update-divergence
  variance in the real-history regime (R² 0.69–0.92); they are statistically
  tied (the per-step history's ΔR² ≤ 0.03 is not robust — the R² ordering flips
  at k128)."
- "The continuous first-order theory fails in the real-history regime (truncated
  history + converged state); the discrete replay is robust to this."
- "The fresh-start regime is uninformative (target near-constant); this is shown
  by Var_w(h), not inferred."

**We CANNOT say:**
- "The discrete replay fully explains the optimizer divergence" — R² < 0.92, and
  the residual is non-scalar.
- "The first-order theory is always wrong" — it is valid for exact-scalar gaps
  from a fresh start (verified, Corr~0.9998); it fails in the truncated-real-history
  regime.
- "Any predictor causes the FID improvement" — no FID is measured. **No
  FID-causality claim is made.**

---

## 5. Artifacts

- `analysis/predictor_comparison/summary.json` — R²/Corr/wRMSE/Var_w per predictor per horizon per K per regime.
- `analysis/predictor_comparison/provenance.json` — protocol, method, environment.
- `figures/predictor_comparison/fig1_R2_vs_stage`, `fig2_R2_vs_horizon`, `fig3_Var_w_vs_stage`, `fig4_corr_vs_stage` (PDF+PNG).

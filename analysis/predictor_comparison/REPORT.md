# Three-Predictor Comparison (P1) — Report

**Role C (theory) · P1 · 2026-08-17**
**Question:** how much more quantitative explanatory power does our discrete
finite-history characterization provide over the generic continuous first-order
theory?

---

## 0. Protocol

Three predictors of the update ratio h_{t,i} = U^g_{t,i}/U_{t,i} (g=1.3 vs g=1.0),
all replayed from a **uniform fresh start** (m0=0, v0=0, step0=0) over the same real
paired gradient sequence {G^1.0, G^1.3}_{t=1..20} from the frozen #49 Arm-A checkpoints:

1. **current scalar predictor** (cross-K): a\*_j = ⟨G^g_j, G_j⟩/‖G_j‖², Ĝ_j = a\*_j·G_j,
   replay RAdam → ĥ^scalar = replay(Ĝ)/u1.
2. **2026 first-order scale-lag predictor** (arXiv:2601.21739, "Why Adam Works Better
   with beta1=beta2"; first-order moment-memory expansion): δ_j = a\*_j − 1,
   A^(1)_t = (Σ_{j≤t} β1^{t−j} δ_j G_j)/(Σ_{j≤t} β1^{t−j} G_j),
   A^(2)_t = (Σ_{j≤t} β2^{t−j} δ_j G_j²)/(Σ_{j≤t} β2^{t−j} G_j²),
   ĥ^firstorder = 1 + A^(1) − A^(2).
3. **discrete finite-history replay** (our exact characterization): ĥ^replay =
   replay(G^g)/u1 — the exact moment recursion, the reference (R²≈1).

Fresh start is used so the first-order formula is **complete**: the full history is the
20-step gradient history. (The real extracted state would encode pre-checkpoint history
that the first-order formula cannot see.) Metrics: weighted R², Corr, wRMSE of each
predictor vs the actual h, per horizon, per K.

---

## 1. Results — R² and Corr at h=20 (fresh start)

| K | scalar R² | firstorder R² | replay R² | scalar Corr | firstorder Corr | replay Corr |
|---|-----------|---------------|-----------|-------------|-----------------|-------------|
| 32  | −0.410 | −0.375 | **1.000** | −0.249 | −0.282 | **1.000** |
| 64  | −0.631 | −0.685 | **1.000** | −0.016 | −0.022 | **1.000** |
| 128 | −0.108 | −0.072 | **1.000** | 0.051 | 0.051 | **1.000** |
| 256 | −0.101 | −0.100 | **1.000** | 0.029 | 0.026 | **1.000** |

**Both the scalar and the first-order predictors have NEGATIVE R² and near-zero Corr at
every stage; the finite-history replay is exact (R²=1.0, Corr=1.0).**

---

## 2. Findings (honest)

**F1 — In the fresh-start regime, the finite-history replay provides essentially ALL the
explanatory power; the generic first-order theory provides none.** The scalar and
first-order predictors are worse than predicting the mean h (negative R²) and have
essentially zero correlation with the actual h. Only the exact discrete replay captures
the update discrepancy.

**F2 — Why: gradient-scale-invariance.** In the fresh-start regime, the actual update
ratio is near-constant (h ≈ 1.01, from the balanced-β finding): RAdam's sqrt(v)
normalization absorbs the constant gradient scale (a\* ≈ 0.77). The scalar predictor
predicts h ≈ a\* ≈ 0.77 (systematically wrong — the scale is already absorbed), and the
first-order predictor predicts a wide coordinate distribution (from the A^(1)−A^(2)
G-weighting) that does not materialize. Neither captures the near-constant actual h.

**F3 — The scalar predictor's cross-K success (R²=0.86) is regime-specific.** It works in
the real-state regime (moments converged, update ratio tracks the gradient scale) but
fails in the fresh-start regime. The first-order predictor is incomplete in the real-state
regime (only 20 steps of gradient history; the pre-checkpoint history encoded in m0/v0 is
invisible to the closed-form formula), so the fresh-start regime is the only complete
three-way comparison.

**F4 — The finite-history characterization is not a "better approximation" of the
first-order theory; it is a different object.** The exact replay is the mechanism itself
(R²=1.0); the first-order theory is a perturbative approximation that fails outside its
validity regime (constant-ish scale, where the spurious coordinate variation dominates).

---

## 3. Answer to the core question

**How much more explanatory power does finite-history provide over first-order theory?**
In the fresh-start regime, the finite-history replay is exact (R²=1.0) while the first-order
theory fails (R² < 0). The finite-history characterization provides essentially **all** the
explanatory power; the generic continuous first-order theory provides **none** in this
regime. This is a strong negative result for the first-order theory as a standalone
predictor of the update discrepancy.

---

## 4. Claim boundaries (must be preserved)

**We CAN say:**
- "In the fresh-start regime, the exact finite-history replay explains the update
  discrepancy (R²=1.0), while both the scalar and first-order scale-lag predictors fail
  (negative R², near-zero Corr)."
- "The failure is due to gradient-scale-invariance: the actual update ratio is
  near-constant, which neither the scalar nor the first-order theory captures."
- "The scalar predictor's cross-K success is regime-specific (real-state)."

**We CANNOT say:**
- "The first-order theory is always wrong" — it is validated in its own regime (the theory
  doc's T2 test, exact-scalar time-varying δ).
- "Balancing β or any predictor causes the FID improvement" — no FID is measured here.
- "The finite-history replay is a novel predictor" — it is the exact mechanism, not a
  predictor.

**Why:** this is a controlled replay comparison in the fresh-start regime. It shows the
first-order theory does not explain the update discrepancy there, and that the exact
discrete replay does. It does not establish any causal claim about ECT's quality outcomes.
**No FID-causality claim is made.**

---

## 5. Artifacts

- `analysis/predictor_comparison/summary.json` — R²/Corr/wRMSE per predictor per horizon per K.
- `analysis/predictor_comparison/provenance.json` — protocol, method, environment.
- `figures/predictor_comparison/fig1_R2_vs_stage`, `fig2_R2_vs_horizon`, `fig3_corr_vs_stage` (PDF+PNG).

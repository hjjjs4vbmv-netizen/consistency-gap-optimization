# Balanced-β Intervention on Gap-Induced RAdam Divergence — Report

**Role C (theory) · P0 · 2026-08-17**
**Question:** does the gap-induced R_opt — the non-scalar residual between the
g=1.3 and g=1.0 RAdam updates — shrink when β1 = β2?

---

## 0. Protocol (controlled, per collaborator)

- **Same real paired gradient sequence** {G_t^1.0, G_t^1.3}_{t=1..20} from the frozen
  #49 Arm-A checkpoints (k32/k64/k128/k256, same seed-3 trajectory).
- **Uniformly defined starting point** for every β config: m0=0, v0=0, step0=0.
  The real extracted state is **never** used, so no "old-β history + new-β step" mixing.
- **β configs:** `standard` (0.9, 0.999) vs balanced (0.9, 0.9), (0.99, 0.99), (0.999, 0.999).
- **Metrics per config per horizon t:** R_opt (headline, over the full vector),
  Disp(h) and h_i stats (on the top-1% effective support by |u1|), Corr(u1, ug).
- **Core prediction:** R_opt^{β1=β2} < R_opt^{0.9,0.999}. Reported as observed, not asserted.

> **Note on the effective support.** Fresh-start updates span many orders of magnitude
> across β configs (the β2=0.999 rectification transient damps updates to ~1e-7), so an
> absolute threshold would empty the support. R_opt is scale-invariant (the rect factor
> is a global scalar that cancels in u1/ug — verified numerically: exact-scalar gradients
> give R_opt~1e-7 for both standard and balanced configs), so it is computed over the full
> vector; the top-1% quantile mask (via np.partition) is used only for the h_i/Disp
> statistics, and matches the cross-K effective support (~1% of coords).
>
> **Regime note.** This experiment replays from a *fresh* start (m0=0, v0=0, step0=0), so
> its R_opt values (0.04–0.07) are **not directly comparable** to the cross-K R_opt
> (0.09–0.12), which replayed from the real converged state. The two answer different
> questions: cross-K measured the divergence in the real accumulated history; this
> experiment measures the divergence under a controlled β intervention. The replay uses the
> same pure-numpy float32 RAdam validated against stored torch histories to ~1 ulp in the
> cross-K experiment (verify_u1_final); the fresh-start case is the same radam_step with
> step0=0.

---

## 1. Results — R_opt at h=20 (headline)

| K (kimg) | standard (0.9,0.999) | β=0.9 | β=0.99 | β=0.999 |
|----------|----------------------|-------|--------|---------|
| 32  | 0.0493 | 0.0497 ✗ | **0.0460** ✓ | **0.0456** ✓ |
| 64  | 0.0419 | **0.0417** ✓ | 0.0431 ✗ | 0.0432 ✗ |
| 128 | 0.0561 | 0.0571 ✗ | **0.0529** ✓ | **0.0523** ✓ |
| 256 | 0.0745 | 0.0748 ✗ | **0.0623** ✓ | **0.0609** ✓ |

✓ = balanced < standard (prediction holds); ✗ = prediction fails.

**Prediction check (12 comparisons):** 7 hold, 5 fail.

---

## 2. Findings (honest)

**F1 — The prediction is PARTIALLY confirmed, not uniformly.** The high-β balanced
configs (0.99, 0.999) reduce R_opt at 3 of 4 stages (k32, k128, k256); the short-memory
balanced config (0.9) mostly does not (it slightly *increases* R_opt at k32/k128/k256).
At k64 all balanced configs fail or are marginal.

**F2 — The strongest confirmation is at the final checkpoint (k256).** There the
gap-induced divergence is largest (R_opt=0.0745) and balanced_0.999 reduces it to 0.0609
(−18%). This is the stage where the cross-K experiment found the *lowest* scalar-history
explanatory power — the two results are complementary: at k256 the divergence is largest
and most reducible by β-balancing.

**F3 — The effect is monotone in β2 within the balanced family at k32/k128/k256**
(0.9 → 0.99 → 0.999 gives decreasing R_opt), but inverted at k64. The benefit of matching
β1=β2 requires a sufficiently long moment memory; short-memory balancing (β=0.9) does not
help. This is consistent with the "Why Adam Works Better with beta1=beta2" first-order
theory (arXiv:2601.21739): the scale-history effect needs a long enough v memory to matter.

**F4 — Corr(u1,ug) is high (≥0.997) for all configs** — the updates are near-scalar in
direction; R_opt captures the residual that β-balancing can shrink.

**F5 — The updates are near-scale-invariant: h_mean ≈ 1.01, not the gradient scalar
a\* ≈ 0.77.** RAdam's sqrt(v) normalization cancels a *constant* gradient scale: for an
exact-scalar pair G^1.3 = 0.77·G^1.0, the replayed updates are near-equal (h_mean=1.00000,
R_opt~1e-7, verified numerically). The real data show h_mean ≈ 1.00–1.01 across stages —
so the gap-induced R_opt (0.04–0.07) arises from the **non-constant** part of the scale
history (the per-step δ_j variation), not from the constant 0.77 scale. This is exactly the
gradient-scale-invariance regime of the "Why Adam Works Better with beta1=beta2" theory
(arXiv:2601.21739), and it is why β-balancing can shrink the residual.

---

## 3. Answer to the core question

**Does balancing β1=β2 reduce the gap-induced R_opt?** **Partially, and stage-dependently.**
Long-memory balanced configs (β=0.99, 0.999) reduce R_opt at most stages and most strongly
at the final checkpoint (−18%); short-memory balancing (β=0.9) does not. The prediction is
**not uniformly confirmed** — this is an honest, informative falsification of the strong
form of the claim, and a partial confirmation of the mechanism-consistent direction.

---

## 4. Claim boundaries (must be preserved)

**We CAN say:**
- "Balancing β1=β2 with long moment memory (0.99, 0.999) reduces the gap-induced R_opt at
  most training stages, most strongly at the final checkpoint (−18% at k256)."
- "The reduction is stage- and β-dependent; short-memory balancing (0.9) does not help."
- "This is consistent with the gradient-scale-history mechanism, but is not a proof of it."

**We CANNOT say:**
- "Balancing β causes the FID improvement" — this experiment never trains or measures FID.
- "We have proven the gap effect enters through gradient-scale history" — a controlled
  replay intervention is evidence, not causal identification.
- "Balanced β is uniformly better" — the data show it is not (k64, and β=0.9).

**Why:** this is a *replay* intervention on frozen gradients, not a training intervention.
It shows the divergence R_opt is sensitive to β1=β2 balancing in a mechanism-consistent
direction, but does not establish that this divergence drives ECT's quality outcomes.
**No FID-causality claim is made.**

---

## 5. Artifacts

- `analysis/balanced_beta/summary.json` — full R_opt/Disp/Corr/h_i per config per horizon per K.
- `analysis/balanced_beta/provenance.json` — protocol, seeds, method, environment.
- `analysis/balanced_beta/{k32,k64,k128,k256}/raw_h20/{config}_h_actual.npy` — h_i on the
  effective support at h=20, per config (auditability).
- `figures/balanced_beta/fig1_Ropt_vs_beta`, `fig2_Ropt_vs_horizon`, `fig3_disp_corr_h20` (PDF+PNG).

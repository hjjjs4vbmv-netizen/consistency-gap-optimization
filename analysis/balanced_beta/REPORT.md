# Balanced-β Intervention on Gap-Induced RAdam Divergence — Report

**Role C (theory) · P0 · 2026-08-17**
**Question:** does the gap-induced R_opt — the non-scalar residual between the
g=1.3 and g=1.0 RAdam updates — shrink when β1 = β2?

---

## 0. Protocol

- **Same real paired gradient sequence** {G_t^1.0, G_t^1.3}_{t=1..20} from the frozen
  #49 Arm-A checkpoints (k32/k64/k128/k256, same seed-3 trajectory).
- **Uniformly defined starting point** for every β configuration: m0=0, v0=0, step0=0.
  The real extracted state is **never** used, so there is no "old-β history + new-β step"
  mixing.
- **β configurations:** `standard` (0.9, 0.999) versus balanced (0.9, 0.9),
  (0.99, 0.99), and (0.999, 0.999).
- **Metrics per configuration and horizon:** R_opt (headline, full update vector), Disp(h),
  h_i statistics, and Corr(u1, ug).
- **Theory-guided question:** prior work shows a first-order gradient-scale invariance
  property when the two Adam moment time scales are balanced. We therefore test whether
  β1=β2 also reduces the finite-history, gap-induced R_opt in these controlled ECT
  gradient replays. The external theory does **not** predict that the reduction must be
  uniform across absolute β values or training stages.

> **Effective-support note.** Fresh-start updates span many orders of magnitude across β
> configurations, so an absolute threshold would make the h_i support configuration
> dependent. R_opt is computed over the **full vector**. The top-1% quantile mask by |u1|
> is used only for h_i/Disp summaries.
>
> **Regime note.** This experiment replays from a *fresh* start (m0=0, v0=0, step0=0), so
> its R_opt values (roughly 0.04–0.07) are **not directly comparable** to the cross-K
> real-state R_opt values (roughly 0.09–0.12). The real-state experiment asks how the
> already accumulated optimizer state responds; this experiment asks how a controlled β
> choice changes a newly accumulated 20-step history.
>
> **Reproducibility boundary.** The committed h=20 arrays independently reproduce the
> effective-support h_i statistics. The headline full-vector R_opt requires the large
> hash-identified gradient histories referenced by the cross-K raw manifest and is not
> independently recomputable from the small Git-committed h_i arrays alone. This is an
> explicit artifact-size limitation, not a claim that R_opt is Git-self-contained.

---

## 1. Results — R_opt at h=20

| K (kimg) | standard (0.9,0.999) | β=0.9 | β=0.99 | β=0.999 |
|----------|----------------------|-------|--------|---------|
| 32  | 0.0493 | 0.0497 | **0.0460** | **0.0456** |
| 64  | 0.0419 | **0.0417** | 0.0431 | 0.0432 |
| 128 | 0.0561 | 0.0571 | **0.0529** | **0.0523** |
| 256 | 0.0745 | 0.0748 | **0.0623** | **0.0609** |

Bold entries are lower than the standard configuration at the same K. These within-stage
comparisons are descriptive controlled replays; they are not twelve independent statistical
replicates.

---

## 2. Findings

**F1 — β balancing does not uniformly reduce gap-induced divergence.** The long-memory
balanced configurations (β=0.99 and β=0.999) reduce R_opt at k32, k128, and k256, while
k64 reverses this pattern. The short-memory balanced configuration (β=0.9) is essentially
neutral or slightly worse at three of the four stages.

**F2 — The largest observed reduction occurs at k256.** R_opt decreases from 0.0745 under
standard (0.9,0.999) to 0.0609 under balanced (0.999,0.999), approximately an 18% relative
reduction. This is a finite-history replay result, not a training-quality result.

**F3 — Absolute memory timescale matters empirically.** At k32, k128, and k256, R_opt
decreases as the common balanced β increases from 0.9 to 0.99 to 0.999; k64 is an
exception. This dependence on the *absolute* memory timescale is an empirical finite-history
observation of this ECT replay. It is **not** implied by the generic first-order statement
that the leading scale-sensitivity term vanishes when β1=β2.

**F4 — Update directions remain highly aligned.** Corr(u1,ug) is at least 0.997 across the
reported configurations, while R_opt resolves the smaller non-scalar component left after
the best global scaling.

**F5 — Constant scale and real gap history should be distinguished.** In a synthetic
exact-constant-scalar check, G^g = 0.77 G^1 produces near-equal normalized RAdam updates,
consistent with scale cancellation through the second-moment normalization. In the real
ECT gradients, the scale factor varies over steps and a small non-scalar gradient residual
is also present. The observed R_opt therefore reflects the full finite-history response to
these departures from an exact constant scalar relation; this experiment does **not**
identify how much is attributable separately to time-varying scalar history versus
non-scalar gradient content.

---

## 3. Answer to the core question

**Does balancing β1=β2 reduce the gap-induced R_opt?** Sometimes, but not uniformly.
Long-memory balanced configurations reduce R_opt at three of four sampled training stages,
with the largest reduction at k256, whereas short-memory balancing provides little or no
reduction and k64 reverses the long-memory pattern. Thus the strong universal prediction is
falsified; the observed sensitivity to β remains mechanism-consistent evidence that the
finite-history adaptive-optimizer response matters.

---

## 4. Claim boundaries

**We CAN say:**
- "Long-memory balanced-moment replays reduce gap-induced R_opt at three of four sampled
  stages, with an approximately 18% reduction at k256 for β1=β2=0.999."
- "The effect is training-stage- and absolute-timescale-dependent rather than universal."
- "The result is consistent with optimizer-history sensitivity and provides a targeted
  intervention on finite-history RAdam dynamics."

**We CANNOT say:**
- "Fernández-Hernández et al. predict that sufficiently long balanced memory must reduce
  R_opt" — their first-order invariance result concerns balancing the moment time scales,
  not a universal threshold in the common β value.
- "Balancing β causes the FID improvement" — this experiment never retrains or measures
  FID/KID.
- "The gap-induced R_opt is caused only by scalar scale history" — real gradients retain a
  small non-scalar residual and this experiment does not separate the two contributions.
- "Balanced β is uniformly better" — k64 and the β=0.9 condition falsify that statement.

**Why:** this is a controlled replay intervention on frozen gradient sequences, not a
training intervention. It establishes sensitivity of the optimizer-level divergence to β
configuration, while leaving endpoint generative-quality causality open.

---

## 5. Artifacts

- `analysis/balanced_beta/summary.json` — R_opt/Disp/Corr/h_i summaries per configuration
  and horizon.
- `analysis/balanced_beta/provenance.json` — protocol, trajectory identity, code binding,
  and artifact locations.
- `analysis/balanced_beta/{k32,k64,k128,k256}/raw_h20/{config}_h_actual.npy` — committed
  h_i values on the effective support at h=20.
- `analysis/crossk_scalar_history/raw_manifest.json` — SHA256/locator binding for the large
  external gradient histories needed to regenerate full-vector R_opt.
- `figures/balanced_beta/{fig1_Ropt_vs_beta,fig2_Ropt_vs_horizon,fig3_disp_corr_h20}.pdf|png`.

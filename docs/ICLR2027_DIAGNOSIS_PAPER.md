# Honest-Diagnosis Paper Framework — GFCT (Role C)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. This is the paper
skeleton for the **honest diagnosis** framing — the only framing the measured
data support. It consolidates: the clean g=1.3-vs-1.0 FID result, E5 (residual
structure), E6 (dose-response), and the moment-memory null falsification.

**Status:** framework + measured numbers, not a finished paper. All numbers are
from the repo's own audits/evaluations (see `analysis/`).

---

## Title (candidate)

*Non-Scalar Gradient Content in Few-Step Consistency Training: A Paired
Counterfactual Measurement*

(Alternative: *Where Scalar-Equivalence Holds and Fails in Few-Step Training*)

---

## Abstract (draft)

> Few-step generative training commonly assumes that a training intervention —
> a change in the discretization gap — is *scalar-equivalent* to a learning-rate
> change, i.e. absorbable by a single scalar multiplier on the parameter update.
> We test this assumption in the optimizer-update space, conditioned on a fixed,
> nontrivial RAdam state, using a paired counterfactual audit (identical
> minibatch, timestep, noise, dropout; gap g=1.0 vs g=1.3).
>
> We find that the scalar moment-memory chain — an exact identity relating the
> coordinate-wise update-ratio gauge to the gradient-scale history — predicts
> h≈1.001, but the actual update ratio is h≈0.837, driven by a non-scalar
> per-step gradient residual. The residual is reproducible and structured: it
> grows with layer depth and is a monotone function of |gap−1|, but is not
> magnitude-driven. Crucially, the residual is NOT harmful: the arm carrying it
> (g=1.3) achieves dramatically better FID/KID than the zero-residual arm
> (g=1.0) at both NFE=1 and NFE=2 (FID −34%/−36%, KID −37%/−45%).
>
> We conclude that scalar-equivalence reasoning in few-step training is
> systematically violated by non-scalar gradient content — a phenomenon the
> field's scale-invariance literature (which analyzes only scalar/global
> rescaling) does not cover — but that this content is benign, not a source of
> quality degradation. We provide a reproducible paired-counterfactual
> measurement and a structural characterization of where and why scalar
> equivalence fails.

---

## 1. Introduction

- The scalar-equivalence assumption and why it matters (a schedule/gap change
  is often treated as a learning-rate change).
- The question: does it hold in the optimizer-update space, conditioned on a
  fixed optimizer state?
- The honest finding: it fails, but benignly.

## 2. Setup and method

- Rectified RAdam, coordinate-wise update-ratio gauge `h_{k,i} = U_g/U_1`.
- The moment-memory identity `h = (1+A^(1))/sqrt(1+2A^(2)+B^(2))` as a
  falsifiable null model.
- The paired counterfactual audit (identical randomness; g=1.0 vs g=1.3).
- Measured audit numbers: `a_K*=0.761`, `R_grad=5.85%`, `R_opt=8.57%`,
  `H_K=R_opt` exact (energy gap 1.7e-18), `R_opt−R_grad=+0.027`.

## 3. Results (the diagnosis)

### 3.1 The scalar null is falsified
- Predicted h≈1.001 (median) vs actual h≈0.837 (std 0.058), Corr≈0 (both
  stably different, not noisy).

### 3.2 The residual is structured, not noise (E5)
- Gauge deviation from 1 grows with layer depth (pearson +0.46).
- Not magnitude-correlated.
- Gradient direction residual is a monotone function of |gap−1|.

### 3.3 The residual is not harmful (E6 + clean FID)
- g=1.3 (max residual) vs g=1.0 (zero residual): FID-5k −34%/−36%,
  KID-5k −37%/−45% at NFE=1/2.
- corr(residual, FID) ≈ −0.99: larger residual → better FID.
- The zero-residual arm (g=1.0) is the worst.

### 3.4 The residual is state-conditioned (E7)
- The gradient residual R_grad grows with optimizer maturity (n_K): 0.031
  (n_K=243) → 0.091 (n_K=1991), roughly monotone. The residual is not a fixed
  property of the network; it is optimizer-state-conditioned.
- The update distortion R_opt is small (0.012–0.017) and R_opt−R_grad is
  negative (the optimizer compresses the gradient residual).
- Measurement-sensitivity caveat: R_opt magnitude and the sign of R_opt−R_grad
  depend on the loss_fn checkpoint / minibatch (earlier audit: R_opt=0.086,
  +0.027; here: 0.014, −0.076 at the same n_K). Report the convention.

### 3.5 Robustness (E11)
- Cross-seed: at 1024k, the larger-gap arm (g=1.1) beats g=1.0 on FID-50k
  across all 3 seeds (5/6 cells). The "deviating from g=1.0 improves FID"
  finding is not a single-seed fluke.
- Support-threshold: Corr≈0 / scalar-null-falsified is stable across atol
  1e-5/1e-6.

### 3.6 Universality (E3)
- The gradient residual R_grad reproduces with AdamW (0.095 vs 0.091 for RAdam
  at the same state): the non-scalar GRADIENT residual is a property of the
  loss/gradient, not optimizer-specific.
- The update distortion R_opt is optimizer-dependent: AdamW amplifies
  (R_opt=0.39, R_opt−R_grad positive), RAdam compresses (R_opt=0.014, negative).

## 4. Related work / positioning

- β1=β2 gradient-scale invariance (global scalar; we are coordinate-wise,
  state-conditioned, non-scalar).
- Adam-atan2 / DeVA (scalar rescaling decomposition; we analyze non-scalar
  content).
- ADCM (schedule for quality; we analyze optimizer-state equivalence).
- SCT (objective/variance theory; we analyze update geometry).
- Dead-Direction Conditioners (symmetry-orbit gauge; we rename ours to
  "coordinate-wise update-ratio gauge").

## 5. Discussion and limitations

- g=1.3 differs in the whole gap (scalar + non-scalar), so we do not claim the
  residual *causes* the FID improvement — only that it is not harmful.
- FID-5k (not FID-50k), 256-kimg runs, seed 3, single schedule pair.
- The residual-magnitude measurement convention varies (3.2% from 20-step
  replay vs 5.85%/8.57% from single-step audit); we report the audit values.
- The 3.2%→16% attribution (1.001→0.837) is not derived; we report the
  measured h_actual directly.

## 6. What this paper does NOT claim

- NOT "the residual is harmful" (falsified: g=1.3 wins FID).
- NOT "a residual-corrected update improves quality" (contraindicated).
- NOT "first adaptive discretization / optimizer invariance / history-gauge
  theorem" (preempted).
- NOT "gap is equivalent to learning-rate change" (falsified by h=0.837).

---

## Measured numbers (for the paper)

| quantity | value | source |
|---|---|---|
| a_K* (gradient scalar fit) | 0.761 | stateful audit |
| R_grad | 5.85% | stateful audit |
| R_opt | 8.57% | stateful audit |
| H_K = R_opt | exact (1.7e-18) | stateful audit |
| h_predicted (scalar null) | 1.001 | moment-memory |
| h_actual | 0.837 (std 0.058) | moment-memory result |
| Corr(h_pred, h_actual) | ≈0 | moment-memory result |
| gauge dev vs depth (pearson) | +0.46 | E5 |
| corr(residual, FID) | −0.99 | E6 |
| g=1.3 vs g=1.0 FID-5k | −34%/−36% (NFE1/2) | clean FID |
| g=1.3 vs g=1.0 KID-5k | −37%/−45% (NFE1/2) | clean FID |
| R_grad vs n_K (243→1991) | 0.031→0.091 | E7 |
| cross-seed (1024k, g=1.1) | wins 5/6 FID-50k cells | E11 |
| Corr≈0 across atol 1e-5/1e-6 | 0.005 / 0.0002 | E11 |
| R_grad with AdamW (vs RAdam) | 0.095 (vs 0.091) | E3 |
| R_opt with AdamW (vs RAdam) | 0.39 (vs 0.014) | E3 |

## Files
- This framework: `docs/ICLR2027_DIAGNOSIS_PAPER.md`
- Strategy: `docs/ICLR2027_STRATEGY.tex`
- Review: `docs/ICLR2027_PLAN_REVIEW.md`
- Results: `analysis/g13_vs_g10_fid_result.md`, `analysis/e5_residual_structure_result.md`,
  `analysis/e6_dose_response_result.md`, `analysis/e7_optimizer_state_result.md`,
  `analysis/e11_robustness_result.md`, `analysis/e3_universality_result.md`,
  `analysis/moment_memory_real_history_result.md`

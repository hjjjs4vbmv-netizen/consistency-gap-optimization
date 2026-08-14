# Cross-K Moment-Memory Mechanism Experiment — Report

**Role C (theory) · P0 · 2026-08-14**
**Goal:** replicate the PR #47 scalar-history predictor across four training stages
K ∈ {32, 64, 128, 256} kimg of the same frozen seed-3 Arm-A trajectory, producing the
R²(K, h) matrix and answering Q1–Q4 honestly.

---

## 0. Protocol (controlled replication)

- **Only K varies.** The four states are the #49-frozen canonical Arm-A checkpoints
  `training-state-{000001,000002,000004,000008}.pt` of the *same* seed-3 trajectory
  (`arm_a_g1_0_lr_fixed_s3`). No retraining, no new trajectory, no new seed, no
  optimizer/gap/LR/predictor change, no tuning for higher R².
- **Predictor (identical to #47):** per-step global scalar
  `a_j* = ⟨G_j^1.3, G_j^1.0⟩ / ‖G_j^1.0‖²`, predicted gradient `Ĝ_j^1.3 = a_j*·G_j^1.0`.
- **Replay:** RAdam replayed from the real extracted state (m0/v0/step0) over the full
  20-step history, reading the update at every index t, giving ĥ_scalar_t and ĥ_actual_t
  for all horizons t = 1..20. Pure-numpy float32 RAdam (update independent of parameter value).
- **Weighting:** w = u1_t² (squared reference-update magnitude) on effective support.
- **Sanity anchor:** K=256 must reproduce Corr≈0.8582, R²≈0.735, a*=0.7726±0.0229. It does
  (0.8585 / 0.7355 / 0.7726±0.0229). **Anchor satisfied — interpretation proceeds.**

---

## 1. Results — the R²(K, h) matrix

Weighted R² (explained variance, primary metric):

| h \ K | 32   | 64   | 128  | 256  |
|-------|------|------|------|------|
| 1     | 0.992| 0.996| 0.989| 0.996|
| 2     | 0.982| 0.988| 0.990| 0.994|
| 4     | 0.981| 0.988| 0.979| 0.969|
| 5     | 0.981| 0.988| 0.982| 0.973|
| 8     | 0.973| 0.974| 0.987| 0.981|
| 10    | 0.976| 0.974| 0.983| 0.976|
| 15    | 0.952| 0.961| 0.973| 0.928|
| **20**| **0.861**| **0.911**| **0.918**| **0.736**|

**Headline horizon h=20, per stage:**

| K (kimg) | Weighted R² | Corr | wRMSE | R_opt | a* (mean±std) | eff. coords |
|----------|-------------|------|-------|-------|---------------|-------------|
| 32  | 0.861 | 0.937 | 1.25e-2 | 0.090 | 0.7702±0.0094 | 492,062 |
| 64  | 0.911 | 0.957 | 1.47e-2 | 0.108 | 0.7697±0.0094 | 144,981 |
| 128 | 0.918 | 0.966 | 1.56e-2 | 0.114 | 0.7700±0.0147 | 322,204 |
| 256 | 0.736 | 0.858 | 2.95e-2 | 0.117 | 0.7726±0.0229 | 989,959 |

> **Agreement with the canonical #47 receipts (`analysis/real_history/{k}/scalar_prediction.json`):**
> the a* mean±std above reproduce the receipts exactly (e.g. K=32: 0.770184±0.009437). The
> weighted R²/Corr here agree with the receipts to ~1e-3 (e.g. K=256: 0.7355/0.8585 here vs
> 0.7350/0.8582 in the receipt). The residual is implementation-level: the #47 receipts
> replayed RAdam through *torch* on a dummy parameter, while this experiment uses the
> pure-numpy float32 replay (`analysis/numpy_radam.py`), which is validated against the
> stored torch-generated histories to ~1 ulp (see `verify_u1_final`). Both paths reproduce
> the k256 anchor (0.735/0.8582) within the documented float32 tolerance.

---

## 2. Findings (honest)

**F1 — Explanatory power is present at every training stage, not only the final checkpoint.**
At short horizons h ≤ 10, Weighted R² ≥ 0.97 for all four K. Even the *longest* horizon
(h=20) exceeds 0.73 at every stage. The scalar-history predictor explains a substantial
fraction of the prospective optimizer discrepancy at K=32, 64, 128 *and* 256.

**F2 — R² at h=20 is non-monotonic across training stage.** It *rises* through the
intermediate stages (32→0.861, 64→0.911, 128→0.918) then *drops* at the final checkpoint
(256→0.736). The 256-kimg checkpoint — the one used in the original #47 analysis — has the
**lowest** long-horizon R², not the highest. The maximum h=20 explanatory power occurs at an
intermediate stage (128 kimg), not at the final state.

**F3 — a* is approximately stable around 0.77 across all four stages.** Per-K mean±std
over the 20-step window: 0.7702±0.0094 (K=32), 0.7697±0.0094 (K=64), 0.7700±0.0147
(K=128), 0.7726±0.0229 (K=256). The *mean* is tightly clustered (0.770–0.773, ≈0.77),
but the per-step *dispersion* grows with training stage (σ ≈ 0.009 at K=32/64 → 0.015 at
K=128 → 0.023 at K=256). So the scalar geometry is **approximately stable, not exactly
stage-invariant**; what varies with K is how well a fixed scalar predictor *predicts*
the multi-step accumulated update, which degrades most sharply at the final checkpoint.

**F4 — Precision caveat (benign):** the replay-vs-stored float32 exactness check is
`False` for all K (max_abs_diff up to ~9.3e-7 on rare near-zero-reference coords). This is
characterized as benign float32 rounding (median |diff| ≈ 4.6e-9, ratio u1/ref = 0.99999),
and the k256 anchor reproduces to the 4th decimal — so the matrix above is trustworthy. It
is documented here for full auditability, not swept under the rug.

---

## 3. Answers to Q1–Q4

- **Q1. Does the scalar-history predictor explain a substantial fraction of the prospective
  optimizer discrepancy across *multiple* training stages?** Yes. R² ≥ 0.73 at h=20 and
  ≥ 0.97 at h≤10 for every stage K ∈ {32,64,128,256}.
- **Q2. Is moment-history explanatory power restricted to the final 256-kimg checkpoint?**
  No. It is present and generally *higher* at earlier stages (F2).
- **Q3. Does a* (the scalar geometry) change across training?** Approximately stable: the
  mean is 0.770–0.773 (≈0.77) at every stage, but the per-step dispersion grows with
  training stage (σ ≈ 0.009 at K=32/64 → 0.015 at K=128 → 0.023 at K=256). The stage
  dependence lives in the *prediction fidelity* of the scalar, not in its mean.
- **Q4. Which stage has the strongest long-horizon explanatory power?** The intermediate
  128-kimg stage (R²=0.918 at h=20); the final 256-kimg stage is the weakest (0.736).

---

## 4. Claim boundaries (must be preserved)

**We CAN say:**
- "The scalar-history predictor explains a substantial fraction of the prospective optimizer
  discrepancy across multiple training stages (R² ≥ 0.73 at h=20, ≥ 0.97 at h≤10)."
- "Moment-history explanatory power is not restricted to the final 256-kimg checkpoint; it
  is present, and for long horizons strongest, at intermediate stages."
- "The scalar geometry a* is approximately stable around 0.77 across stages (mean
  0.770–0.773; per-step dispersion grows from σ≈0.009 at K=32/64 to σ≈0.023 at K=256)."

**We CANNOT say:**
- "Moment memory causes the FID improvement."
- "We have proven that optimizer memory is the mechanism behind ECT performance."
- "RAdam memory fully explains the update mismatch."

**Why:** even R² ≈ 0.9 is correlation/explained-variance, not complete causal
identification. The predictor explains the *optimizer discrepancy*; whether (and how much)
that discrepancy drives ECT's FID/quality outcomes is a separate question this experiment
does not touch. **This experiment makes no claim about FID causality.**
The non-monotonic R²(K) itself is a caution: had we only looked at the final checkpoint we
would have reported a *lower* bound on explanatory power than the method actually sustains.
No figure title asserts "degrades"/"causes"; all titles are descriptive.

---

## 5. Artifacts

- `analysis/crossk_scalar_history/summary.json` — full R²(K,h) matrix + all metrics per K.
- `analysis/crossk_scalar_history/provenance.json` — frozen-state identity, git commit,
  method, validation, environment.
- `analysis/crossk_scalar_history/{k32,k64,k128,k256}/raw_predictions/*.npy` — per-coordinate
  h_pred/h_actual/weights/a* at h=20 for independent recomputation.
- `figures/cross_k_scalar_history/fig1_R2_vs_training_stage`, `fig2_R2_vs_horizon`,
  `fig3_scatter_h20` (PDF+PNG); `figures/summary_table.csv`.

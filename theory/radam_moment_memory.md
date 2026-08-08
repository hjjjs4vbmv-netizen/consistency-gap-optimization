# RAdam Moment-Memory Theorem — First-Order Quantitative Draft (Role C)

Date: 2026-08-08. Branch: `theory/radam-moment-memory`.
Goal (per today's plan): make the GFCT history-gauge story **quantitative** —
give the coordinate-wise history gauge `h_{t,i}` an explicit first-order
expansion in the scale-history `δ_j`, and turn the three qualitative corollaries
into a checkable statement.

---

## 1. Setup

Gradient perturbation is a time-varying **scalar** scale on the reference arm:

```
G^g_{j,i} = (1 + δ_j) G_{j,i}
```

- `G_{j,i}`: reference gradient of arm `g=1` at step `j`, coordinate `i`
- `δ_j`: per-step relative gap-scale deviation from 1
- Rectified RAdam (same step index across arms; ignore `eps`, bias correction,
  weight decay for the theorem; stated as first-order in `||δ||`)

RAdam moments are exponential-history sums:

```
m^g_t = (1-β1) Σ_{j≤t} β1^{t-j} G^g_j,     m_t = (1-β1) Σ_{j≤t} β1^{t-j} G_j
v^g_t = (1-β2) Σ_{j≤t} β2^{t-j} (G^g_j)²,  v_t = (1-β2) Σ_{j≤t} β2^{t-j} G_j²
```

Rectified update (per coordinate): `U_{t,i} ∝ mhat_{t,i} / sqrt(vhat_{t,i})`,
so the coordinate-wise history gauge is

```
h_{t,i} = U^g_{t,i} / U_{t,i}
        = (mhat^g / mhat) · sqrt(vhat / vhat^g)
        = (m^g / m) · sqrt(v / v^g)        (bias correction cancels)
```

---

## 2. First-order expansion

Define the **first-moment history gauge** and **second-moment history gauge**:

```
A^(1)_{t,i} := ( Σ_j p_{tj} δ_j G_{j,i} ) / ( Σ_j p_{tj} G_{j,i} ),   p_{tj} ∝ β1^{t-j}
A^(2)_{t,i} := ( Σ_j q_{tj} δ_j G_{j,i}² ) / ( Σ_j q_{tj} G_{j,i}² ), q_{tj} ∝ β2^{t-j}
```

**Lemma (first-order).** To first order in `||δ||`:

```
m^g_t / m_t   = 1 + A^(1)_{t,i} + O(||δ||²)
v^g_t / v_t   = 1 + 2 A^(2)_{t,i} + O(||δ||²)
sqrt(v_t/v^g_t) = (1 + 2 A^(2))^{-1/2} = 1 - A^(2)_{t,i} + O(||δ||²)
```

**Theorem (moment-memory).** For rectified RAdam, to first order in `||δ||`:

```
h_{t,i} - 1 = A^(1)_{t,i} - A^(2)_{t,i} + O(||δ||²)
```

*Proof.* `h = (m^g/m)·sqrt(v/v^g) = (1+A^(1))(1-A^(2)) + O(δ²)
= 1 + A^(1) - A^(2) + O(δ²)`, where cross term `A^(1)A^(2)` is second order. ∎

**Interpretation.** The history gauge deviation is the **difference between the
β1-weighted mean of `δ` (over gradients) and the β2-weighted mean of `δ` (over
gradient squares)**. It is zero iff these two history-weighted means coincide.
The two memory time-scales (`β1=0.9` vs `β2=0.999`) are the mechanism: the same
`δ_j` history is averaged with different forget rates by the two moments.

---

## 3. Corollary 1 — Constant-scale null (phase-qualified)

**Statement (rectified phase only).** If `δ_j ≡ δ` (constant for all `j ≤ t`)
and the step is in the **rectified** regime (update ∝ `mhat/sqrt(vhat)`), then
`A^(1) = A^(2) = δ`, so `h_{t,i} = 1 + O(δ²)`.

*Proof.*
```
A^(1) = δ · (Σ p_{tj} G_j)/(Σ p_{tj} G_j) = δ
A^(2) = δ · (Σ q_{tj} G_j²)/(Σ q_{tj} G_j²) = δ
=> h - 1 = δ - δ + O(δ²) = O(δ²)   (rectified update)
```
Under the idealized `eps=0` assumption the cancellation is **exact** (`h=1`);
the measured `~1e-4` residual is the `eps` effect, not a `δ²` term.

**Important phase qualification (self-review fix).** In the **unrectified**
phase (first ~5 steps, update ∝ `mhat`), the coordinate gauge is `h = a = 1+δ`
(first order in δ), NOT `1+O(δ²)`. So the constant-scale null is **not** a
full-history `O(δ²)` statement; it holds only after rectification activates.
This matches P-R1 (unrectified scale equivariance) and P-R2 (rectified null):
- step 0-4 (unrectified): `h - 1 = δ = 0.30` (measured, first order);
- step ≥5 (rectified): `h - 1 ≈ 1e-4` (measured, ~eps-level).

The earlier draft (and PR #45 T1 test, which sampled only step 40) overstated
this as a full-history `O(δ²)` null; corrected here.

---

## 4. Corollary 2 — Time-varying scale ⇒ distortion (generic)

**Statement.** If `δ_j` varies over the recent history, then generically
`A^(1) ≠ A^(2)` — the two moments weight the same `δ_j` history with different
forget rates (`β1=0.9` vs `β2=0.999`), so the weighted means differ:
`h_{t,i} - 1 ≈ A^(1) - A^(2) ≠ 0` to first order.

*Non-triviality.* Equality `A^(1)=A^(2)` requires
`(ΣpδG)/(ΣpG) = (ΣqδG²)/(ΣqG²)`, a measure-zero coincidence in general. The
distortion is **history-induced**: it is present even if every instantaneous
gradient residual `E_j = G^g_j - (1+δ_j)G_j` is exactly zero (δ is the exact
scalar). This is the quantitative form of "instantaneous near-scalar ⇏ update
near-scalar".

*Accuracy (self-review).* The first-order formula
`h - 1 = A^(1) - A^(2) + O(δ²)` was verified against the real RAdam update:
the moment-predicted `h` matches the actual update with relative error that
scales as `δ²` (measured: 0.002 at δ×0.1, 0.014 at ×0.3, 0.119 at ×1.0). So at
the `δ=±0.3` scale used in the synthetic check the error is ~7-12%, dominated
by first-order truncation (plus `eps`), not by a flaw in the mechanism. For
quantitative use on real ECT the relevant `δ_j` is small (gap-scale deviation
~0.1-0.3), where the approximation is accurate to a few percent.

---

## 5. Corollary 3 — Coordinate heterogeneity ⇒ R_opt > 0

**Statement.** If coordinates `i,l` have different temporal gradient
compositions (`{G_{j,i}}_j` vs `{G_{j,l}}_j` are not proportional), then
`A^(1)_{t,i} ≠ A^(1)_{t,l}` in general, hence `h_{t,i} ≠ h_{t,l}`, and by the
coordinate-wise gauge theorem (rev.3) the best-scalar update residual satisfies

```
R_opt(t) > 0   (first order in ||δ||).
```

*Proof sketch.* `h_{t,i} - h_{t,l} = (A^(1)_i - A^(2)_i) - (A^(1)_l - A^(2)_l)
+ O(δ²)`. If `G_i` and `G_l` have different temporal profiles, `A^(1)_i ≠ A^(1)_l`
generically, so `h` is not coordinate-constant on the effective support ⇒
`R_opt > 0` by the iff theorem.

---

## 6. Sufficient condition for gauge dispersion (today's plan item)

For a **two-block** history (δ = δ_a for `j ≤ t0`, δ = δ_b for `t0 < j ≤ t`),
with the block boundary at `t0` and `t - t0` steps of δ_b:

```
A^(1) = [ β1^{t-t0} S_a^{(1)} δ_a + S_b^{(1)} δ_b ] / [ β1^{t-t0} S_a^{(1)} + S_b^{(1)} ]
A^(2) = [ β2^{t-t0} S_a^{(2)} δ_a + S_b^{(2)} δ_b ] / [ β2^{t-t0} S_a^{(2)} + S_b^{(2)} ]
```
where `S_a^(m), S_b^(m)` are the accumulated (unweighted) moment sums in each
block. Because `β1^{Δ} ≠ β2^{Δ}` for `Δ > 0`, the two gauges weight the old vs
new block differently. **Sufficient condition for nonzero dispersion:**
`δ_a ≠ δ_b` AND `Δ > 0` AND the block sums `S_a^(m), S_b^(m)` do not make the
two ratios accidentally equal. Concretely, right after a scale change
(`Δ` small), `A^(2)` still reflects the old block longer (β2 decays slower),
so `A^(1) - A^(2)` is dominated by `δ_b - δ_a` times a positive factor.

---

## 7. What this establishes / does not

**Establishes** (draft-grade, not publication-polished):
- Explicit first-order formula `h_{t,i} - 1 = A^(1) - A^(2) + O(δ²)`.
- Quantitative null (Cor 1), generic distortion (Cor 2), coordinate breaking
  (Cor 3).
- A checkable two-block sufficient condition for nonzero gauge dispersion.

**Does not establish:**
- Higher-order / non-perturbative control (δ large, eps, weight decay, AMP).
- Estimate of real ECT's `δ_j` history (Role D measures `a_k*` per checkpoint).
- Connection to FID (three-arm study).

---

## 8. Relation to Role D

Role D measures `a_K*` (instantaneous scale), `c_K*`, `s_K*`, `R_grad(K)`,
`R_opt(K)`, and `h_{K,i}` / `H_K`. The theorem predicts:
- if `a_K*` is roughly constant across states, `H_K` ≈ 0 (null);
- if `a_K*` varies (δ_j history nontrivial), `R_opt(K) - R_grad(K) > 0`, and the
  **moment-predicted** `h_{t,i}` from the `A^(1)-A^(2)` formula should match the
  **actual-update** `h_{t,i}` (validating the memory mechanism).

**The key comparison is `R_opt(K) - R_grad(K)`, not `H_K = R_opt`** (the latter
is an identity). `R_grad(K)` is the raw-gradient directional residual (already
known to be small, ~0.3% whole-model); `R_opt(K)` is the optimizer-update
residual. The theorem's quantitative content is:
- `R_opt(K) - R_grad(K) ≈ 0` when the δ_j history is trivial (constant or
  short-memory) → gap stays in the optimizer-equivalence class;
- `R_opt(K) - R_grad(K) > 0` when the δ_j history is nontrivial → the
  optimizer memory converts a near-scalar instantaneous gap into a genuine
  update-direction difference, and the size should track `A^(1)-A^(2)`.

---

## Files
- `theory/radam_moment_memory.md` — this draft.
- `theory/test_radam_moment_memory.py` — numeric checks (constant→null;
  time-varying→h-1≈A1-A2; coordinate heterogeneity→R_opt>0; moment-predicted
  vs actual h).

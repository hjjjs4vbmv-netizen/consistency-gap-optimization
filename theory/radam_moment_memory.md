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

## 2. Exact memory identity (upgraded per review)

**Effective support (domain condition — added per review).** The gauges involve
divisions by the signed first moment `m_t` and the (positive) second moment
`v_t`. Define the **effective support** at step `t` as

```
S_t := { i : m_{t,i} ≠ 0  AND  v_{t,i} > 0 }.
```

- On `S_t` the quantities `A^(1), A^(2), B^(2), m^g/m, v^g/v, h` are all
  well-defined.
- Coordinates with `m_{t,i} = 0` (signed-moment cancellation) are **outside**
  the support: `A^(1)` and `m^g/m` are undefined there (denominator zero), and
  `h` is undefined. These coordinates inherit the **off-support treatment** of
  PR #43 (rev.4): scalar update equivalence requires `U_{g,i} = 0` there, and
  the residual formula carries the explicit `Σ_{i∉S_t} U_{g,i}²` term.
- `v_{t,i} > 0` is included because RAdam's rectified update divides by
  `sqrt(vhat)`; `v_{t,i} = 0` (no gradient history) also drops the coordinate
  out of the rectified update.

All statements below are **on `S_t`** unless noted.

Define the **first-moment history gauge**, **second-moment history gauge**, and
**second-moment quadratic gauge** (for `i ∈ S_t`):

```
A^(1)_{t,i} := ( Σ_j p_{tj} δ_j G_{j,i} ) / ( Σ_j p_{tj} G_{j,i} ),   p_{tj} ∝ β1^{t-j}
A^(2)_{t,i} := ( Σ_j q_{tj} δ_j G_{j,i}² ) / ( Σ_j q_{tj} G_{j,i}² ), q_{tj} ∝ β2^{t-j}
B^(2)_{t,i} := ( Σ_j q_{tj} δ_j² G_{j,i}² ) / ( Σ_j q_{tj} G_{j,i}² )
```

**Exact identity (theorem).** For rectified RAdam, same step index, bias
correction cancels, and for every `i ∈ S_t`, the coordinate-wise history gauge
is **exactly**

```
m^g_t / m_t   = 1 + A^(1)_{t,i}                       (exact, linear in δ)
v^g_t / v_t   = 1 + 2 A^(2)_{t,i} + B^(2)_{t,i}        (exact, quadratic in δ)

h_{t,i} = (1 + A^(1)_{t,i}) / sqrt( 1 + 2 A^(2)_{t,i} + B^(2)_{t,i} )
```

*Proof.* `m^g = Σ p (1+δ_j) G = m + Σ p δ_j G`, divide by `m` (nonzero on
`S_t`): `1 + (ΣpδG)/(ΣpG) = 1 + A^(1)`. Similarly `v^g = Σ q (1+δ_j)² G²
= v + 2ΣqδG² + Σqδ²G²`, divide by `v`: `1 + 2A^(2) + B^(2)`. Then `h =
(m^g/m)·sqrt(v/v^g)` by the rectified update ratio. ∎

This is **not a Taylor heuristic**: it is an exact identity of the moment
recursion. The first-order expansion is its corollary:

**Corollary (first order).** Expanding the exact identity in `||δ||`:

```
h_{t,i} - 1 = A^(1)_{t,i} - A^(2)_{t,i} + O(||δ||²)
```

and to second order (explicit cross/quadratic terms):

```
h - 1 = A^(1) - A^(2) - A^(1)A^(2) - ½ B^(2) + ³⁄₂ (A^(2))² + O(||δ||³)
```

**Mechanism (corrected per review).** The gauge deviation is the **difference
between the effective scale histories seen by the first and second moments**.
This mismatch has two sources:
1. **different decay kernels** (`β1 ≠ β2`): the same δ_j history is forgotten at
   different rates by `m` and `v`;
2. **different gradient weighting** (`G_j` vs `G_j²`): even with `β1 = β2`,
   weighting gradients vs squared gradients yields different effective averages
   of the same δ_j history.

Either source can independently induce the mismatch **under a nontrivial scale
history** (i.e. when `δ_j` varies over the history); the statement is not
unconditional — with `δ_j ≡ δ` both sources give identical gauges and the null
holds regardless of kernels/weighting.

So the mechanism is **first/second-moment effective-history mismatch**, of
which unequal decay rates are one (important but not the only) source.

---

## 3. Corollary 1 — Constant-scale null (phase-qualified, sign-qualified)

**Statement (rectified phase only, positive scale).** Assume (i) `δ_j ≡ δ`
(constant for all `j ≤ t`), (ii) the step is in the **rectified** regime
(update ∝ `mhat/sqrt(vhat)`), and (iii) **`1 + δ > 0`** (the constant scale is
positive — the regime relevant to ECT's gap multipliers, where `a = 1+δ ≈ 0.77`).
Then `A^(1) = A^(2) = δ`, so `h_{t,i} = 1 + O(δ²)`.

*Proof.*
```
A^(1) = δ · (Σ p_{tj} G_j)/(Σ p_{tj} G_j) = δ
A^(2) = δ · (Σ q_{tj} G_j²)/(Σ q_{tj} G_j²) = δ
=> h - 1 = δ - δ + O(δ²) = O(δ²)   (rectified update)
```
Under the idealized `eps=0` assumption the cancellation is **exact** (`h=1`);
the measured `~1e-4` residual is the `eps` effect, not a `δ²` term.

**Sign qualification (per review).** The positivity `1 + δ > 0` is necessary.
In the idealized identity, `m^g = (1+δ)m`, `v^g = (1+δ)²v`, so
`h = (1+δ)/|1+δ|`. With `1+δ > 0` this is `h = 1` (the null); with
`1+δ < 0` it gives `h = -1` (a sign flip, not a null). The assumption `a>0`
is exactly the actual ECT regime (gap multipliers are positive), but must be
stated for the theorem to hold as written.

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
the `δ=±0.3` block-alternating scale used in the synthetic check the error is
~7-12%, dominated by first-order truncation (plus `eps`), not by a flaw in the
mechanism.

**Accuracy as a function of δ amplitude (honest, per review).** The
block-alternating synthetic is a *worst case* (large δ jumps at block
boundaries inflate `A^(1)-A^(2)` and the second-order cross term). For smaller,
smoother δ the first-order rel-err drops:

| δ amplitude | rel-err |
|---|---:|
| 0.10 | 0.1% |
| 0.15 | 0.2% |
| 0.23 | 0.2% |
| 0.30 | 0.3% |

**Important caveat (added per review).** This is a *synthetic* accuracy
statement only. The observed `a* ≈ 0.755` (from #38) is the whole-model
**mean-gradient best scalar fit**, not evidence that per-training-step
`1 + δ_j` is exact, small, or smooth. The δ=0.23 row should be read as
"amplitude matched to the observed whole-model scalar coefficient", **not**
"the real-ECT regime". Whether the formula is quantitatively reliable for the
actual experiment requires Role D's real history measurement; it is not
established here.

---

## 5. Corollary 3 — Gauge-dispersion condition ⇒ R_opt > 0 (tightened)

**Corrected statement (per review).** The object of interest is the **gauge
deviation**
```
D_{t,i} := A^(1)_{t,i} - A^(2)_{t,i}
```
(the first-order `h_{t,i} - 1`). Coordinate heterogeneity is necessary but not
sufficient for `R_opt > 0`: what matters is whether `D_{t,i}` is constant on
the effective support, not whether each `A` individually differs.

**Off-support qualification (per review).** The iff must carry an off-support
condition. On a coordinate with reference `m_{t,i} = 0` (`i ∉ S_t`), the
reference update is `U_{1,i} = 0`, but a time-varying `δ_j` history can give
`m^g_{t,i} = Σ_j p δ_j G_{j,i} ≠ 0`, hence `U_{g,i} ≠ 0`. This contributes
off-support energy `Σ_{i∉S_t} U_{g,i}²` to `R_opt` independently of `D`. The
precise first-order statement is therefore:

```
[ U_{g,i} = 0 for all i ∉ S_t ]  ⇒  ( R_opt(t) > 0  ⟺  D_{t,i} not coord-constant on S_t )
```

and in general `R_opt` has **two independent first-order sources**: (i) `D`
non-constant on `S_t`, and (ii) off-support candidate energy (coordinates where
the reference update vanishes but the candidate's does not).

*Proof (on-support part).* By the exact identity, `h_{t,i} - 1 = D_{t,i} + O(δ²)`
on `S_t`. If `D_i ≠ D_l` then `h_i ≠ h_l`, so by the rev.3 iff theorem `U_g` is
not a scalar multiple of `U_1`, giving `R_opt > 0` to first order. Conversely,
if `D` is constant on `S_t` **and** there is no off-support candidate energy,
then `h` is constant to first order and `R_opt = O(δ²)`. ∎

*Why this tightening matters.* It is possible for `A^(1)_i ≠ A^(1)_l` while
`A^(2)_i ≠ A^(2)_l` *cancels* the difference (`D_i = D_l`); in that case the
coordinate heterogeneity produces no first-order `R_opt`. The earlier draft's
"different temporal composition ⇒ R_opt > 0" was too strong. And even with `D`
constant on `S_t`, off-support energy can still drive `R_opt > 0`.

---

## 6. Two-block history: exact gauge-dispersion formula (per review)

Consider a **two-block** history: `δ_j = δ_a` for `j ≤ t0`, `δ_j = δ_b` for
`t0 < j ≤ t`. Let `Δ = t - t0` (steps since the block boundary) and define the
**block-weighted moment sums** (these carry the exponential decay *within* each
block — they are not "unweighted"):

```
P_{a,i} := Σ_{j≤t0} β1^{t0-j} G_{j,i},      P_{b,i} := Σ_{t0<j≤t} β1^{t-j} G_{j,i}
Q_{a,i} := Σ_{j≤t0} β2^{t0-j} G_{j,i}²,     Q_{b,i} := Σ_{t0<j≤t} β2^{t-j} G_{j,i}²
```

Then the gauges are exact:

```
A^(1)_i = ( β1^Δ P_a δ_a + P_b δ_b ) / ( β1^Δ P_a + P_b )   = δ_b + (δ_a - δ_b) α^(1)_i
A^(2)_i = ( β2^Δ Q_a δ_a + Q_b δ_b ) / ( β2^Δ Q_a + Q_b )   = δ_b + (δ_a - δ_b) α^(2)_i
```
where the **old-block effective weights** are

```
α^(1)_i := β1^Δ P_{a,i} / ( β1^Δ P_{a,i} + P_{b,i} )    (signed, on m-support)
α^(2)_i := β2^Δ Q_{a,i} / ( β2^Δ Q_{a,i} + Q_{b,i} )    (nonnegative, in [0,1])
```

**Sign caveat (per review).** `α^(1)` is an **effective signed old-block
weight/ratio**, NOT a probability-like `[0,1]` fraction: the first-moment sums
`P_a, P_b` are *signed* gradient sums, so `α^(1) < 0` or `α^(1) > 1` can occur,
and the denominator `β1^Δ P_a + P_b = m_t` is zero exactly off the m-support
(coordinates with `m_{t,i} = 0` — consistent with §2's effective support; there
`A^(1)` is undefined and the coordinate is excluded from `S_t`). `α^(2)` is
nonnegative and in `[0,1]` because `Q_a, Q_b ≥ 0` (squared gradients), with the
denominator `v_t > 0` on `S_t`. The factorization holds wherever the gauges are
defined (i.e. on `S_t`).

**Clean formula.** The gauge deviation factors exactly:

```
D_i := A^(1)_i - A^(2)_i = (δ_a - δ_b) · ( α^(1)_i - α^(2)_i )
```

**Sufficient condition.** To first order,
```
h_i ≠ 1   ⟺   δ_a ≠ δ_b  AND  α^(1)_i ≠ α^(2)_i.
```
And if `α^(1)_i - α^(2)_i` differs across coordinates, then `D_i` is not
coordinate-constant and `R_opt > 0` (first order).

*Why this is rigorous (vs the earlier draft).* The old "dominated by `δ_b-δ_a`
times a positive factor" is not generally valid — the first-moment gradients can
have signs and cancellations. The factored formula `D_i = (δ_a-δ_b)(α_i^(1)-α_i^(2))`
holds exactly, with no sign assumptions and no positivity claim on `α^(1)`.
The difference of the two effective weights is the clean driver (defined on
`S_t`; off-support coordinates handled per §2 / PR #43).

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

Role D measures `a_K*` (best-fit instantaneous scale), `c_K*`, `s_K*`,
`R_grad(K)`, `R_opt(K)`, and `h_{K,i}` / `H_K`.

**Theorem-guaranteed predictions — qualified (per review).** These hold
**only** under the theorem's assumptions: exact scalar history
`G^g_j = (1+δ_j) G_j` (i.e. no non-scalar `E_j`), idealized rectified RAdam
(`eps=0`, no weight decay), same step index, and on the effective support.
- `h_{K,i}^moment ≈ h_{K,i}^update`: the coordinate gauge computed from the
  moment formula `(m^g/m)·sqrt(v/v^g)` should match the actual-update ratio
  (validating the memory identity);
- `R_opt(K)` should be predictable from the coordinate dispersion of
  `D_{K,i} := A^(1)_{K,i} - A^(2)_{K,i}` (the exact two-block formula gives
  the mechanism, and the weighted-dispersion identity ties `H_K` to `R_opt`).

**On real ECT these are falsifiable predictions / diagnostics, not guaranteed**
**theorems**: real training has non-scalar `E_j ≠ 0`, `eps > 0`, weight decay,
AMP, and `a_K*` is a best-fit scalar rather than the exact `1+δ_j`. The
`h^moment ≈ h^update` check is the one that stays most assumption-light and is
the recommended first diagnostic.

**Empirical hypotheses (NOT theorem-guaranteed, per review):**
- `a_K*` roughly constant across states ⇒ `H_K ≈ 0` (null);
- `R_opt(K) - R_grad(K) > 0` when the δ_j history is nontrivial. This is a
  **diagnostic hypothesis, not a theorem consequence**: `a_K*` is a best-fit
  scalar coefficient, not the exact `1+δ_j` assumed by the theorem, and ECT has
  non-scalar `E_j ≠ 0`; moreover the optimizer can partly compress as well as
  amplify a residual, so `R_opt > R_grad` is not unconditionally guaranteed.
  The size and sign of `R_opt - R_grad` should be measured and compared with
  the moment-predicted `D_{K,i}` dispersion.

**The key comparison is `R_opt(K) - R_grad(K)`, not `H_K = R_opt`** (the latter
is an identity). The quantitative content to test on real checkpoints is
whether `R_opt - R_grad` tracks the `D_{K,i}` dispersion predicted by the
memory identity.

---

## Files
- `theory/radam_moment_memory.md` — this draft.
- `theory/test_radam_moment_memory.py` — numeric checks (constant→null;
  time-varying→h-1≈A1-A2; coordinate heterogeneity→R_opt>0; moment-predicted
  vs actual h).

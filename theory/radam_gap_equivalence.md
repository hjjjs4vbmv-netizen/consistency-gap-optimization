# RAdam Gap Equivalence — Proposition Draft (Role C, rev.2)

Date: 2026-08-07. Branch: `theory/radam-gap-equivalence`.
Part of the GFCT main line: **separate optimizer reparameterization from genuine
gap effects in the optimizer-update space.**

rev.2 addresses the review: (i) splits the two `c*` definitions to match
PR #42; (ii) fixes the cumulative-displacement bug (now measures single-step
updates); (iii) reframes the headline as **constant-scale invariance vs
history-induced gauge breaking** (RAdam's warm-up phase is ~5 steps, not
32-256 kimg).

---

## 1. Setup and notation

State-augmented optimizer update (GFCT §5.1):

```
z_k = (theta_k, m_k, v_k, n_k, s_k, EMA_k)
```

- `theta`: network parameters
- `m`, `v`: RAdam first/second moments
- `n`: step index
- `s`: GradScaler / AMP state
- `EMA`: exponential moving average of parameters

Real update under gap `g`:

```
theta_{k+1} = Phi_g(z_k, xi_k),   U_g(z, xi; eta) = theta_g^+ - theta
```

## 2. Two scalar-gauges (review fix: do not conflate)

Let `U_g` be the single-step parameter update of arm `g`, `U_1` the reference.

**Update scale** (fit reference direction to candidate):
```
s_g^* = <U_g, U_1> / ||U_1||^2        (U_g ~= s_g^* U_1)
```
If `U_g = a U_1` then `s_g^* = a`.

**Candidate LR multiplier** (what to multiply the candidate update by to match
the reference — the quantity PR #42's Arm C needs):
```
c_g^* = <U_g, U_1> / ||U_g||^2        (c_g^* U_g ~= U_1)
```
If `U_g = a U_1` then `c_g^* = 1/a`.

`c_g^*` is used everywhere in this PR and in PR #42. `s_g^*` is kept as a
separate, named gauge for the raw-gradient/update-scale direction.

**Raw-gradient observation** (previous work, not re-derived): `mu_g = a_g mu_1`
with `a_{1.3} ≈ 0.770` (256 kimg, whole-model residual 0.30%), so at the
gradient level `g` is near a scalar rescaling.

---

## 3. P-R1 — Unrectified scale equivariance

**Setup.** (i) `G^g = a G^1` (constant `a > 0`); (ii) matched moments
`m_0^g = a m_0^1`, `v_0^g = a^2 v_0^1`; (iii) `eps = 0`, weight decay = 0;
(iv) unrectified regime `r_k = 1` (update `∝ mhat`).

**Claim.**
```
m_k^g = a m_k^1,   v_k^g = a^2 v_k^1   (induction)
U_g = a U_1
=> s_g^* = a,   c_g^* = 1/a.
```

*Proof (induction).* Base `k=0` by assumption. Step:
```
m_k^g = beta1 m_{k-1}^g + (1-beta1) G^g
      = beta1 (a m_{k-1}^1) + (1-beta1) (a G^1) = a m_k^1
v_k^g = beta2 v_{k-1}^g + (1-beta2) (G^g)^2
      = beta2 (a^2 v_{k-1}^1) + (1-beta2) (a^2 (G^1)^2) = a^2 v_k^1.
```
Bias correction is linear in `m,v` (and `n` is identical across arms), so
`mhat^g = a mhat^1`, `vhat^g = a^2 vhat^1`. In the unrectified regime the
update is `eta * mhat`, hence `U_g = a U_1`, `c_g^* = 1/a`. ∎

**Numeric anchor** (real `torch.optim.RAdam`, `G_g = a G_1`, `a=1.3`,
single-step updates `u = theta_{k+1} - theta_k`):

| step | c\* |
|---:|---:|
| 0-4 | **0.7692** (= 1/a = 1/1.3) |
| ≥ 10 | 1.0000 |

**The unrectified→rectified switch happens within ~5-10 optimizer steps**,
NOT over 32/64/128/256 kimg. The earlier "1.30 → 1.04 gradual drift" was an
artifact of measuring cumulative displacement `-theta_k` (mixed-phase average);
it is retracted.

---

## 4. P-R2 — Rectified constant-scale invariance (null theorem)

**Setup.** Same as P-R1 but the update is in the rectified regime
(`update ∝ mhat / sqrt(vhat)`), and the scale is constant over the *entire
history*: `G^g_j = a G^1_j` for all `j <= k`.

**Claim (null theorem).**
```
mhat_g / sqrt(vhat_g) = (a mhat_1) / sqrt(a^2 vhat_1) = mhat_1 / sqrt(vhat_1)
=> U_g = U_1   (up to eps, weight decay, rectification factor r_k)
=> c_g^* = 1.
```
A **constant** positive gradient rescaling is absorbed by rectified RAdam.

**Numeric anchor** (same run): at step ≥ 10, `c^* = 1.0000` exactly.

**Key consequence for the GFCT line:**
> A fixed LR multiplier matches the reference in the rectified phase
> (`c^* = 1`), but needs `c^* = 1/a` in the unrectified first ~5 steps.
> Since the unrectified phase is a tiny fraction of training, the warm-up
> transition is NOT the main phenomenon. The real object is history-induced
> gauge breaking (P-R3).

---

## 5. P-R3 — Coordinate-wise history gauge theorem (the headline)

**Setup.** Rectified RAdam, ignore `eps` and weight decay, identical step index
across arms. Per parameter coordinate `i`, the update is
`U_{g,i} ∝ mhat_{g,i} / sqrt(vhat_{g,i})`, `U_{1,i} ∝ mhat_{1,i} / sqrt(vhat_{1,i})`.

**Definition (coordinate-wise history gauge).**
```
h_{k,i} := (mhat^g_{k,i} / mhat^1_{k,i}) * sqrt( vhat^1_{k,i} / vhat^g_{k,i} )
         = U_{g,i} / U_{1,i}.
```

**Theorem.** For rectified RAdam (same step index, `eps=0`, no weight decay):
```
U_{g,i} = h_{k,i} U_{1,i}   for every coordinate i,   i.e.  U_g = h_k ⊙ U_1.
```
Consequently,
```
U_g = s U_1  for some scalar s   ⟺   h_{k,i} = s on all effective coordinates.
```

*Proof.* Direct from the update formula: `U_{g,i}/U_{1,i} =
(mhat_g / mhat_1) · sqrt(vhat_1 / vhat_g) = h_{k,i}`. ∎

**Corollary 1 (constant-scale null, = P-R2).** If `G^g_j = a G^1_j` for all
`j <= k` (constant `a`), then `mhat^g = a mhat^1`, `vhat^g = a^2 vhat^1`, so
`h_{k,i} = 1` for all `i` → `U_g = U_1`. Verified: `h` std ~8e-6,
`R_opt ~1.9e-6` (machine-level).

**Corollary 2 (generic history breaking).** If `a_j` varies over time and
different coordinates have different temporal gradient compositions, then in
general `h_{k,i} != h_{k,l}` for some `i,l`, so `U_g` is NOT a scalar multiple
of `U_1` and `R_opt(k) > 0` — **even when every instantaneous gradient residual
is zero**. This is the sharp prediction:
> instantaneous gradient near-scalar   ⇏   history-conditioned optimizer
> update near-scalar.

**Not a blanket implication:** time-varying `a_j` does NOT by itself force
`R_opt > 0` (e.g. a single-parameter model, or all gradients along one fixed
direction, keep `h` coordinate-constant). The theorem is the iff statement:
non-scalar breaking ⇔ `h_k` coordinate-varying.

**Analytic residual.** Let `w_i = U_{1,i}^2`. Since `U_{g,i} = h_i U_{1,i}`:
```
s* = <U_g,U_1>/||U_1||^2 = sum_i w_i h_i / sum_i w_i   (update-energy-weighted mean of h)
||U_g - s* U_1||^2 = sum_i w_i (h_i - s*)^2             (weighted dispersion of h)
R_opt(k) = sqrt( weighted variance of h_k ) / ||U_1||
```
So the non-scalar residual is exactly the **weighted dispersion of the
coordinate-wise history gauge**.

**History dispersion statistic `H_k` (exact, for Role D):**
```
H_k^2 := sum_i w_i (h_{k,i} - s*_k)^2 / sum_i w_i        (weighted variance)
```
Under the idealized rectified-RAdam assumption, `H_k` is proportional to the
update residual `R_opt(k)` (up to `||U_1||` normalization), so it is a
directly measurable internal-theory quantity for the three-arm study.

**Numeric anchor** (`theory/test_radam_history_gauge.py`, real RAdam):
- identity `U_{g,i} = h_{k,i} U_{1,i}` holds to rel-err **9.2e-8**;
- constant `a`: `h` coordinate std ~8e-6, `R_opt ~1.9e-6` (P-R2 null);
- time-varying `a` (alternating 1.3/0.8, 20-step blocks, instantaneous
  residual = 0 by construction): `h` coordinate std 0.95-3.4, `R_opt` 0.14-0.27
  — a **synthetic existence example / mechanism check** (not "all time-varying
  `a_j` imply breaking").

---

## Appendix A — Trajectory perturbation bound (P-R3-trajectory, lemma)

Standard Lipschitz recursion (kept as a lemma, not a headline):

```
D_{k+1} <= eps_k + L_k D_k   =>   D_K <= sum_{j=0}^{K-1} (prod_{ell=j+1}^{K-1} L_ell) eps_j
```
where `L_k` is the state-Lipschitz constant of the reference dynamics.
Numeric check (real RAdam): convex quadratic contracts (amplification 0.72),
non-convex MLP transient amplifies up to ~1.4x. This is a standard
perturbation bound; it does NOT estimate ECT's `L_k` nor connect to FID.

---

## 6. Non-trivial terms (open, acknowledged)

Bias correction (`1/(1-beta^{n+1})`, cancels under paired `n`); `eps > 0`
(`O(eps/sqrt(vhat))`, largest early); weight decay (`-eta wd theta`, drifts
`theta` so `a` not exactly constant); time-varying `a_k` (this IS P-R3);
non-scalar residual `E_g` (~0.3% whole-model / ~3.8% layer); AMP skipped steps
(cancel under paired execution, but `s_k` may diverge); nonzero `m_0,v_0`.

---

## 7. What this establishes / does not

**Establishes:**
- The two-phase structure with exact `c^*` values (`1/a` unrectified, `1`
  rectified), verified on single-step updates (rev.2, bug fixed).
- The **null theorem**: constant gradient rescaling is absorbed by rectified
  RAdam.
- **History-induced gauge breaking**: time-varying scale produces update
  residual even with zero instantaneous gradient residual (numeric anchor).

**Does not establish:**
- Not a convergence/FID theorem; the trajectory bound is a lemma.
- Does not estimate ECT's actual `L_k` or connect residual → FID (three-arm
  study's job).
- `a_j`'s actual time-variation in real ECT training is unmeasured (Role D can
  estimate it via `a_k^*` per checkpoint).

---

## 8. Consequence for Role D's experiment

Instead of expecting a slow `c_K^*` drift from RAdam warm-up (which completes
in ~5 steps), measure:
- `a_k^*` (instantaneous raw-gradient scale) at each state 0/32/64/128/256 kimg
  — expect it roughly stable;
- `c_k^*` (history-conditioned update gauge) — expect it to deviate from
  `1/a_k^*` if `a_j` varies;
- `H_k` history statistic — test whether it predicts the update residual.

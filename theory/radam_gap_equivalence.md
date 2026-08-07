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

## 5. P-R3 — History-induced gauge breaking (the headline)

**Setup.** The gradient is NOT a constant scalar rescaling:
```
G^g_j = a_j G^1_j + E_j,   with a_j varying over time (E_j small).
```
RAdam moments are history-weighted sums:
```
m_k^g = (1-beta1) sum_{j<=k} beta1^{k-j} a_j G^1_j + ...
v_k^g = (1-beta2) sum_{j<=k} beta2^{k-j} a_j^2 (G^1_j)^2 + ...
```

**Claim.** If `a_j` is not constant over the recent history (`a_j != a_{j-1}`),
then **no single scalar `a` makes both** `m_k^g = a m_k^1` **and**
`v_k^g = a^2 v_k^1` **hold simultaneously** (the first-moment sum weights
`a_j`, the second weights `a_j^2`; they cannot share one factor when `a_j`
varies). Hence the best scalar gauge leaves a residual:
```
R_opt(k) = ||U_g(k) - c_k^* U_1(k)|| / ||U_1(k)|| > 0
```
even when **every instantaneous raw-gradient residual `E_j` is zero**.

**This is the sharp prediction:**
> instantaneous gradient near-scalar   ⇏   history-conditioned optimizer
> update near-scalar.

The genuine GFCT signal is **scale history**, not the current gradient scale.

**Numeric anchor** (real RAdam, `a_j` alternates 1.3 / 0.8 in 20-step blocks,
instantaneous residual = 0 by construction):

| phase | residual |
|---|---|
| inside a constant-`a` block | **0.0000** (scalar fully absorbed) |
| just after `a_j` changes | **0.24 - 0.30** (history can't track new scale) |
| residual jumps | +0.24, +0.10, +0.16 (each change) |

**History statistic `H_k` (proposed for Role D):**
```
H_k = (effective second-moment scale) / (effective first-moment scale) - 1
```
intended to predict the update residual `R_opt(k)` from the mismatch between
the two history-weighted scale signals.

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

# RAdam Gap Equivalence — Proposition Draft (Role C)

Date: 2026-08-07. Branch: `theory/radam-gap-equivalence`.
Part of the GFCT main line: **separate optimizer reparameterization from genuine
gap effects in the optimizer-update space**.

---

## 1. Setup and notation

State-augmented optimizer update (GFCT §5.1):

```
z_k = (theta_k, m_k, v_k, n_k, s_k, EMA_k)
```

- `theta`: network parameters
- `m`, `v`: RAdam first/second moments (bias-corrected)
- `n`: step index
- `s`: GradScaler / AMP state
- `EMA`: exponential moving average of parameters

Real update under gap `g`:

```
theta_{k+1} = Phi_g(z_k, xi_k),   U_g(z, xi; eta) = theta_g^+ - theta
```

One-dimensional optimal match of `U_g` to `U_1`:

```
c_g^*(z) = argmin_{c>0} || c U_1 - U_g ||^2 = <U_g, U_1> / ||U_1||^2
```

Raw-gradient observation (previous work, not re-derived today):
`mu_g = a_g mu_1` with `a_{1.3} ≈ 0.770` (256 kimg, whole-model residual 0.30%),
so at the gradient level `g` is near a scalar rescaling.

---

## 2. RAdam update structure

PyTorch `torch.optim.RAdam` (`betas=(0.9,0.999)`, `eps=1e-8`) update:

```
m_k = beta1 m_{k-1} + (1-beta1) g_k
v_k = beta2 v_{k-1} + (1-beta2) g_k^2
mhat = m_k / (1 - beta1^{n+1})          # bias-corrected first moment
vhat = v_k / (1 - beta2^{n+1})          # bias-corrected second moment

rho_inf = 2/(1-beta2) - 1
rho_k   = ( (n+1)/(1-beta2) - 2 ) / ...   # rectification denominator
r_k     = sqrt( (rho_k-4)(rho_k-2) rho_inf / (rho_inf-4)(rho_inf-2) rho_k )  # <= 1

theta_{k+1} = theta_k - eta * r_k * mhat / (sqrt(vhat) + eps)
```

Two regimes:
- **Unrectified** (`r_k ≈ 1`, early steps / before rectification activates):
  `update ≈ eta * mhat`, a pure first-moment scaling.
- **Rectified** (`r_k < 1`): `update ∝ mhat / sqrt(vhat)`.

---

## 3. Proposition P-R1 (unrectified branch)

**Setup.** Assume (i) the per-step gradient satisfies `G^g = a G^1` for a fixed
`a > 0` (raw-gradient scalar equivalence), (ii) moments are initialized
correspondingly: `m^g_0 = a m^1_0`, `v^g_0 = a^2 v^1_0`, (iii) `eps = 0`,
(iv) weight decay = 0, (v) in the unrectified regime `r_k = 1`.

**Claim.** Then for all `k` (induction):
```
m_k^g = a m_k^1,   v_k^g = a^2 v_k^1
```
and the parameter update scales linearly:
```
U_g = a U_1   =>   c_g^* = a   =>   eta_g = eta_1 / a  (to match U_g = U_1)
```

*Proof sketch.* Induction: `m_1^g = beta1 m_0^g + (1-beta1) G^g = a(beta1 m_0^1 + (1-beta1) G^1) = a m_1^1`;
`v` likewise with `(G^g)^2 = a^2 (G^1)^2`. Since `r_k=1` and `eps=0`, the update is
`eta * mhat` (or `eta * mhat/sqrt(vhat)` with `vhat` irrelevant when... — in the
pure first-moment regime, only `m` matters), so `U_g = a U_1`. Matching
`eta_g U_g = eta_1 U_1` gives `eta_g = eta_1 / a`.

**Consequence.** In the fresh-state / unrectified phase, `c_g^*` should track the
raw-gradient `1/a_g` value. **This is the "null" expectation for Role D's
fresh-state audit**: `c_0^* ≈ 1/a^* ≈ 1/0.77 ≈ 1.30` for g=1.3. If the audit
finds this, it is NOT evidence against GFCT; it is the predicted early-phase
behavior.

---

## 4. Proposition P-R2 (rectified branch)

**Setup.** Same as P-R1 except the update is in the rectified regime
(`r_k < 1`, `update ∝ mhat/sqrt(vhat)`).

**Claim.** Under `m^g = a m^1`, `v^g = a^2 v^1`:
```
mhat_g / sqrt(vhat_g) = (a mhat_1) / sqrt(a^2 vhat_1) = mhat_1 / sqrt(vhat_1)
```
i.e. the **ideal positive scalar a cancels** in the ratio. Therefore, up to the
rectification factor `r_k` (same in both), `U_g ≈ U_1` at the SAME learning rate.

**Consequence.**
```
unrectified:  eta_g = eta_1 / a   (needed to match)
rectified:    eta_g ≈ eta_1       (already matched)
```
**Key testable prediction (the GFCT core):**
> **A single fixed LR multiplier cannot, in general, match both the early
> (unrectified) and later (rectified) RAdam trajectories.**

This is the theoretical source of `c_K^*` drift as training progresses, and it is
the sharpest new proposition of the GFCT line: the "gap = optimizer-step
rescaling" equivalence holds only in one RAdam phase; in the other phase the
adaptive preconditioner re-absorbs the scalar.

---

## 5. Approximate trajectory bound (P-R3)

Let the state residual after best matching satisfy
```
|| Phi_g(z, xi; c_g^* eta) - Phi_1(z, xi; eta) || <= eps_k,
```
and let the reference dynamics be `L_k`-Lipschitz in state. Standard recursion
(report §5.3) gives
```
|| z_K^g - z_K^1 || <= sum_{j=0}^{K-1} ( prod_{ell=j+1}^{K-1} L_ell ) eps_j.
```
This is **not** an FID theorem; it is a checkable statement that a small
per-step residual `eps_j` can be *amplified* by the product of Lipschitz
constants over the trajectory. The empirical question for the three-arm study:
is there a training phase where `prod L` is large enough to turn the ~0.3-3.8%
gradient residual into a finite-budget quality difference?

---

## 6. Non-trivial terms (open, acknowledged, not proven today)

The clean scalar-cancellation argument assumes away:

1. **Bias correction** (`mhat = m/(1-beta1^{n+1})`): at early `n` this is a
   strong `~1/n` factor; it is the same for both arms only if the arm-wise `n`
   is identical (it is, under paired execution) — so it cancels, but must be
   stated.
2. **`eps > 0`**: `1/(sqrt(vhat)+eps)` vs `1/sqrt(vhat)` breaks exact
   cancellation; the residual is `O(eps / sqrt(vhat))`, largest when `vhat` is
   small (early phase).
3. **Weight decay**: adds `-eta wd theta` per step; under `theta^g = a theta^1`
   it scales as `a`, so it is absorbed into `c_g^*` — but `theta` itself drifts,
   so `a` is not exactly constant.
4. **Time-varying `a_k`**: the raw-gradient scalar `a_g` is not exactly constant
   across the trajectory (measured whole-model residual is small but nonzero).
5. **Non-scalar residual `E_g`**: the ~0.3% (whole-model) / ~3.8% (per-layer)
   deviation is the *genuine* signal; the propositions are about the scalar part.
6. **AMP / GradScaler skipped steps**: a skipped step means `U_g = U_1 = 0` for
   that step regardless of g — both arms skip together under paired execution,
   so it cancels at the update level, but the scaler state `s_k` may diverge
   across arms if finite-gradient statistics differ (open).
7. **Nonzero moment initialization**: P-R1 assumes `m_0, v_0` matched; the
   fresh-state audit sets `m_0 = v_0 = 0` for both arms, so this is satisfied
   at step 0; later `c_K^*` drift is the object of study.

---

## 7. What this proposition does / does not establish

**Establishes:**
- A clean two-phase structure: unrectified phase is gradient-scalar-matched
  (`eta_g = eta_1/a`), rectified phase re-absorbs the scalar (`eta_g ≈ eta_1`).
- The **key falsifiable prediction**: a fixed LR multiplier cannot match both
  phases; `c_K^*` should drift as rectification activates.

**Does not establish (honest scope):**
- Not a convergence/FID theorem (P-R3 is a trajectory bound, not an end-quality
  bound).
- Not a proof that the residual `E_g` predicts FID — that is the three-arm
  causal experiment's job.
- The non-trivial terms in §6 are acknowledged and left open.

---

## 8. Relation to Role D's fresh-state audit

Role D computes (PR #42, fresh RAdam, `m_0=v_0=0`, step=0, paired minibatch):
```
c_0^* = <Delta_{1.3}, Delta_1> / ||Delta_1||^2,   rho_0 = ||Delta_1|| * ||c_0^* Delta_{1.3} - Delta_1|| / ...
```
**Prediction P-R1**: in the fresh/unrectified state, `c_0^* ≈ 1/a^* ≈ 1.30` for
g=1.3, with small residual. **This is the expected null**, not a GFCT failure.
The GFCT-relevant quantity is whether `c_K^*` drifts toward 1 as rectification
activates (later states 32/64/128/256 kimg).

---

## Files
- `theory/radam_gap_equivalence.md` — this document.
- `theory/test_radam_equivalence.py` — numeric checks of P-R1/P-R2 (see repo).

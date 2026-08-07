# RAdam Gap Equivalence — Proposition Draft (Role C, rev.4)

Date: 2026-08-07. Branch: `theory/radam-gap-equivalence`.
Part of the GFCT main line: **separate optimizer reparameterization from genuine
gap effects in the optimizer-update space.**

rev.4 closes the remaining theorem/test review points: (i) the coordinate-wise
history gauge is defined on the effective reference-update support; (ii) the
zero-reference support condition is explicit; (iii) the history-dispersion
statistic is normalized so that it equals the reference-normalized optimizer
residual exactly under the idealized rectified-RAdam assumptions.

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

## 2. Two scalar gauges (do not conflate)

Let `U_g` be the single-step parameter update of arm `g`, `U_1` the reference.

**Update scale** (fit the reference update direction to the candidate):
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

`c_g^*` is the learning-rate multiplier convention used by PR #42. `s_g^*` is
kept separately as the update-scale projection.

**Raw-gradient observation** (previous work, not re-derived): `mu_g = a_g mu_1`
with `a_{1.3} ≈ 0.770` (256 kimg, whole-model residual 0.30%), so at the
gradient level `g` is near a scalar rescaling.

---

## 3. P-R1 — Unrectified scale equivariance

**Setup.** (i) `G^g = a G^1` (constant `a > 0`); (ii) matched moments
`m_0^g = a m_0^1`, `v_0^g = a^2 v_0^1`; (iii) `eps = 0`, weight decay = 0;
(iv) unrectified regime (update `∝ mhat`).

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
Bias correction uses the same step index in both arms, so
`mhat^g = a mhat^1`, `vhat^g = a^2 vhat^1`. In the unrectified branch the
update is proportional to `mhat`, hence `U_g = a U_1` and `c_g^* = 1/a`. ∎

**Numeric anchor** (real `torch.optim.RAdam`, `G_g = a G_1`, `a=1.3`,
single-step updates `u = theta_{k+1} - theta_k`):

| step | c\* |
|---:|---:|
| 0-4 | **0.7692** (= 1/a = 1/1.3) |
| ≥ 10 | 1.0000 |

The unrectified→rectified transition occurs within the first few optimizer
steps, not over 32/64/128/256 kimg. The earlier cumulative-displacement drift
was retracted in rev.2.

---

## 4. P-R2 — Rectified constant-scale invariance (null theorem)

**Setup.** Same as P-R1 but the update is in the rectified regime
(`update ∝ mhat / sqrt(vhat)`), and the scale is constant over the *entire
history*: `G^g_j = a G^1_j` for all `j <= k`.

**Claim (null theorem).**
```
mhat_g / sqrt(vhat_g) = (a mhat_1) / sqrt(a^2 vhat_1) = mhat_1 / sqrt(vhat_1)
=> U_g = U_1   (up to eps and weight decay; the rectification factor is shared)
=> s_g^* = 1,   c_g^* = 1.
```
A **constant** positive gradient rescaling is absorbed by rectified RAdam.

**Numeric anchor:** after rectification, the single-step update scale and LR
multiplier both approach `1.0000` under the ideal constant-scale construction.

**Key consequence for the GFCT line:**
> A constant scalar change in the gradient belongs to the rectified-RAdam null
> equivalence class. The interesting object is therefore not warm-up, but the
> optimizer-state history that can make the effective coordinate gauges differ.

---

## 5. P-R3 — Coordinate-wise history gauge theorem (headline)

**Setup.** Rectified RAdam, identical step index across arms, `eps=0`, and no
weight decay. Let

```
S_k := { i : U_{1,k,i} != 0 }
```

be the effective support of the reference update. For each `i in S_k`, define

```
h_{k,i} := U_{g,k,i} / U_{1,k,i}
         = (mhat^g_{k,i} / mhat^1_{k,i})
           * sqrt(vhat^1_{k,i} / vhat^g_{k,i}).
```

**Theorem (coordinate-wise history gauge).** For every `i in S_k`,

```
U_{g,k,i} = h_{k,i} U_{1,k,i}.
```

Moreover, for a scalar `s`,

```
U_{g,k} = s U_{1,k}
```

if and only if both conditions hold:

1. `h_{k,i} = s` for every `i in S_k`; and
2. `U_{g,k,i} = 0` for every `i not in S_k`.

Thus, when both arms share the same effective support, scalar update
equivalence is exactly equivalent to the coordinate history gauge being
constant on that support.

*Proof.* On `S_k`, the first identity follows directly from the rectified update
formula because the step-dependent rectification factor and learning rate are
shared. If `U_g=sU_1`, the two support conditions follow coordinate-wise.
Conversely, if both support conditions hold, then every coordinate satisfies
`U_{g,i}=sU_{1,i}`. ∎

### Corollary 1 — Constant-scale null (= P-R2)

If `G^g_j = a G^1_j` for all `j <= k` with constant `a>0`, then
`mhat^g = a mhat^1`, `vhat^g = a^2 vhat^1`, so `h_{k,i}=1` on `S_k` and the
candidate update vanishes wherever the reference update vanishes. Hence
`U_g=U_1`.

### Corollary 2 — Generic history breaking

If the scalar relation varies over time and different coordinates have
heterogeneous temporal gradient histories, then the resulting moment ratios can
produce `h_{k,i} != h_{k,l}` on the effective support. In that case no single
learning-rate scalar can match the full update, even when the instantaneous raw
gradient residual is zero.

This is a **generic mechanism statement**, not a blanket implication. A
single-parameter model or a fixed-direction gradient history can keep `h_k`
coordinate-constant despite time-varying scales.

The sharp prediction is therefore:

> instantaneous gradient near-scalar does not imply history-conditioned
> optimizer update near-scalar.

### Exact analytic residual

Let `w_i = U_{1,k,i}^2` for `i in S_k`. The optimal update-scale projection is

```
s_k^* = <U_g,U_1> / ||U_1||^2
      = sum_{i in S_k} w_i h_{k,i} / sum_{i in S_k} w_i.
```

The exact squared projection residual is

```
||U_g - s_k^* U_1||^2
  = sum_{i in S_k} w_i (h_{k,i} - s_k^*)^2
    + sum_{i not in S_k} U_{g,k,i}^2.
```

Define the reference-normalized optimizer residual

```
R_opt(k) := ||U_g - s_k^* U_1|| / ||U_1||.
```

Define the **history-gauge dispersion**

```
H_k^2 := [ sum_{i in S_k} w_i (h_{k,i} - s_k^*)^2
           + sum_{i not in S_k} U_{g,k,i}^2 ]
         / sum_{i in S_k} w_i.
```

Then, under the idealized rectified-RAdam assumptions,

```
H_k = R_opt(k)     (exactly).
```

When both arms share support, the second term vanishes and `H_k` is simply the
update-energy-weighted standard deviation of the coordinate-wise history gauge.
This gives Role D a directly measurable internal-theory quantity rather than a
heuristic proxy.

### Numeric anchor

`theory/test_radam_history_gauge.py` checks the idealized identities with real
`torch.optim.RAdam` (the implementation has its default small `eps`, so the
moment-formula identity is numerical rather than symbolic exactness):

- constant scalar history gives coordinate-constant `h`, update scale `s*≈1`,
  and negligible `R_opt` after rectification;
- an alternating scale history (`1.3/0.8`) with zero instantaneous directional
  residual provides a **synthetic existence/mechanism example** in which
  `h_k` becomes coordinate-varying and `R_opt>0`;
- the direct projection residual is checked against the analytic weighted-
  dispersion expression and the exact normalization `H_k=R_opt`.

---

## Appendix A — Trajectory perturbation bound (lemma)

The finite-horizon propagation statement remains a standard Lipschitz lemma,
not a headline theorem:

```
D_{k+1} <= eps_k + L_k D_k
=> D_K <= sum_{j=0}^{K-1} (prod_{ell=j+1}^{K-1} L_ell) eps_j.
```

It only states that local optimizer residuals can accumulate or contract under
the state dynamics. It does not estimate real ECT Lipschitz constants and is
not an FID theorem.

---

## 6. Non-trivial terms left open

The clean propositions intentionally isolate the scalar/history mechanism.
Real ECT additionally contains:

1. `eps > 0`, which breaks exact second-moment scale cancellation when
   `sqrt(vhat)` is small;
2. weight decay and parameter drift;
3. time-varying scalar fits and genuine non-scalar raw-gradient residuals;
4. AMP/GradScaler skipped-step and scaler-state effects;
5. nonzero optimizer-state initialization at later checkpoints;
6. support changes and near-zero update coordinates.

These terms should be measured rather than silently folded into the theorem.

---

## 7. What this establishes / does not establish

**Establishes:**
- unrectified scale equivariance under constant positive gradient scaling;
- rectified constant-scale invariance as the optimizer null class;
- the coordinate-wise history gauge identity and an exact iff criterion for
  scalar update equivalence;
- an exact decomposition of the best scalar projection residual into
  history-gauge dispersion plus any off-support candidate update;
- `H_k = R_opt(k)` under the idealized assumptions.

**Does not establish:**
- convergence or an FID guarantee;
- that real ECT necessarily exhibits large history-gauge dispersion;
- that any measured optimizer residual causally explains generation quality;
- that time-varying scalar scale alone always causes gauge breaking.

Those are empirical questions for the paired optimizer-state diagnostic and the
three-arm causal experiment.

---

## 8. Consequence for Role D

At states such as 0/32/64/128/256 kimg, measure together:

- `a_k^*`: instantaneous raw-gradient scalar fit;
- `s_k^*` / `c_k^*`: best scalar update fit and candidate LR multiplier;
- coordinate- or layer-aggregated `h_{k,i}`;
- `H_k`, which should numerically agree with `R_opt(k)` under the idealized
  decomposition and expose deviations introduced by real optimizer details.

The GFCT experimental chain is therefore:

```
raw-gradient scalar gauge
    -> optimizer-history coordinate gauge h
    -> gauge dispersion H_k (= idealized R_opt)
    -> finite-horizon trajectory / quality divergence (empirical test).
```

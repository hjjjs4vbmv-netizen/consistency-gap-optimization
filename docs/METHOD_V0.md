# Method V0: Factorized Global–Local Gap Control for ECT

## 1. Scope and notation

Let `t > 0` denote the sampled noise level, and let
`r_sigmoid(t; m)` denote the official ECT sigmoid mapping at curriculum stage
`m`. Define its absolute training-pair gap as

\[
d_{\mathrm{base}}(t)
= t-r_{\mathrm{sigmoid}}(t;m).
\]

The official implementation uses

\[
n(t)=1+k\,\sigma(-bt),\qquad
r_{\mathrm{sigmoid}}(t;m)
=\max\!\left(0,\,
t\left[1-\frac{n(t)}{q^{m+1}}\right]\right).
\]

All proposed schedules retain this `t`-dependent sigmoid mapping as the
baseline.

## 2. Factorized intervention

Let `b(t) ∈ {1,…,B}` denote the bin containing `t`. Before constraints needed
to realize a valid pair are applied, the proposed gap is

\[
d_{\mathrm{pre}}(t)
=g\,\ell_{b(t)}\,d_{\mathrm{base}}(t),
\]

where:

- `g > 0` is a global calibration factor shared by every noise level;
- `ℓ_i > 0` is the local scale for bin `i`;
- the local scales satisfy

\[
\frac{1}{B}\sum_{i=1}^{B}\log \ell_i=0,
\]

or equivalently

\[
\left(\prod_{i=1}^{B}\ell_i\right)^{1/B}=1.
\]

This constraint applies to the local scale factors before realized-gap
clipping effects. It does not imply that the realized training gaps have an
unchanged arithmetic mean, geometric mean, or expected value.

## 3. Quantile timestep bins

ECT samples `log t` from a normal distribution,

\[
\log t\sim\mathcal N(P_{\mathrm{mean}},P_{\mathrm{std}}^2).
\]

The controller partitions this distribution into `B=4` quantile bins, so that

\[
\Pr[t\in\mathcal B_i]\approx \frac{1}{B}.
\]

Quantile bins provide approximately balanced sample counts and more stable
per-bin exponential moving averages than equal-width bins in `t` or `log t`.

## 4. Local feedback signal

For each training pair, the controller observes the unweighted squared pair
loss

\[
L_{\mathrm{raw}}
=\left\|
f_\theta(x_t,t)
-\operatorname{sg}\!\left[f_\theta(x_r,r)\right]
\right\|_2^2.
\]

It does not use the final ECT-weighted objective as its feedback signal. This
choice avoids treating variation in the prescribed loss weighting as if it
were variation in pair-learning difficulty.

For bin `i`, let `S_i` and `L_i` denote short- and long-horizon EMAs of the raw
loss. The unnormalized update is

\[
\tilde{\ell}_i
=\operatorname{clip}\!\left(
\exp\left[-\eta
\tanh\!\left(\log L_i-\log S_i\right)\right],
\ell_{\min},\ell_{\max}
\right),
\]

with a deadband around zero trend. The vector of log scales is then projected
onto the bounded zero-mean set, yielding `ℓ_i` with
`B^{-1} Σ_i log ℓ_i = 0`.

## 5. Realized gap and clipping

The factorized equation defines the intended pre-clipping gap. The
implementation must also guarantee `0 ≤ r ≤ t` and, for local schedules, a
minimum relative gap `δ_min`. For `local_tbin_v3`, the implemented sequence is

\[
d_{\mathrm{local}}(t)
=t\,
\operatorname{clip}\!\left(
\ell_{b(t)}\frac{d_{\mathrm{base}}(t)}{t},
\delta_{\min},1
\right),
\]

\[
d_{\mathrm{realized}}(t)
=\min\!\left(t,\,g\,d_{\mathrm{local}}(t)\right),
\qquad
r_{\mathrm{new}}(t)=t-d_{\mathrm{realized}}(t).
\]

Consequently,

\[
d_{\mathrm{realized}}(t)
\neq g\,\ell_{b(t)}\,d_{\mathrm{base}}(t)
\]

whenever a lower or upper bound is active. The accurate claim is:

> The local scale factors have geometric mean one before realized-gap clipping
> effects.

It is not accurate to claim that the local controller always preserves the
global realized training gap.

## 6. Experimental arms

| Arm | Global factor `g` | Local factors `ℓ_i` |
| --- | ---: | --- |
| Fixed sigmoid | `1` | all `1` |
| Global-only | selected `g*` | all `1` |
| Local-only | `1` | adaptive, geometric mean `1` before clipping |
| Global + local | selected `g*` | adaptive, geometric mean `1` before clipping |

Stage 1 selects `g*` from the seed-0 response curve. Stage 2 evaluates the
factorial arms at `g*` with training seeds 0, 1, and 2. Because seed 0
participates in selection, the headline confirmation comparison uses only
held-out seeds 1 and 2.

## 7. Reported quantities

For metric `M`, method `A`, and held-out set `H={1,2}`, the headline percentage
is

\[
\Delta_{\mathrm{headline}}(A,M)
=100\left[
\frac{\frac{1}{|H|}\sum_{s\in H}M_{A,s}}
{\frac{1}{|H|}\sum_{s\in H}M_{\mathrm{fixed},s}}
-1
\right].
\]

This is the percentage difference between held-out-seed arithmetic metric
means. It is not

\[
\frac{1}{|H|}\sum_{s\in H}
100\left(\frac{M_{A,s}}{M_{\mathrm{fixed},s}}-1\right).
\]

Training-time gap diagnostics are defined as

\[
R_{\mathrm{gap}}
=\mathbb E\left[
\frac{d_{\mathrm{realized}}(t)}
{d_{\mathrm{base}}(t)}
\right],
\]

\[
C_{\mathrm{lower}}
=\mathbb E\left[
\mathbf 1\{d_{\mathrm{realized}}>d_{\mathrm{pre}}+\tau\}
\right],
\qquad
C_{\mathrm{upper}}
=\mathbb E\left[
\mathbf 1\{d_{\mathrm{realized}}<d_{\mathrm{pre}}-\tau\}
\right],
\]

where `τ` is a floating-point tolerance. These are logged as
`gap_over_sigmoid_gap_mean`, `lower_gap_clip_rate`, and
`upper_gap_clip_rate`.

## 8. Current evidence and limitation

The completed 2026-07-27 runs were produced before the three realized-gap
telemetry fields were added. Exact training-time clipping rates and
`realized gap / sigmoid_gap` means are therefore unavailable for those runs and
must not be inferred from KID/FID summaries. The updated implementation records
them for future runs and marks older validation artifacts
`not_recorded_pre_instrumentation`.

KID-5k and FID-5k are 5,000-sample proxy metrics. With three training seeds,
the reported intervals and factorial contrasts are descriptive rather than
population-level significance claims.

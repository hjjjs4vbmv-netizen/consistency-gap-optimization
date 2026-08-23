# q, g, and effective pair spacing

**Status:** theory clarification for PR #81. This note specifies a future
control; it does not preregister or authorize a new run.

## Claim ceiling

The current study does **not** establish that the effect of $g$ is independent
of $q$. The experiment fixes $q=256$ and uses $g=1.10$ as a controlled probe of
the realized pair spacing. The scientific object is the pair-spacing
intervention, not a distinct mechanism attached to the symbol $g$.

## Definitions in the current ECT implementation

Let $j=0,1,\ldots$ denote the curriculum stage and
$n(t)=1+k\sigma(-bt)$. With the repository's stage convention, the official
sigmoid mapping gives

$$
r_q(t,j)=\max\left\{0,t\left(1-\frac{n(t)}{q^{j+1}}\right)\right\},
\qquad
\Delta_q(t,j)=t-r_q(t,j)
=\min\left\{t,\frac{t n(t)}{q^{j+1}}\right\}.
$$

The global probe operates on the realized baseline spacing:

$$
\Delta_{q,g}(t,j)=\min\{t,g\Delta_q(t,j)\},
\qquad
r_{q,g}(t,j)=t-\Delta_{q,g}(t,j).
$$

Thus $q$ and $g$ both parameterize the same realized pair. At a fixed stage and
away from clipping,

$$
q_{\mathrm{eff}}(j)=\frac{q}{g^{1/(j+1)}}
$$

produces the same spacing as the $g$-scaled rule. A single fixed
$q_{\mathrm{eff}}$ generally cannot match one constant $g$ across multiple
stages. For the current enlargement $g=1.10\ge 1$, the stage-specific formula
also preserves equality through the production upper clamp; this must still be
verified on realized pairs rather than assumed.

We define the **effective pair spacing** of a training design as the joint law

$$
\mathcal{P}_{q,g}
=\operatorname{Law}\!\left(J,T,\Delta_{q,g}(T,J)\right)
$$

under the frozen stage curriculum and time sampler. It is not the mean gap, the
nominal value of $q$, or the nominal value of $g$. The coordinate used here is
the native ECT spacing $\Delta=t-r$; a log-SNR or solver-coordinate distance
would be a different estimand and must be declared separately.

## Why spacing is a composite ECT intervention

For the CIFAR-10 ECT objective,

$$
\ell(\theta;t,r)
=\frac{1}{t-r}
d\!\left(f_\theta(x_t,t),
\operatorname{sg}[f_\theta(x_r,r)]\right).
$$

Changing realized spacing moves the detached target from $r_1$ to $r_g$ and
changes the explicit weight from $1/\Delta_1$ to $1/\Delta_g$. PR #81
factorizes these two changes exactly under the one-sided stop-gradient training
derivative. It does not factorize $q$ and $g$ as independent causes.

## Strict future effective-spacing-matched control

**Question.** Does the parameterization label ($q$-rule versus $g$-wrapper)
matter after the realized pairs and weights are matched?

**Arm G (reference).** Use the production mapping with $q_0=256$ and
$g_0=1.10$.

**Arm Q-match.** Set $g=1$ and use the frozen stage-specific rule

$$
q_j^\star=q_0/g_0^{1/(j+1)}.
$$

The stage sequence $j(k)$, sampled $t$, production clamps, target/denominator
assignment, initialization, data order, noise, dropout, AMP, optimizer, EMA,
and all checkpoint/evaluation settings must be identical to Arm G.

**Admission gate before training.** On a frozen $t\times j$ grid and on paired
production minibatches, require:

1. pointwise equality of realized $r$ and $\Delta$ within a declared FP32
   tolerance;
2. identical clipping indicators and target inputs $x_r$;
3. matching per-sample losses and one-sided gradients under shared RNG; and
4. no code path in which nominal $q$ changes the model, sampling law,
   optimizer, or loss outside construction of the matched pair.

A suitable numerical gate is

$$
\max_i
\frac{|\Delta_i^G-\Delta_i^Q|}{\max(1,|t_i|)}
\le 32\epsilon_{\mathrm{FP32}},
$$

together with a relative whole-gradient error below $10^{-6}$. Tolerances must
be frozen before any quality result is generated.

**Training estimand.** Because this is a parameterization-equivalence control,
the primary readout is paired trajectory equivalence: checkpoint parameter/EMA
distance and per-budget loss difference. FID/KID learning-curve differences
are secondary and should be tested against a preregistered equivalence margin,
not for superiority. Any material divergence indicates an unmatched
implementation path or numerical nondeterminism; it does not establish a
$q$-independent effect of $g$.

A natural fixed-$q'$ tuning comparison is a different, necessarily imperfect
control because one constant $q'$ cannot match the constant-$g$ rule at every
stage. If pursued later, $q'$ must be selected using only the frozen
$(J,T,\Delta)$ exposure distribution, with residual spacing mismatch reported
before training outcomes are inspected.

**Decision today:** specification only; do not launch this control.

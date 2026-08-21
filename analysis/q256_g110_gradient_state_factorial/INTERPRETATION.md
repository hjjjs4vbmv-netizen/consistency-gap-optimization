# Interpretation guardrail for the q256 g=1.10 gradient × RAdam-state factorial

This note fixes the claim boundary for the four-cell audit without changing any frozen measurements.

## Four cells

- **A — observed gradient / real state**: the natural q256 g=1.00/1.10 virtual-update comparison from the accumulated RAdam state.
- **B — observed gradient / reset moments**: the same observed gradient pair after zeroing `exp_avg` and `exp_avg_sq` while preserving per-parameter step and optimizer hyperparameters.
- **C — exact-scalar gradient / real state**: replace the observed g=1.10 gradient by `a* G_1.00`, while preserving the real accumulated RAdam state.
- **D — exact-scalar gradient / reset moments**: the exact-scalar pair under the reset-moment stress condition.

The cells are a controlled diagnostic factorial. They are **not** an additive causal decomposition, and contrasts such as `B-D` and `C-D` must not be ranked as percentages of a common total effect.

## Paper-safe interpretation

The strongest cross-seed statement supported by the audit is:

> Real accumulated RAdam state sustains substantial non-scalar update divergence even when the instantaneous g=1.10 gradient is projected to an exact scalar multiple of the g=1.00 gradient. Conversely, resetting both moments strongly amplifies the effect of the observed non-scalar gradient residual. The two factors therefore interact strongly rather than acting as separable monotonic sources of divergence.

At K=256, the seed-level median `R_opt` values are:

| Training seed | A observed/real | B observed/reset | C exact-scalar/real | D exact-scalar/reset |
|---:|---:|---:|---:|---:|
| 3 | 0.083028 | 0.477003 | 0.070350 | 0.001708 |
| 4 | 0.087762 | 0.490736 | 0.077187 | 0.001635 |
| 5 | 0.074052 | 0.466910 | 0.048855 | 0.000602 |

Thus the exact-scalar/real-state condition remains clearly non-zero across all three independent training trajectories, while the exact-scalar/reset condition is near zero up to the documented RAdam epsilon effect.

## What this does establish

The experiment supports the claim that stored optimizer state **causally modulates the mapping from a gap-induced gradient transformation to the next RAdam update** in these frozen virtual updates. It also shows that the simple model “optimizer memory monotonically amplifies divergence” is false: real accumulated state can preserve scalar-history sensitivity while simultaneously regularizing the much larger reset-state response to the observed stochastic non-scalar gradient residual.

## What this does not establish

Do not claim any of the following from this audit:

- optimizer state causes the observed FID/KID improvement;
- cell differences form an additive mediation decomposition;
- `C/A` is a percentage of divergence explained by optimizer memory;
- moment reset is a valid memory-neutral training intervention;
- audit minibatches are independent training replicates;
- the result predicts continuation-training or endpoint-quality behavior.

The independent replication unit is the training seed (`n=3`). The eight audit minibatches per seed are paired repeated measurements.

## Relation to the FID-closure program

This audit strengthens the local optimizer-state mechanism but does not close the endpoint-quality loop. In particular, the large `B` values show that zeroing both moments is an unsuitable full-training neutralizer. Any future quality intervention must first demonstrate, on the exact q256 g=1.00/1.10 treatment and across training seeds, that it selectively suppresses the exact-scalar/real-state readout (`C`) without creating the reset-state instability exposed here.

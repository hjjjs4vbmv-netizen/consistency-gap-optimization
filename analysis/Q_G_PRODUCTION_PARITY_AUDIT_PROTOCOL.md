# q--g production parity audit protocol

**Frozen before checkpoint measurement:** 2026-08-25 (Asia/Singapore)

## Question

In the current stage-0, unclipped ECT implementation, does the real-arithmetic
reparameterization

\[
(q,g)=(256,1.10)
\quad\Longleftrightarrow\quad
(q',g')=(256/1.10,1)
\]

remain numerically equivalent through the production pair, target, weight,
loss, and one-sided-gradient paths?

This is an implementation-equivalence diagnostic. It is not a training run
and does not test whether nominal `q` has an independent causal effect.

## Frozen design

1. Use the production `global_sigmoid` schedule with stage 0, `k=8`, `b=1`.
2. Compare the reference `(q=256, g=1.10)` with the candidate
   `(q=232.72727272727272, g=1)`.
3. Dense-grid gate: 8192 log-spaced positive `t` values from `1e-6` to `1e4`,
   evaluated in FP32 and FP64.
4. Network gate: online arm-A state, seed 3, 512 kimg; two real CIFAR-10
   minibatches of 16 examples with shared data order, `t`, noise, and dropout
   RNG.
5. Evaluate both the native network-precision path and a forced-FP32 reference
   path.
6. Construct no optimizer and perform no parameter update.

## Readouts

- realized `r` and `Delta=t-r`;
- target input `x_r` and detached target output;
- explicit weight `1/Delta`;
- per-sample ECT loss;
- one-sided parameter gradient.

Pair-coordinate error is normalized by `max(1, |t|)` and must not exceed
`32 * eps(dtype)`. The target-input coordinate uses its own elementwise scale.
The relative L2 errors for the weight, detached target output, per-sample loss,
and whole parameter gradient must each not exceed `1e-6`. These thresholds are
frozen before the checkpoint audit and will not be widened after observing the
result.

## Interpretation

A pass establishes numerical parity for the audited grid, state, batches, and
precision paths. It does not prove bitwise equality for every possible future
update. A failure identifies an implementation/numerical mismatch at the
frozen tolerance; it does not establish a scientific effect of nominal `q`.

Irrespective of the verdict, historical q256-B and q128-Bmatch runs used
different executions. Their quality difference cannot identify a
`q`-independent mechanism. The internally controlled q128 five-arm contrasts
remain valid within that experiment.


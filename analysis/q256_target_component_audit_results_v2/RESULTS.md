# q=256 target-component audit: budget-resolved results

Status: **PASS_PRIMARY_12_STATE_MATRIX** and **PASS_LOCAL_DOWNLOAD_INTEGRITY**.

## Question

At fixed model parameters and fixed stochastic inputs, does the realized
target-endpoint component

\[
\tau_{\mathrm{tar}} = G_B-sG_A,
\qquad s=1/1.1,
\]

systematically contract as training proceeds from 256 to 1024 kimg?

## Design

- States: arm A, seeds 3--5, budgets 256/512/768/1024 kimg.
- Probe: eight fixed batches of 16 examples; identical images, labels, time,
  noise, endpoint values, and dropout RNG realization in every state.
- Estimand: FP32 one-sided stop-gradient objective gradient.
- Primary whole-model metric:
  \(R_{\mathrm{tar}}=\lVert\tau_{\mathrm{tar}}\rVert_2/
  \lVert G_A\rVert_2\).
- The independent unit is the training seed. Budgets are repeated
  measurements, not additional replicates.

## Result

The target-endpoint component is small and nonzero in this fixed-batch
whole-model mean-gradient audit. Across all 12 states,
\(R_{\mathrm{tar}}\) ranges from 0.00174 to 0.00363. The best-fit scalar
\(a^\star\) remains close to the explicit denominator factor
\(s=0.90909\), while the exact factorial identity errors remain below
\(8.30\times10^{-6}\), well inside the frozen \(10^{-4}\) gate.

| Budget (kimg) | Mean \(R_{\mathrm{tar}}\) | SD across 3 seeds | Mean NFE1 \(\log\mathrm{FID}_B-\log\mathrm{FID}_A\) |
|---:|---:|---:|---:|
| 256  | 0.002529 | 0.000978 | -0.03633 |
| 512  | 0.002377 | 0.000529 | +0.02993 |
| 768  | 0.002414 | 0.000614 | -0.07544 |
| 1024 | 0.002530 | 0.000434 | -0.05127 |

The endpoint change in \(R_{\mathrm{tar}}\) is not seed-consistent:

| Seed | \(R_{\mathrm{tar}}(1024)-R_{\mathrm{tar}}(256)\) |
|---:|---:|
| 3 | -0.000602 |
| 4 | +0.000078 |
| 5 | +0.000526 |

Layerwise target components are also nonzero. Across states, the median
layerwise ratio is 0.00144--0.00298, the 90th percentile is
0.00396--0.00511, and the maximum is 0.00907--0.01228.

## Interpretation

The audit does not support a simple magnitude-contraction account in which
the target-endpoint gradient component systematically shrinks as training
budget increases. Instead, the instantaneous objective field remains close
to the explicit scalar rescaling, with a small target correction whose
magnitude and direction vary across seeds and budgets. The aligned NFE1 FID
contrasts also change sign and magnitude without a corresponding monotone
change in \(R_{\mathrm{tar}}\).

This result sharpens the paper's trajectory claim: instantaneous target
geometry is present, but its whole-model magnitude at a frozen state does not
explain the budget-resolved quality curves by itself. Finite-training
differences therefore require an account of accumulated trajectory feedback;
this audit does not identify optimizer causality.

The metric is computed from the gradient of the equal-weight mean over the
fixed batches. Batchwise target components can partially cancel, so the
whole-model result does not establish that every minibatch-level target
effect is negligible.

## Provenance and protocol amendment

The first matrix run used the v1 validator, which required the raw trajectory
configuration hash to be identical across budgets. It rejected the matrix
because the 256-kimg source run records `total_kimg=256`, whereas the
continuation states record `total_kimg=1024`. A read-only comparison found
`total_kimg` to be the sole configuration difference. Before inspecting the
matrix-level scientific summaries, Amendment 001 replaced this cross-budget
gate with a dynamics hash that removes only the top-level terminal horizon
and separately requires `trajectory_total_kimg >= audited budget`. The v1
outputs are preserved and excluded from the formal result. All formal v2
states were rerun after the amendment.

The server-side v2 matrix receipt binds every manifest and CSV by SHA256. The
local download integrity receipt verifies all 12 manifests and 24 CSV files
against it. Three server execution files differ from the current local
checkout; their exact hash-matching versions are preserved under
`implementation_snapshot/`.

## Paper-ready result paragraph

> Across arm-A states from seeds 3--5 and 256--1024 kimg, the exact
> denominator identities held to relative error below
> \(8.3\times10^{-6}\). The realized target-endpoint component was small but
> nonzero: its whole-model norm was 0.17--0.36% of the baseline gradient,
> with layerwise 90th percentiles of 0.40--0.51%. Its magnitude did not
> contract consistently with training budget: the 1024-minus-256 kimg change
> was negative for one seed and positive for two. Moreover, budget-resolved
> NFE1 FID contrasts changed in magnitude and sign without a corresponding
> monotone change in the target-component norm. Thus, the instantaneous
> target correction alone does not account for the observed learning-curve
> dynamics; the finite-training effect depends on the trajectory generated
> by repeatedly applying the composite intervention.


# Fresh q128 Regime and History Generalization Study v1 — Final Report

Generated: 2026-09-04T09:35:46.277740Z

## Governance headline

All 272 frozen evaluation jobs for the effective cohort passed the opaque sealing gate before unified decode. The frozen statistical analysis was executed without changing the cohort, metric, hypotheses, thresholds, NFE, budgets, or practical margin.

The scientific identity is **CONFIRMATORY_IDENTITY_FAILED_CLOSED**. There were 151 AMP GradScaler skips after the frozen 10,000-nimg warm-up bound. The frozen verifier explicitly classifies such events as fail-closed. Therefore the numerical results below are pre-specified, protocol-deviated fresh-cohort results, not a clean confirmatory replication. Loss, model, EMA, and applied-update states remained finite on the recorded skip events, but that does not waive the frozen rule.

## Sole primary: H_A

- Directional axis: **DIRECTION_UNRESOLVED**
- Mean H_A: -0.0320218; median: -0.0326354; SD: 0.0642948
- Two-sided 95% CI: [-0.0857735, 0.02173]
- One-sided 95% upper bound: 0.0110451
- One-sided paired-t p for E[H_A]&lt;0: 0.100881
- Negative seeds: 5/8; exact one-sided sign-flip p: 0.105469
- LOSO means: -0.0409051, -0.0301993, -0.0395144, -0.0322398, -0.0423731, -0.0152386, -0.0316285, -0.0240755

Directional evidence uses only the frozen one-sided paired t-test. The two-sided CI, sign count, sign-flip test, and LOSO results are robustness summaries and do not override that verdict.

## Independent practical-magnitude axis

- Verdict: **PRACTICAL_MAGNITUDE_UNRESOLVED**
- Equivalence band: ±log(1.03) = ±0.0295588
- Two-sided 90% CI: [-0.0750886, 0.0110451]
- TOST p-values, lower/upper: 0.541621, 0.0151208
- Material negative supported: **FALSE**

## Key secondary results

Phase shift P: **DIRECTION_UNRESOLVED**; mean -0.168172, 95% CI [-0.572165, 0.235821], one-sided p 0.178872, negative seeds 4/8, exact sign-flip p 0.226562.

NFE diagnostic R: mean 0.0885095, 95% CI [-0.0656971, 0.242716], positive/negative seeds 5/3. This supports only a statement about whether the Bmatch/Cmatch diagnostic trajectory ranking is NFE-dependent; it is not a causal channel attribution.

## Seed-level log-FID contrasts

| Seed | H_A | H_B | S_A | S_B | I | P | R |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 201 | 0.0301615 | 0.0491982 | 0.0309224 | 0.0499591 | 0.0190367 | 0.0499263 | 0.231522 |
| 202 | -0.0447791 | -0.0309246 | 0.00572048 | 0.019575 | 0.0138546 | 0.266859 | -0.1168 |
| 203 | 0.0204264 | 0.0152177 | 0.044707 | 0.0394984 | -0.00520865 | 0.0876699 | 0.105044 |
| 204 | -0.0304958 | -0.0573735 | 0.0262457 | -0.000631938 | -0.0268777 | -0.0120459 | -0.201797 |
| 206 | 0.0404376 | 0.0397526 | 0.0443973 | 0.0437122 | -0.000685045 | -0.0813108 | 0.300489 |
| 207 | -0.149504 | -0.125989 | 0.0222335 | 0.0457487 | 0.0235152 | -1.04293 | 0.245647 |
| 208 | -0.034775 | -0.0502864 | 0.0370202 | 0.0215088 | -0.0155114 | 0.191618 | -0.0300712 |
| 209 | -0.0876454 | -0.113101 | 0.0439664 | 0.0185109 | -0.0254554 | -0.805156 | 0.174042 |

H_B, S_A, S_B, and I are secondary/descriptive and cannot rescue the primary result. No interaction-equivalence claim is made because no independent interaction equivalence band was frozen.

## Missingness and replacement

- Formal seeds started: 9 (201–208 plus replacement 209).
- All-started terminal failure count: 1 (seed205).
- Arm-specific terminal failure: seed205 Dmatch, 1 event, deterministically reproduced at attempted iteration 7951 / 1,017,728 processed images.
- B-history-specific terminal failure count: 0.
- Effective complete cohort: seeds 201–204 and 206–209 (n=8); seed209 replaced seed205 under the pre-sorted pool.
- The estimand is completion-conditioned. Replacement does not eliminate possible informative missingness.

## Separation from the old q128 n=3 study

Only the fresh n=8 cohort enters inference. Old q128 seeds3–5 are not pooled, do not contribute to any p-value, and cannot change either verdict axis. Any later side-by-side comparison must be marked DESCRIPTIVE EVIDENCE SYNTHESIS.

## Integrity

The decode gate verified 272 unique opaque IDs, all SEALED_PASS receipts, effective-cohort membership, evaluator/source/dataset hashes, category coverage, KID/FID shared-feature identity, and the hashes of all sealed scalar artifacts. Full per-cell decoded values and descriptive cell summaries are retained in the immutable analysis artifacts accompanying this report.

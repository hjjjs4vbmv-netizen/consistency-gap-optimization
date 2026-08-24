# q128 matched-spacing five-arm results

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-24
- Verification Status: ANALYZED
- Version Label: q128_matched_spacing_results_v1

## Audit status

- 210/210 unique `SEALED_PASS` jobs and 420/420 metric values.
- FP32, 50,000 samples, sample seeds 0-49999, metric seed 20260730.
- NFE2 uses `mid_t=0.821`; invalidated/pre-reuse directories are excluded.
- Preassigned server/data partitions override redundant attempts without quality selection.

## Arm summaries

AULC is normalized trapezoidal area under the natural-log FID curve; lower is better.
Values below are three-seed means.

| Arm | NFE1 AULC | NFE2 AULC | 1024 NFE1 FID / KID | 1024 NFE2 FID / KID |
| --- | ---: | ---: | ---: | ---: |
| A | 3.1797 | 1.8980 | 8.798 / 0.005566 | 2.982 / 0.001105 |
| Bsame | 3.1997 | 1.8727 | 8.518 / 0.005373 | 2.956 / 0.001085 |
| Bmatch | 3.2619 | 1.9492 | 8.084 / 0.004928 | 2.749 / 0.000958 |
| Cmatch | 3.1291 | 1.8466 | 7.447 / 0.004426 | 2.816 / 0.000965 |
| Dmatch | 3.4220 | 2.2326 | 9.904 / 0.006313 | 2.975 / 0.001135 |

## Frozen AULC contrasts

Negative values favor the first named arm because lower AULC is better.

| Contrast | NFE | Mean | Median | Range | Negative seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bmatch-Bsame | 1 | 0.0622 | 0.0004 | [-0.0254, 0.2115] | 1/3 |
| Bmatch-Bsame | 2 | 0.0764 | -0.0383 | [-0.0510, 0.3185] | 2/3 |
| Bmatch-A | 1 | 0.0822 | 0.0243 | [-0.0270, 0.2494] | 1/3 |
| Bmatch-A | 2 | 0.0512 | -0.0642 | [-0.0657, 0.2835] | 2/3 |
| Bsame-A | 1 | 0.0200 | 0.0239 | [-0.0016, 0.0378] | 1/3 |
| Bsame-A | 2 | -0.0252 | -0.0275 | [-0.0350, -0.0132] | 3/3 |
| Cmatch-A | 1 | -0.0506 | -0.0558 | [-0.0618, -0.0342] | 3/3 |
| Cmatch-A | 2 | -0.0513 | -0.0430 | [-0.1161, 0.0051] | 2/3 |
| Dmatch-A | 1 | 0.2424 | 0.2536 | [0.0982, 0.3753] | 0/3 |
| Dmatch-A | 2 | 0.3347 | 0.3958 | [0.0344, 0.5739] | 0/3 |
| interaction | 1 | -0.1095 | -0.0634 | [-0.3168, 0.0516] | 2/3 |
| interaction | 2 | -0.2321 | -0.1742 | [-0.4651, -0.0571] | 3/3 |

## Interpretation

- The primary `Bmatch-Bsame` AULC contrast is not directionally stable: NFE1 is negative for 1/3 seeds and NFE2 for 2/3 seeds. Seed 3 has a large early-curve penalty.
- At 1024 kimg, `Bmatch` has lower FID and KID than `Bsame` for all three seeds at both NFEs.
- `Cmatch-A` improves NFE1 AULC for 3/3 seeds and NFE2 for 2/3 seeds.
- `Dmatch-A` is worse for 3/3 seeds at both NFEs.
- The outcome-level interaction is negative for 2/3 seeds at NFE1 and 3/3 at NFE2; it is not an objective-level causal decomposition.

## Direction consistency

FID and KID agree in direction for 36/42 `Bmatch-Bsame` cells. The terminal `Bmatch-Bsame` contrast agrees in all 6 seed-by-NFE cells.

## Exploratory TTQ

q128 TTQ was not preregistered. `ttq_exploratory.csv` therefore reports it only as a descriptive auxiliary analysis.

## Limitations and fallacy scan

- Three seeds are insufficient for population-level significance claims; no p-value is used as the primary narrative.
- The look-elsewhere and forking-path risks are mitigated by the frozen AULC contrasts; TTQ is explicitly labeled exploratory.
- Structural and causal fallacies are not indicated by this paired controlled design, but the 11/11 statistical fallacy checklist was reviewed.
- Verification status is `ANALYZED`, not `VERIFIED`, because metrics were not independently rerun in this report step.

## Files

- `evaluation_results.csv`: authoritative 210-job raw matrix.
- `audit.json` and `duplicate_attempts.json`: matrix/protocol audit and redundant-attempt record.
- `per_seed_aulc.csv`, `per_seed_aulc_contrasts.csv`, `contrast_summary.csv`: frozen AULC analysis.
- `arm_summary.csv` and `direction_consistency.csv`: arm-level and FID/KID summaries.
- `a_bsame_bmatch_trajectories.csv`: requested per-seed, per-checkpoint comparison.
- `ttq_exploratory.csv`: non-preregistered descriptive TTQ.
- `validation_summary.json`: machine-readable validation status and limitations.

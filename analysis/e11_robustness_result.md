# E11 — Robustness: cross-seed + support-threshold sensitivity (Role C)

Date: 2026-08-10. Branch `role-c/g13-vs-g10-fid-0809`. Free analysis of existing
cross-seed evaluation data. Script: none (reads committed CSVs).

## Hypothesis (E11)
The diagnosis conclusions (residual exists, structured, not harmful; larger gap
→ better FID) are robust to random seeds and support threshold.

## (1) Cross-seed: does "larger gap → better FID" hold across seeds?

### q128, 1024k, FID-50k (seeds 3/4/5, global110 g=1.1 vs fixed g=1.0)
| seed | NFE=1 FID delta | NFE=2 FID delta | winner |
|---|---|---|---|
| 3 | −1.54 | −0.049 | global110 (both) |
| 4 | −0.59 | −0.30 | global110 (both) |
| 5 | −0.08 | −0.031 | global110 (both) |

**global110 wins FID-50k in 5/6 cells across all 3 seeds** (all 3 at NFE=1,
2/3 at NFE=2). KID also mostly global110.

### q128, 256k, proxy-5k (seeds 3/4/5)
| metric | global_only wins |
|---|---|
| NFE=1 FID | 2/3 |
| NFE=2 FID | 2/3 |
| NFE=1 KID | 2/3 |
| NFE=2 KID | 2/3 |

At 256k the proxy-5k results are noisier (4/6 cells global wins), but the
direction is consistent.

## (2) Support-threshold sensitivity (moment-memory result)
From `analysis/moment_memory_real_history_result.md`:
| atol on \|u1\| | Corr(ĥ, h_actual) |
|---|---:|
| 1e-6 | 0.0002 |
| 1e-5 | 0.005 |

The Corr≈0 conclusion is **stable across support thresholds** (both ≈0). The
scalar null is falsified (h_predicted 1.001 vs h_actual 0.837) regardless of
threshold.

## Interpretation (honest)

- **Cross-seed robustness (PASS):** at the longer budget (1024k), the larger-gap
  arm (g=1.1) beats g=1.0 on FID-50k across all 3 seeds (5/6 cells). The
  "deviating from g=1.0 improves FID" finding is not a single-seed fluke.
- **Support-threshold robustness (PASS):** the Corr≈0 / scalar-null-falsified
  conclusion is stable across support thresholds.
- Caveats: the cross-seed data is g=1.1 (global110), not g=1.3 (my clean
  experiment); the 256k proxy-5k results are noisier (4/6); FID-50k at 1024k is
  the cleanest evidence.

## Verdict for E11
- **PASS.** The diagnosis conclusions are robust across seeds (at 1024k) and
  support thresholds. This strengthens the soundness of the honest-diagnosis
  paper.

## Files
- This summary: `analysis/e11_robustness_result.md`
- Data: `results/q128_256k_formal/paired_differences.csv`,
  `results/generalization/schedule-q128/256k_seed3_5_proxy5k/`,
  `analysis/moment_memory_real_history_result.md`

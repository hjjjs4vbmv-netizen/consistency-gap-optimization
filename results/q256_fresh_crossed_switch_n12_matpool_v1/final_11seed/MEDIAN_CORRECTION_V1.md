# Median correction for the amended n=11 analysis

The archived post-seal analysis used a median implementation that averaged the
two order statistics at indices `n//2 - 1` and `n//2` for every sample size.
That is the conventional definition for even samples, including the originally
planned n=12 analysis. After the outcome-blind amendment to n=11, however, the
median should be the single middle order statistic at index `n//2`.

The corrected descriptive medians, recomputed from the preserved seed-level
table `H_C_I_Q_G_per_seed.csv`, are:

| Quantity | Archived value | Corrected n=11 median |
| --- | ---: | ---: |
| H | -0.05653251399225845 | -0.027237725223582254 |
| C | 0.012764043099208422 | 0.016993355924826092 |
| I | 0.012006230782704241 | 0.018658080576266833 |
| Q | -0.00955469520114649 | 0.008284592074332409 |
| H_A | -0.06292188150822842 | -0.04627475067322706 |
| G | -0.04207919909200486 | 0.008818626019299725 |

This correction changes descriptive medians only. The primary H mean, sample
standard deviation, confidence intervals, sign counts, exact sign-flip p value,
leave-one-seed-out means, category checks, and **INCONCLUSIVE** verdict do not
depend on the median and remain unchanged. No training, evaluation, decoding, or
metric computation was rerun.

The formal-node `analysis.json`, `primary_decision.json`, `REPORT_11SEED.md`, and
`SHA256SUMS_11SEED.txt` are retained byte-for-byte as historical finalization
artifacts. They therefore still display the archived median values. For any
manuscript, table, or downstream analysis, use the corrected values above and
the machine-readable `median_correction_v1.json`. That JSON binds the correction
to the preserved protocol, decoded results, per-seed table, formal report, and
formal manifest by SHA-256.

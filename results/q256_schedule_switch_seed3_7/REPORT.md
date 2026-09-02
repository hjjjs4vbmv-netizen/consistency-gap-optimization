# q256 seed3-7 crossed schedule-switch results

Status: **EXECUTION AND ANALYSIS PIPELINE PASS**

This status records artifact completeness and validation; it is not a scientific-hypothesis verdict.

Protocol SHA256: `195ca2843791c0ea28ac5a87f3c9e0fb24a4fb8c9214b665331dbbd92648b32d`

All five training seeds, four trajectories, five budgets, two NFEs, and both KID/FID metrics are reported in the adjacent CSV files.

Means and medians are descriptive summaries over five training seeds. Budget × NFE cells are not independent samples.

## Main finite-horizon finding

At NFE1 and 1024 kimg, a prior B spacing produced lower FID under both current policies in 5/5 seeds. Current A produced lower FID than current B after A history in 5/5 seeds and after B history in 4/5 seeds. The history-by-current interaction was smaller in its signed mean and mixed in sign (3 negative, 2 positive). Over this audited domain, the data therefore support a persistent prior-spacing quality carryover plus a smaller late current-policy preference; they do not support a strong interaction claim.

## Delayed source-to-future reversal

At the 512-kimg switch, B had worse NFE1 FID than A in seeds 5, 6, and 7. At 1024 kimg, B history had lower NFE1 FID under both current policies in all three of those seeds (and in all five seeds overall). Thus, in this q256/CIFAR-10 experiment, at this switch point, future budget, shared continuations, and NFE, current FID did not always rank training states by their future generation quality.

## Sequence ranking and sensitivity

BA had the lowest 1024-kimg NFE1 FID in 4/5 seeds; BB was lowest in seed 6. This pattern is consistent with coarse-then-fine temporal role separation in the audited q256 regime, without establishing a universal curriculum. `contrast_summaries_logfid.csv` repeats the endpoint decomposition on log FID to reduce raw-scale sensitivity. Because this transform was added after results were known, it is explicitly marked post-unblind and descriptive.

## Evidence class and exclusions

This is a pre-result-frozen recovery protocol over an availability-selected five-seed cohort, with compatibility-audited archived controls and pre-parity operational amendments. It is not the unavailable seed14-18 confirmatory experiment. The results do not establish a confirmed interaction null, a universal schedule ranking, mediation by the state blocks measured in the separate same-state audit, or a unique optimizer mechanism.

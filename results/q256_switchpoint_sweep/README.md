# TASK 2: complete q256 switch-point sweep

The 12-seed cohort does not resolve the prespecified ordering after a fixed
512-kimg A chase: exact one-sided Page L=296, p=0.671872582404378. The four paired
geometric FID changes are −4.40%, −4.63%, −6.41%, and −5.39%; all four descriptive
mean log-ratio intervals include zero. Training-seed variation is substantial.

Read the [Chinese report](full_cohort/REPORT_ZH.md),
[132 raw FID/KID observations](full_cohort/raw_metrics.csv),
[48 fixed-chase pairs](full_cohort/fixed_chase_seed_results.csv), and
[G/H summaries](full_cohort/summary.csv).

## Interpretation

Seeds 81–92 are the independent units. All 132 evaluation jobs passed; the primary
matrix contains 96 jobs and the common-endpoint descriptive matrix adds 36.
FID and KID use FP32, NFE1, and the same 50,000 generated samples per job.
The [protocol](../../analysis/q256_switchpoint_sweep/PROTOCOL.md) and
[analysis plan](../../analysis/q256_switchpoint_sweep/analysis_plan.json) define
the fixed-chase G and common-1024-endpoint H comparisons.

This completes the same cohort whose six-seed recovery results were previously
reported. It is not an independent replication or an uninterrupted blinded
confirmation. `ORDERING_NOT_RESOLVED` and `primary_status=ANALYZED` in
`frozen_calculation.json` are outputs of the original statistical functions;
`reporting_context` records the prior disclosure. No effect, equivalence,
early writing, or dose accumulation is established. Other cohorts and TASK1's
frozen verdict are unchanged.

The generation-noise companion is incomplete: only the reused formal block 0
is available, and its four additional paired blocks have not been evaluated.
There is no generation-noise SD estimate or noise-based tie classification.

## Reproduce with Python

Run from the repository root; these commands need only the Python standard library:

```bash
python3 -m analysis.q256_switchpoint_sweep.summarize_results --output-dir /tmp/task2-summary
python3 -m unittest tests.test_q256_switchpoint_sweep_analysis
python3 -m unittest tests.test_q256_switchpoint_sweep_results
```

The first command regenerates the Chinese report, raw and paired CSVs, G/H
summaries, Page calculation, and comparison with the old six-seed observations.
It reads the published `decoded_results.json`; it does not read an external
server, rerun evaluation, or rewrite the historic archive verification record.

The original calculation remains directly callable:

```bash
python3 -m analysis.q256_switchpoint_sweep.analyze \
  --decoded-results results/q256_switchpoint_sweep/full_cohort/decoded_results.json \
  --output-dir /tmp/task2-frozen-calculation
```

Its statistical outputs must be read together with the prior-disclosure context
above; the original function does not evaluate blinding history.

## Provenance and retained archive

- `matrix_seal.json` is the original 2026-09-05 seal, byte-for-byte unchanged.
  Its `decoded=false` is the historical state at sealing, not the current
  disclosure state of this published bundle.
- `verification.json` records the completed local checks of 264 metric-file
  hashes, 132 options-file hashes, and 132 terminal/seal bindings against the
  archived receipts. Private login addresses have been omitted from this public
  copy. Checkpoints and generated-feature arrays were not rehashed in that pass.
- This small public bundle supports numerical reproduction. The full per-job
  options, receipts, features, checkpoints, and original manifests remain in the
  private archive; their byte-level verification cannot be repeated from this
  public bundle alone. Holders of the archive can run the summarizer with
  `--archive-root /path/to/evaluation --output-dir /tmp/task2-archive-check`.
- `preliminary/` retains the previous 48 metric rows and report. Its original
  verification record is retained privately. `preliminary_comparison.json`
  compares those same cells with the completed matrix, not with a new cohort.
- The training/evaluation implementation was committed as `d5a6883` locally and
  `9154a9a` on the execution host; both have tree
  `d5a6f6a8319627841b3a509ab38b189a29d07906`. The evaluator commit is
  `d6aba02fb88e9db0993623895eb2228ed717d810`.

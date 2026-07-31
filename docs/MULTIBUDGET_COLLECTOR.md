# Multi-budget result collector

Use scripts/collect_multibudget_results.py when the 512/768/1024 evaluation
CSV is available. The collector is not hard-coded to those three budgets: it
uses every budget, seed, NFE, and metric observed in a complete two-method
matrix.

## Input contract

The CSV is long-form, with one metric per row.

| Column | Required | Meaning |
| --- | --- | --- |
| method | yes | One of the baseline or candidate method names. |
| training_seed | yes | Independent training seed. |
| budget_kimg | yes | Completed training budget in kimg. |
| nfe | yes | Sampling NFE setting. |
| metric_name | yes | Metric identifier, such as kid5k_full, fid5k_full, kid50k_full, or fid50k_full. |
| metric_value | yes | Metric value; the collector assumes lower is better. |
| training_time_hours | no | Cumulative wall-clock training time at this budget. Required for an actual time-to-quality time. |
| quality_target | no | Pre-specified threshold for this metric/NFE; repeated consistently across its rows. |
| checkpoint_sha256 | no | Checkpoint provenance retained in normalized output. |

Endpoint sets may differ by budget. Every
method/seed/budget/NFE/metric combination implied by the endpoints observed at
that budget must appear exactly once; the collector does not require a
metric-by-budget Cartesian product. This supports the frozen q=256 protocol:
KID/FID-5k at 512/768 kimg and KID/FID-50k at 1024 kimg. Use the same
quality_target for all rows of a given metric and NFE. The script
does not invent a target or convert budget into elapsed wall-clock time.

## Run

    python scripts/collect_multibudget_results.py \
      --input-csv results/multibudget_512_768_1024/evaluation_results.csv \
      --outdir results/multibudget_512_768_1024/collected \
      --baseline-method fixed \
      --candidate-method global110

## Outputs

- budget_curves.csv and figures/budget_curves in SVG, PNG, and PDF
- time_to_quality.csv and, when at least one trajectory reaches a target with
  wall-clock time, figures/time_to_quality in SVG, PNG, and PDF
- per_seed_trajectories.csv and figures/per_seed_trajectories in SVG, PNG, and PDF
- paired_deltas.csv, paired_summary.csv, and figures/paired_deltas in SVG, PNG, and PDF
- summary_table.md and summary_table.tex
- figure_ready_budget_curves.csv, figure_ready_per_seed_trajectories.csv,
  figure_ready_paired_deltas.csv, and figure_ready_time_to_quality.csv

Paired deltas are candidate minus baseline, so a negative value favors the
candidate. Sample SD is descriptive across training seeds, not a confidence
interval or a significance result.

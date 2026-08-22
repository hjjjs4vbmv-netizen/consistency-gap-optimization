# Role D analysis contract

This contract turns the q256 long-budget records into paper assets without
mixing evaluation protocols. It is binding for the current compute-to-quality
mainline.

## Admissible source

The only current endpoint input is a complete, receipt-backed FID-50k matrix
for seeds 3–5, arms A–D, NFE 1/2, and budgets 256/1024 kimg. The normalizer
requires 50,000 generated samples, generation seeds `0-49999`, metric seed
`20260730`, and a `PASS` receipt for every cell. Its frozen definition is
[`q256_two_budget_fid50k_source.frozen.json`](../configs/paper_assets/q256_two_budget_fid50k_source.frozen.json).

The checked-in 256-kimg source is
[`formal_evaluation_seed3_5_results_dcca41b.csv`](../analysis/q256_target_weight_factorial/formal_evaluation_seed3_5_results_dcca41b.csv).
The 1024-kimg raw receipt CSV is not present in this workspace and must be
supplied before a final source CSV or a paper figure is generated.

## Absolute boundary

FID-5k and FID-50k are different protocols. In particular, the historical
FID-5k values at 512 and 768 kimg must never be joined to FID-50k values at
256 or 1024 kimg. The normalizer only reads the `fid50k_full` column; Asset A
is separately locked to a complete 256/512/768/1024 single-protocol series.

## Frozen assets

| Deliverable | Frozen definition | Role |
| --- | --- | --- |
| Two-budget endpoint comparison | `q256_two_budget_endpoint_nfe1.frozen.json` | Visible A/B/C/D values at 256 and 1024 kimg; not a learning curve. |
| Compute-to-quality | `q256_compute_to_quality_nfe1_eta12.frozen.json` | A/B threshold analysis using a single descriptive linear-interpolation definition. |
| Dispersion contraction | `q256_arm_dispersion_nfe1.frozen.json` | Per-seed four-arm FID range at the two matched budgets. |

The threshold `FID-50k <= 12` and crossing rule were frozen on 2026-08-22.
They must not change when the 1024 raw receipt CSV is added.

## Regeneration commands

Replace `/path/to/q256_1024k_fid50k_receipts.csv` only with the delivered raw
receipt CSV. The commands fail closed if a cell, protocol field, or status is
missing.

```bash
python scripts/normalize_q256_two_budget_fid50k.py \
  --config configs/paper_assets/q256_two_budget_fid50k_source.frozen.json \
  --budget-input 256=analysis/q256_target_weight_factorial/formal_evaluation_seed3_5_results_dcca41b.csv \
  --budget-input 1024=/path/to/q256_1024k_fid50k_receipts.csv \
  --out-csv results/q256_compute_to_quality/evaluation_results.csv \
  --manifest results/q256_compute_to_quality/source_manifest.json

python scripts/render_paper_asset_endpoint.py \
  --input-csv results/q256_compute_to_quality/evaluation_results.csv \
  --endpoint-config configs/paper_assets/q256_two_budget_endpoint_nfe1.frozen.json \
  --outdir paper_assets/q256_two_budget_endpoint_nfe1

python scripts/render_paper_asset_b.py \
  --input-csv results/q256_compute_to_quality/evaluation_results.csv \
  --threshold-config configs/paper_assets/q256_compute_to_quality_nfe1_eta12.frozen.json \
  --outdir paper_assets/q256_compute_to_quality_nfe1_eta12

python scripts/render_paper_asset_c.py \
  --input-csv results/q256_compute_to_quality/evaluation_results.csv \
  --arm-config configs/paper_assets/q256_arm_dispersion_nfe1.frozen.json \
  --outdir paper_assets/q256_arm_dispersion_nfe1
```

Each renderer emits PDF and SVG masters, a 600-dpi PNG preview, normalized
source CSV, hashes, an exact rendering command, a caption, an interpretation
boundary, and a grayscale preview for final visual review.

## Evidence labels and reporting

Seeds 3–5 are the preregistered formal matrix. Seeds 6–7, 8–13, and 14–18
are descriptive sensitivity evidence and must remain separate from a
prospective confirmation label. All figures retain each evaluated seed; a
mean or median is a labelled summary rather than a substitute for the
individual observations.

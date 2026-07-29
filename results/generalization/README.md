# Generalization Results

This directory stores lightweight, reviewable evidence for a legacy retrospective exploratory q=128 screening. Checkpoints, generated single images, caches, and complete logs remain outside Git.


## q=128 evidence status

The q=128 setting was not formally frozen before its results were observed.
Results are legacy retrospective exploratory evidence produced from a
pre-merge implementation. Canonical dataset-content equivalence was not
established, so this directory does not claim that q was the only changed
experimental input.

Each metric cell uses one generated 5k sample set. Three recomputations check
numerical reproducibility on that same sample set; they are not independent
statistical repetitions. The statistical unit is the training seed (`n=3`).

## Planned layout

```text
results/generalization/
├── README.md
└── schedule-q128/
    ├── protocol_snapshot.json
    ├── asset_manifest.json
    ├── training/
    │   ├── fixed_seed3.json
    │   ├── fixed_seed4.json
    │   ├── fixed_seed5.json
    │   ├── global_only_seed3.json
    │   ├── global_only_seed4.json
    │   └── global_only_seed5.json
    ├── evaluation/
    │   ├── per_seed_metrics.csv
    │   ├── paired_differences.csv
    │   └── aggregate_results.csv
    ├── figures/
    │   ├── paired_metrics.png
    │   └── qualitative_grid.png
    └── CONCLUSION.md
```

## Required result fields

Each metric row must include:

- setting,
- method,
- global scale,
- training seed,
- budget kimg,
- checkpoint SHA256,
- NFE,
- `mid_t`,
- sampling seed specification,
- sample count,
- KID and/or FID,
- metric code commit,
- reference-stat SHA256,
- precision and device,
- run status.

## Reporting rules

1. Report seeds 3, 4, and 5 separately before aggregation.
2. Define paired delta as `global_only - fixed`; negative KID/FID delta favors global-only.
3. Label all 5k-sample results as screening or proxy evidence.
4. Do not interpret q=128 results as confirmatory generalization evidence.
5. Do not reselect `g=1.10` or retrospectively promote this setting after observing results.
6. Keep failures and missing cells visible.
7. Store only lightweight evidence in Git.

# Generalization Results

This directory stores lightweight, reviewable evidence for the predeclared secondary setting. Checkpoints, generated single images, caches, and complete logs remain outside Git.

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
4. Reserve formal claims for the predeclared 50k evaluation.
5. Do not reselect `g=1.10` or the second setting after observing confirmatory results.
6. Keep failures and missing cells visible.
7. Store only lightweight evidence in Git.

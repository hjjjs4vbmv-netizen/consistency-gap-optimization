# q=128 1024 kimg reported results — retrospective/supplementary

This package replaces the misleading `q128_256k_formal` location.  Its 1024
kimg name matches every reported checkpoint and metric row, but **it is not a
formal evaluation package**: no 1024-kimg prospective frozen matrix or
independent Role D receipt is available in Git.

| Artifact | Contents | Verification status |
| --- | --- | --- |
| `source_run_manifest.json` | Portable account of the historical import and frozen-matrix assessment | Source manifest unavailable |
| `checkpoint_manifest.csv` | Six reported checkpoint SHAs and every requested provenance field | Required provenance fields explicitly missing |
| `receiver_verification.json` | Role D receiver-verification record | Not received |
| `evaluation_results.csv` | 24 compact reported raw metric records | Internally cross-checked; metric origin unverified |
| `paired_differences.csv` | 12 descriptive within-q128 deltas | Derived from reported raw rows only |
| `environment_manifest.json` | Reported evaluation contract and explicit environment gaps | Partially recorded |
| `reproduction_commands.sh` | Exact package validation command | Reproduces validation, not the historical metrics |
| `validation_report.md` | Scope and pass/fail evidence ledger | Current |

`evaluation_results.csv` deliberately has no absolute path.  `run_id` is a
logical identifier, and `record_status=reported_unverified` is part of every
row.  The result values can be reviewed as provisional within-q128 paired
observations only.  They must not be used for cross-q claims or to explain the
q256-256k result while dataset-semantic equivalence and Role D verification
remain blocked.

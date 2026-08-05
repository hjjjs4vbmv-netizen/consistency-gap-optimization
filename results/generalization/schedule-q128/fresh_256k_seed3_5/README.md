# Fresh q=128 confirmatory training handoff

This directory records six fresh q=128 training runs comparing fixed
sigmoid with global-only gap scaling at 1.10 for seeds 3, 4, and 5.

All six runs reached 256 kimg and produced a final network snapshot,
training state, training options, and training summary. Generation
quality metrics were not run in this training stage.

Files:

- `readiness.csv`: six-cell training acceptance table.
- `checkpoint_manifest.csv`: snapshot, state, and config hashes.
- `metadata.json`: shared experiment identity and provenance.
- `q128_config_diff.md`: controlled configuration comparison.
- `q128_seed3_smoke_report.json`: engineering smoke evidence.
- `q128_fresh_vs_legacy_policy.md`: isolation policy.
- `run_q128_confirmatory.sh`: exact sequential runner.
- `train_summaries/`: compact per-cell training records.

Model snapshots and optimizer states are stored externally and are not
committed to Git.

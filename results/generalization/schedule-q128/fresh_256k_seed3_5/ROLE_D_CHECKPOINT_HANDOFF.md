# Role D checkpoint handoff

The archive contains the six predeclared fresh q=128 checkpoints for
seeds 3, 4, and 5 under fixed sigmoid and global-only g=1.10.

Role D must:

1. Recompute the archive SHA256.
2. Extract the archive.
3. Recompute all six checkpoint SHA256 values.
4. Compare them with `checkpoint_mapping.csv`.
5. Record the evaluator-visible checkpoint paths.
6. Return a machine-readable receiver verification JSON.

The sender manifest is `checkpoint_handoff_sender.json`.

Formal evaluation remains blocked until:

- Role D receiver verification passes; and
- q=128/q=256 dataset semantic equivalence is established.

# Fresh versus legacy q=128 policy

The legacy q=128 experiments remain retrospective exploratory
screening evidence.

The new experiment uses the identifier:

`q128-fresh-fixed-vs-global110-v1`

The fresh and legacy runs must not be combined in one confirmatory
statistics table. Legacy results must not be used to select seeds,
global-gap scale, checkpoints, or evaluation settings for the fresh
matrix.

Fresh runs use:

- seeds 3, 4, and 5;
- fixed sigmoid and global-only 1.10;
- fresh initialization from the common transfer checkpoint;
- 256 kimg training budget;
- isolated run and log directories.

The `.pkl` and `.pt` assets remain outside Git. Git contains only
metadata, hashes, compact training summaries, protocol documents, and
the reproducible runner.

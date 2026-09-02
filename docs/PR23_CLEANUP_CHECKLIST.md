# PR #23 Cleanup Checklist

## Review scope

- [x] Read the requested-changes review.
- [x] Confirmed that there are no unresolved inline review threads.
- [x] Treat the single top-level `CHANGES_REQUESTED` review as five actionable
  clusters: main synchronization, diff reduction, reproducible validation,
  scientific wording, and selection/proxy caveats.

## Branch synchronization

- [x] Fetched `origin/main` at `274786fc040cfd1a1802b146566f43c603dee33d`.
- [x] Preserved the former PR head locally as
  `codex/gap-factorial-controller-precleanup-20260728`.
- [x] Rebuilt `codex/gap-factorial-controller` from the latest `origin/main`.
- [x] Migrated the factorized-controller commit without conflicts.
- [x] Reduced the local PR diff from 198 files to the intended gap-factorial
  scope before adding this cleanup documentation.
- [x] Did not modify or delete files already present on `main`.

## Files retained

### Implementation

- `ct_train.py`
- `training/schedules.py`
- `training/loss.py`
- `training/ct_training_loop.py`

### Factorial experiment pipeline

- `scripts/run_gap_stage1.sh`
- `scripts/run_gap_stage2.sh`
- `scripts/run_gap_after_stage1.sh`
- `scripts/run_gap_factorial_arm.sh`
- `scripts/evaluate_gap_factorial_arm.sh`
- `scripts/select_gap_scale.py`
- `scripts/summarize_gap_factorial.py`
- `scripts/verify_gap_factorial_arm.py`
- `scripts/extract_gap_controller_state.py`

### Tests

- `tests/test_schedules.py`
- `tests/test_training_cli_compat.py`
- `tests/test_local_tbin_controller.py`
- `tests/test_select_gap_scale.py`
- `tests/test_summarize_gap_factorial.py`
- `tests/test_adaptive_signal_updates.py` for telemetry-schema migration and
  aggregation coverage.

### Compact results and documentation

- `results/gap_factorial_20260727/`
- `docs/METHOD_V0.md`
- `docs/PR23_CLEANUP_CHECKLIST.md`

## Historical material removed from the PR diff

- [x] Old `results/final_performance_evaluation/` Role A outputs.
- [x] Role A/D evaluation scripts and tests already represented on `main`.
- [x] Blind-evaluation ZIP files.
- [x] Historical raw metric JSONL trees.
- [x] Evaluation PNGs and controller plots unrelated to this factorial study.
- [x] Old final-evaluation configuration and documentation changes.
- [x] Local untracked `results/local_tbin_*` raw directories remain untouched
  and are not part of the PR.

## Scientific wording

- [x] Replaced global-neutrality language with:
  “the local scale factors have geometric mean one before realized-gap
  clipping effects.”
- [x] Defined the pre-clipping factorization
  `d_pre(t) = g ℓ_b(t) d_base(t)`.
- [x] Defined the implemented realized-gap clipping sequence.
- [x] Added `realized gap / sigmoid_gap`, lower-gap clipping rate, and
  upper-gap clipping rate telemetry.
- [x] Documented that the completed runs predate these telemetry fields, so
  their exact historical values are unavailable.
- [x] Defined headline percentages as ratios of held-out arithmetic means, not
  means of per-seed percentages.
- [x] Quantified the strong influence of seed 1 on the NFE=2 held-out mean.
- [x] Kept seed 0 labeled as the `g*=1.10` selection seed.
- [x] Kept KID/FID-5k and `n=3` conclusions descriptive.

## Final-head validation

- [x] Schedule and controller tests.
- [x] Controller/resume-state tests.
- [x] Selection and summarization tests.
- [x] Python compilation.
- [x] Shell syntax.
- [x] `git diff --check`.
- [x] Confirm final PR changed-file list contains only retained scope (29 files
  relative to synchronized `origin/main`).

The final local suite passed 145 tests with 4 expected skips: two CUDA-only
loss-call tests and two clean-worktree-only runner identity tests. The
gap-specific subset passed 76 tests with 2 expected CUDA skips.

# Merge-readiness failure comparison

The two originally reported failures were rerun individually at the published
branch head and the current target-base head in the same formal container. Raw
commands, test IDs, stdout, stderr, exit codes, JUnit XML, commit IDs, and
environment records are preserved below `branch_original/` and `target_base/`.

## Classification

- `test_amp_path_unscales_and_preserves_source_scaler` fails on both revisions
  because PyTorch 2.2 has no enabled CPU GradScaler, but the exception signatures
  are not identical: base raises `AttributeError`, while the branch's explicit
  compatibility guard raises `RuntimeError`. The underlying environment gap is
  pre-existing, but this is not reported as a base-identical signature. The test
  now uses a deterministic test-only CPU scaler; production CUDA behavior is
  unchanged.
- `test_scaler_falls_back_to_cuda_amp_on_torch22` does not exist at the target
  base and fails only on the original branch because its mock tried to patch a
  missing PyTorch-2.2 attribute without `create=True`. This is a branch-induced
  test-fixture regression and is fixed at the fixture boundary.

Neither test was hidden, deselected, renamed, or skipped. Final post-merge test
results are recorded separately in the canonical aggregate provenance manifest.

# PR #91 integration gate

Status: **WAITING_FOR_PR_91_MERGE**

PR #91 remains the canonical owner of generic exact-resume behavior. PR #94
must not merge first and must not resolve the overlap by replacing PR #91's
`exact_resume` / `exact_reproducibility` path with the q256-specific fork path.

The historical P0 execution is preserved by the immutable tag
`q256-b384-same-state-p0-execution-v1`. Its protocol remains bound to
implementation commit `f755fa62323b373dc2178a5de8c2c56dd39d99ab` and protocol
commit `2de0628117caffa8142602b3d9ccf1ffb1397e1d`. A later integration rebase does
not change those execution identities and does not, by itself, require a P0
rerun.

## Pre-merge audit against PR #91 head

The temporary overlap audit used PR #91 head
`3f0572cd1c1e1e3132b6e8a4016cbdad82240a0c`. Text conflicts are limited to:

- `ct_train.py`
- `training/ct_training_loop.py`

`training/loss.py` auto-merges. The audit does not substitute for integration
against the actual merge commit on `main`.

## Required post-merge resolution

1. Fetch the new `origin/main` after PR #91 merges.
2. Rebase/rebuild PR #94 from its original base `8d4844c` onto that main.
3. Preserve PR #91's generic exact-resume logic as the base implementation.
4. Layer the explicitly scoped q256 B@384 fork on top:
   - generic exact-resume must not require or index `factorial` state;
   - immutable kimg milestones remain available to generic exact-resume;
   - attempted-iteration milestones remain restricted to strict factorial runs;
   - strict fork checkpoints retain both `factorial` and `same_state_fork`;
   - exact net/EMA restoration remains governed by `exact_reproducibility`.
5. Keep the historical protocol's implementation SHA unchanged.

## Combined regression gate

At minimum, run:

- `tests/test_exact_resume_state.py`
- `tests/test_immutable_checkpoint_milestones.py`
- `tests/test_training_cli_compat.py`
- `tests/test_q256_target_weight_factorial.py`
- `tests/test_q256_same_state_p0.py`
- `tests/test_schedules.py`
- `tests/test_recompute_detach_semantics.py`
- Python compilation
- `git diff --check`

The merge gate remains open until PR #91 is merged and these checks pass on the
rebased PR #94 branch.

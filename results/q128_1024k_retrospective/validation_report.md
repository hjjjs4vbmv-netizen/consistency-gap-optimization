# q=128 1024 kimg retrospective-package validation

## Result

**Package structure: pass. Evidence provenance: incomplete. Formal eligibility: fail.**

The package contains 24 reported metric rows: six reported checkpoint IDs ×
two NFE modes × two metrics.  Every row is 1024 kimg, carries a checkpoint
SHA-256 also present in `checkpoint_manifest.csv`, uses the logical `run_id`
field, and contains no absolute host path.  The 12 job-status rows have the
same six identities and two NFE modes; the 12 paired rows rejoin to the raw
records and reproduce their reported deltas within `1e-12`.

## Non-claims and unresolved blockers

The structural checks do not establish that the reported values were generated
from those checkpoints.  The following primary evidence is absent for every
checkpoint: source 256-kimg ID/SHA, training commit, resume start, optimizer /
EMA / GradScaler continuity, training-options SHA, training-state SHA,
integrity receipt SHA, archive SHA, and Role D receiver verification.  The
reported dataset SHA and evaluation commit are import metadata, not an
independent reconstruction.

The repository has no prospective 1024-kimg frozen matrix.  Therefore the
results are retrospective/supplementary, cannot be called formal, and cannot
support cross-q generalization.  The q128/q256 dataset semantic-equivalence
and Role D receiver-verification blockers remain open.

## Commands

Run `bash results/q128_1024k_retrospective/reproduction_commands.sh` to re-run the
portable structural validation.  The exact original evaluator and collector
commands are `NOT_RECORDED`; the original source manifest is unavailable.

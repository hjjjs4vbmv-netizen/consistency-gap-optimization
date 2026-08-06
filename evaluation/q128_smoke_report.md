# q=128 1024 kimg retrospective evaluation record

## Status

This is **retrospective/supplementary evidence**, not a formal or prospective
evaluation.  The repository's only frozen q=128 matrix is the 256-kimg matrix
in `configs/q128_confirmatory_matrix.frozen.json`; its six cell IDs are
`q128-fresh-256k-*` and its checkpoint hashes differ from the 1024-kimg IDs
reported here.  The 1024-kimg result rows entered Git in commit
`f53910ba7ed1890b05d1da6f8a9b616d03e8e576` (2026-08-03), after that matrix
was frozen (2026-07-31).  No pre-result 1024-kimg frozen matrix is present in
the repository.

Accordingly, no statement that “all predeclared cells entered formal
evaluation” applies to this 1024-kimg set.  `integrity_receipt_status=passed`
from the historical import has been withdrawn: the underlying receipts and
the original source manifest were not delivered to this repository.

## What is preserved

`results/q128_1024k_retrospective/` is a portable, path-free audit package.  It
preserves the 24 reported raw metric values and the six reported checkpoint
SHA-256 values, but labels them `reported_unverified`.  The directory name is
only a storage path retained for continuity; the package metadata and every
record classify its evidence as retrospective/supplementary rather than
formal.

The local validation report confirms the internal structure of the imported
records (six reported checkpoint identities, two NFE modes, two metrics, and
24 rows).  It cannot validate checkpoint identity, training provenance, or
metric execution without the missing primary artifacts.

## Blocked claims

Until the primary provenance and a Role D receiver verification are supplied,
the reported fixed/global paired values may be described only as provisional
within-q128 retrospective comparisons.  They do not support cross-q
generalization and must not be used to explain q256-256k FID.  The separate
q128/q256 dataset semantic-equivalence blocker is also still open.

See `results/q128_1024k_retrospective/README.md` for the audit inventory and the
unresolved evidence ledger.

# Frozen prospective evaluation matrices

The matrices in this document are frozen before the new checkpoints are trained or evaluated. They are logical plans, not machine-local runner manifests: each completed endpoint must later be bound to its checkpoint SHA-256 and a passed training-integrity receipt without changing any listed cell identity, metric contract, or comparison rule.

| Matrix | Checkpoints | NFE | Evaluation contract |
| --- | ---: | --- | --- |
| q=256 budget | 18: seeds 3/4/5 × fixed/global110 × 512/768/1024 kimg | 1 and 2 | 512/768: KID/FID-5k; 1024: KID/FID-50k |
| fresh q=128 | 6: seeds 3/4/5 × fixed/global110 × 256 kimg | 1 and 2 | KID/FID-50k |

The machine-readable specifications are:

- `configs/q256_budget_matrix.frozen.json`
- `configs/q128_confirmatory_matrix.frozen.json`

Both matrices freeze `NFE=1` as `mid_t=[]`, `NFE=2` as `mid_t=[0.821]`, the evaluator/metric seed as `20260730`, the pairing key as `training_seed + budget_kimg + nfe + metric`, and the paired delta as `global_only - fixed`. A negative delta favors global-only.

## Completion and promotion rules

Every listed checkpoint is mandatory: it must complete the evaluation contract assigned to its budget. A missing metric, missing arm, or unmatched fixed/global pair is an incomplete matrix, not a reason to report a subset.

For the q=256 budget matrix, the 512- and 768-kimg results are explicitly 5k screening evidence. They cannot decide whether any of the six predeclared 1024-kimg checkpoints enters formal 50k evaluation; those six enter when and only when their immutable provenance matches the frozen cell and their training-integrity receipt passes. They must not be removed, added, substituted, or selected based on a quick value.

The fresh q=128 matrix has six predeclared formal 50k checkpoints. An implementation smoke, if run, is diagnostic only and cannot alter formal eligibility. Its historical q=128 predecessor remains retrospective exploratory evidence and is not promoted by this prospective matrix.

The existing `run_staged_evaluation.py` consumes a machine-local binding manifest with paths and hashes, not either of these path-free prospective plans. Bind the completed cells only after training, validate the binding and receipt identities, and retain the frozen plan unchanged.

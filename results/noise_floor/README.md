# TASK 7: generation-block sensitivity

This directory publishes two post-seal descriptive evaluations using five
disjoint 50k generation blocks (`B1`--`B5`) at NFE1 and NFE2. Generation blocks
are repeated measurements of fixed checkpoints, not new training samples. The
historical `B0` values are anchors and are excluded from every reported mean
and sample SD.

| Campaign | Evaluated checkpoints | Jobs | Result |
|---|---:|---:|---|
| Prespecified checkpoint matrix | 13 | 130 | 130 PASS |
| n=30 companion AA/BA pairs | 6 | 60 | 60 PASS |

The first campaign leaves three inaccessible q256 BA@1024 checkpoints
unevaluated; the companion campaign does not substitute for them. It instead
measures BA-minus-AA sensitivity on the first three complete companion pairs in
ascending seed order: seeds 50, 51 and 52.

## Companion BA-minus-AA results

FID rows use `log(FID_BA) - log(FID_AA)`. Negative values favor BA because
lower FID is better. `2SD` is twice the sample SD of the five paired block
differences.

| Seed | NFE | Mean log-FID difference | BA FID difference | Paired 2SD | Block signs |
|---:|---:|---:|---:|---:|---|
| 50 | 1 | -0.06314 | -6.12% | 0.00661 | 5/5 negative |
| 50 | 2 | -0.03189 | -3.14% | 0.01296 | 5/5 negative |
| 51 | 1 | -0.03628 | -3.56% | 0.00901 | 5/5 negative |
| 51 | 2 | -0.01787 | -1.77% | 0.01322 | 5/5 negative |
| 52 | 1 | -0.12147 | -11.44% | 0.00965 | 5/5 negative |
| 52 | 2 | +0.00687 | +0.69% | 0.01968 | mixed |

Seed 52 at NFE2 is `not interpreted`: both FID and KID change direction across
blocks, and each mean magnitude is below its paired `2SD` scale. This is a
cell-specific descriptive label, not an equivalence decision.

## Prespecified checkpoint matrix

The 130-job matrix completed every accessible checkpoint on the first attempt.
Most evaluated contrasts retained their B0 sign across B1--B5, with two
material qualifications:

- q128 Cmatch-minus-Bmatch NFE2 log FID changed from a positive B0 anchor to
  negative in all five new blocks;
- target-weight B-minus-D@1024 NFE2 KID changed sign across blocks.

The target-weight B-minus-D 256-to-1024 rotation was observed in 5/5 new blocks
for NFE1 KID, 2/5 for NFE2 KID, and 0/5 for FID at either NFE. These are
checkpoint-, metric- and NFE-specific observations.

## Interpretation boundary

`2SD` is a descriptive variation scale. It is not a confidence interval,
equivalence margin or transferable `TIE` rule. No result here modifies frozen
inference. Winner tables should report cell-level generation SD and use `not
interpreted` where the measured ordering is generation-block-sensitive.

Machine-readable outputs are in `checkpoint_matrix/` and `n30_companion/`.

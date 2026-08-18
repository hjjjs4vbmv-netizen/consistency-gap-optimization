# q256 g=1.10 gradient × RAdam-state factorial audit

**Correctness verdict: PASS.** The independent replication unit is the training seed (n=3); the eight audit minibatches within each seed are paired repeated measurements, not independent training replicates.

## Seed-level four-cell medians

| Training seed | A observed/real | B observed/reset | C exact/real | D exact/reset |
|---:|---:|---:|---:|---:|
| 3 | 0.0830280345 | 0.477003133 | 0.0703500848 | 0.00170813566 |
| 4 | 0.087762005 | 0.490735527 | 0.0771869548 | 0.00163518257 |
| 5 | 0.0740520628 | 0.466909624 | 0.0488545137 | 0.000602337327 |

Cells report median `R_opt`; absolute residuals and all paired contrasts are in `seed_summary.csv` and `batch_contrasts.csv`.

## Correct interpretation order

1. D measures the exact-scalar/reset baseline, including RAdam epsilon effects.
2. C shows exact-scalar gradients under the real accumulated state.
3. B−D isolates the descriptive increment associated with the observed gradient residual under reset state.
4. A−C shows the observed-gradient increment under real state.
5. A−B compares real against reset state for the observed gradient pair.

These paired contrasts are diagnostic and are not an additive causal decomposition.

## Mechanism readout

The observed-gradient residual is the larger isolated probe

## Gates and test suite

D identity: PASS. Control-control identity, source preservation, branch-order invariance, same-batch rerun hashes, finite-number checks, and all 96 receipt contracts are included in the overall verdict.
Full test suite: 262 passed, 3 skipped, 2 failed, 0 errors (267 total).

## Conclusion boundary

Allowed: describe whether the formal g=1.10 optimizer-update divergence is associated mainly with the observed non-scalar gradient residual, accumulated RAdam state, or their interaction in these frozen virtual updates.

Not allowed: claim that optimizer memory caused an FID improvement, treat audit minibatches as independent training replicates, call this a full-training intervention, or infer a continuation-training effect. No training, samples, FID, or KID were produced.

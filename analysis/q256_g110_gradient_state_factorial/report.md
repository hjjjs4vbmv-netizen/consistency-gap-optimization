# q256 g=1.10 gradient × RAdam-state factorial audit

**Correctness verdict: PASS.** The independent replication unit is the training seed (n=3); the eight audit minibatches within each seed are paired repeated measurements, not independent training replicates.

## Seed-level four-cell medians

| Training seed | A observed/real | B observed/reset | C exact-scalar/real | D exact-scalar/reset |
|---:|---:|---:|---:|---:|
| 3 | 0.0830280345 | 0.4770031330 | 0.0703500848 | 0.00170813566 |
| 4 | 0.0877620050 | 0.4907355270 | 0.0771869548 | 0.00163518257 |
| 5 | 0.0740520628 | 0.4669096240 | 0.0488545137 | 0.000602337327 |

Cells report median `R_opt`. Machine-readable full-precision values are in `seed_summary.csv` and `summary.json`; the table above uses the task-specified display precision.

## Contrasts of seed-level cell medians

| Contrast | Seed 3 | Seed 4 | Seed 5 | Cross-seed median |
|---|---:|---:|---:|---:|
| (B-D) | 0.4752949973 | 0.4891003444 | 0.4663072867 | 0.4752949973 |
| (C-D) | 0.0686419491 | 0.0755517722 | 0.0482521764 | 0.0686419491 |
| (A-C) | 0.0126779497 | 0.0105750502 | 0.0251975491 | 0.0126779497 |
| (A-B) | -0.3939750985 | -0.4029735220 | -0.3928575612 | -0.3939750985 |
| Interaction (A-B-C+D) | -0.4626170476 | -0.4785252942 | -0.4411097376 | -0.4626170476 |

The derived descriptive interaction is defined as

`I = (A-C) - (B-D) = A-B-C+D`.

These values contrast the four within-seed cell medians. `batch_contrasts.csv` retains all 24 paired batch measurements, and the backward-compatible `mechanism_summary` retains the separately named median-of-within-batch-contrast estimand. Neither estimand is an additive causal decomposition.

## Correct interpretation order

1. D measures the exact-scalar/reset baseline, including RAdam epsilon effects.
2. C shows exact-scalar gradients under the real accumulated state.
3. B−D isolates the descriptive increment associated with the observed gradient residual under reset state.
4. A−C shows the observed-gradient increment under real state.
5. A−B compares real against reset state for the observed gradient pair.

These paired contrasts are diagnostic and are not an additive causal decomposition.

## Mechanism readout

Across all three training seeds, the observed-gradient/reset contrast (B-D) was large, with a cross-seed median of 0.4753, whereas exact-scalar gradients under the real accumulated state retained a smaller but nonzero divergence, with (C-D=0.0686). Crucially, the combined observed/real cell was far below the observed/reset cell, yielding a large negative gradient-state interaction, (A-B-C+D=-0.4626). Thus accumulated RAdam state both breaks exact scale equivariance and strongly attenuates the update divergence exposed by the observed non-scalar gradient residual. Moment zeroing is therefore not a memory-neutral intervention.

1. The observed non-scalar gradient residual is the larger isolated probe under reset state because (B-D) is large.
2. Real accumulated RAdam state itself breaks exact scale equivariance because (C-D)>0 consistently across seeds.
3. The large negative interaction shows that real optimizer history strongly attenuates the divergence exposed by the observed residual; the four cells cannot be interpreted as additive causal contributions.

## Gates and test suite

D identity: PASS. Control-control identity, source preservation, branch-order invariance, same-batch rerun hashes, finite-number checks, and all 96 receipt contracts are included in the overall verdict.
Full test suite: 281 passed, 1 skipped, 0 failed, 0 errors (282 total).

## Environment and schema compatibility

Formal receipts record PyTorch 2.2.0a0+81ea7a4 and CUDA 12.3. This run must not be labeled as a PyTorch 2.3/CUDA 12.4 environment replication.

Summary schema version 1 remains readable: revision 2 is additive, preserves all revision-1 fields, adds explicitly named seed-cell-median contrasts, and exposes the interaction directly. The stateful-audit module also retains a fixed-g=1.0/1.3 legacy wrapper while factorial receipts remain alias-free.

## Conclusion boundary

This result falsifies moment zeroing as a valid memory-neutralization intervention; it does not falsify state-dependent optimizer-history effects.

Allowed: describe whether the formal g=1.10 optimizer-update divergence is associated mainly with the observed non-scalar gradient residual, accumulated RAdam state, or their interaction in these frozen virtual updates.

This is a frozen virtual-update diagnostic, not a continuation-training intervention. The eight audit minibatches per seed must not be treated as independent training replicates. It does not establish that optimizer memory or update divergence caused an FID/KID improvement. No training, samples, FID, or KID were produced.

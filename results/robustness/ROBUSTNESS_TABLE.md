# Cross-seed robustness table

## Scope and conventions

This table consolidates the frozen evidence from PR #49 (single-trajectory
stateful RAdam audit), PR #51 (the seed-4/5 evaluation handoff), and PR #53
(three disjoint NFE=1 FID/KID-5k blocks for training seeds 3/4/5).

`gap` is B−A: fixed learning rate, `g=1.3` minus `g=1.0`. `control` is C−B:
fresh-state LR-matched `g=1.3` minus fixed-LR `g=1.3`. Lower FID/KID is
better, so a negative delta is an improvement. Absorption is
`control / (−gap)`: positive means the matching control removes part of the
gap-associated advantage; negative means it strengthens that advantage.

The three 5k blocks are independent *sampling* blocks, not extra training
seeds. Each seed row is its mean across blocks; `mean` and `std` are
descriptive mean and sample SD across the three training seeds. `R_opt` is
reported only where it was actually measured; missing seed-4/5 measurements
are never filled with the seed-3 result.

The absorption entry in the `mean` row is the arithmetic mean of the three
per-seed absorption ratios. It is deliberately **not** the ratio of the mean
control delta to the mean gap delta, because the former preserves the unit of
cross-seed heterogeneity being summarized.

## Combined table

| row | gap effect | fresh-state control | $R_{opt}$ | absorption ratio | FID delta (gap; control) | KID delta (gap; control) | sign agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed 3 | improves | regresses | 9.90% @256k (range 8.22-9.90% over K) | FID +83.9%; KID +84.1% | −107.91; +90.49 | −0.120986; +0.101760 | gap FID 3/3 blocks −; KID 3/3 blocks −; ctrl FID 3/3 blocks +; KID 3/3 blocks + |
| seed 4 | improves | regresses | not measured | FID +13.3%; KID +8.9% | −37.20; +4.95 | −0.043881; +0.003914 | gap FID 3/3 blocks −; KID 3/3 blocks −; ctrl FID 3/3 blocks +; KID 3/3 blocks + |
| seed 5 | improves | improves | not measured | FID −80.7%; KID −81.8% | −18.00; −14.52 | −0.021835; −0.017869 | gap FID 3/3 blocks −; KID 3/3 blocks −; ctrl FID 3/3 blocks −; KID 3/3 blocks − |
| mean | all seed effects improve | **mean masks sign reversal** | not estimable (1/3 seeds measured) | FID +5.5%; KID +3.7% | −54.37; +26.97 | −0.062234; +0.029268 | gap −3/3; control +2/−1 |
| std | between-seed dispersion | between-seed dispersion | not estimable | FID 82.5%; KID 83.1% | 47.35; 55.86 | 0.052061; 0.063717 | sample SD over 3 training seeds |
| sign agreement | **Robust: FID −3/3; KID −3/3** | **Seed-dependent: FID +2/−1; KID +2/−1** | **Not reproduced: seed 3 only** | FID +2/−1; KID +2/−1 | gap −3/3; control +2/−1 | gap −3/3; control +2/−1 | within every seed, all 3 disjoint blocks agree |

## Observation verdicts

| Observation | Tag | Evidence-supported statement |
| --- | --- | --- |
| Fixed-LR gap effect (B−A) | **Robust** | B improves both FID and KID versus A in every training seed and every disjoint block: 3/3 negative FID deltas and 3/3 negative KID deltas across seeds. |
| Fresh-state control effect (C−B) | **Seed-dependent / sign-unstable** | Seed 3 is strongly positive, seed 4 weakly positive, and seed 5 negative for both FID and KID. The sign is stable within each seed's three disjoint sampling blocks, so sampling-block noise does not explain the seed-4/5 reversal. Do not summarize this as a single positive mean. |
| Absorption ratio | **Seed-dependent / sign-unstable** | It inherits the control reversal: seed 3 removes about 84% of the gap advantage, seed 4 about 9-13%, and seed 5 has negative absorption (the matched control strengthens the advantage). |
| Stateful $R_{opt}$ along the #49 trajectory | **Not reproduced** | Seed 3 has a preserved, non-skipped four-state audit with $R_{opt}=8.22\%-9.90\%$; seeds 4 and 5 were not measured. It is longitudinally observed, but it is not a cross-seed result. |

## What survives across seeds?

Only the fixed-LR larger-gap comparison survives this three-seed check: it
improves both disjoint-block FID-5k and KID-5k in all seeds. The fresh-state
LR control does **not** have a seed-invariant direction. Its positive average
is therefore descriptive bookkeeping, not a robustness claim, and it must be
reported as **seed-dependent / sign-unstable**. The $R_{opt}$ longitudinal
pattern currently has no seeds-4/5 replication and remains **not reproduced**
across seeds.

## Provenance

- PR #49 / merge `5561ca3`: `analysis/same_trajectory_longitudinal/longitudinal_summary.csv` (SHA-256 `864fc251bf9e7ef300117c11b7e91e02b6b1268c7d94bedb19545c53ee12f435`).
- PR #51 / merge `1db0322`: frozen seed-4/5 handoff in `docs/ROLE_E_DISJOINT_5K_HANDOFF.md`; it establishes the admissible endpoint and disjoint-block contract.
- PR #53 / merge `6b0a110`: `results/gap_lr_matched/disjoint_5k_0813/blockwise_results.csv`, copied verbatim here as `disjoint_5k_blockwise_results.csv` for this table's self-contained calculation (SHA-256 `d748a4dcb33589276b82d7c1825b1fce8cfbf2c7a14e0bbe857de150bc960189`).

All FID/KID entries are NFE=1, FP32, 5k-sample proxy evaluations. They are not
FID-50k or KID-50k claims.

# Evaluation storage estimate

## Scope and baseline

This estimate covers the server-side staging and outputs required to evaluate
the two frozen matrices. It does **not** budget training work directories,
optimizer states, or generated image archives that are intentionally not kept
by the staged evaluator.

The baseline is the completed six-checkpoint q=256 formal run inspected on
2026-07-31:

- 12 checkpoint×NFE jobs occupied **20 MiB** of evaluator output, or about
  **1.67 MiB per job**;
- its result summary occupied **44 KiB**;
- each staged network checkpoint was **223,169,426–516 bytes** (about
  **213 MiB**); and
- the shared Inception detector cache occupied **92 MiB**.

The 5k and 50k evaluator jobs use the same retained-output layout. Their
compute cost differs, but the observed retained disk footprint is expected to
be approximately per-job rather than proportional to sample count; generated
samples and features are not retained in the result directory.

## Planned capacity

| Component | Calculation | Estimate |
| --- | --- | ---: |
| q=256 evaluator outputs | 36 jobs × 1.67 MiB | 60 MiB |
| q=128 evaluator outputs | 12 jobs × 1.67 MiB | 20 MiB |
| summaries/manifests | four result summaries, rounded up | 1 MiB |
| shared detector cache | observed cache | 92 MiB |
| 24 staged checkpoints | 24 × 213 MiB | 5.0 GiB |
| canonical dataset allowance | conservative staging allowance | 1.0 GiB |
| **Working subtotal** | checkpoint staging + evaluation workspace | **about 6.2 GiB** |
| **Reserved free space** | subtotal rounded up with ~25% contingency | **8 GiB** |

Before launch, measure the actual canonical dataset archive and the completed
checkpoint files, then increase the reservation if either exceeds this
baseline. If training directories or image/feature caches are retained, budget
them separately; they are outside the 8 GiB evaluation-workspace reservation.

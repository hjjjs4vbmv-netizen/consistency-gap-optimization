# q=256 confirmatory formal evaluation record

**Protocol:** `staged-checkpoint-evaluation-v1`

**Evidence class:** formal 50k benchmark
**Evaluation Git commit:** `8375d46ca4c65e85ab399fcf1effe22ebb766790`

## Frozen matrix and eligibility

This record supersedes the historical four-cell seed 4/5 example. The formal
matrix is the frozen q=256 confirmatory comparison of fixed sigmoid and
global-only sigmoid (`global_gap_scale=1.10`) at training seeds 3, 4, and 5.
All six predeclared cells entered formal evaluation: no cell, NFE mode, or
training seed was removed on the basis of quick 5k results.

Each cell passed the version-2 training-integrity gate. In particular, the
checker loaded the evaluated pickle, found a finite EMA module, verified the
declared schedule and global-gap-scale identity, and matched the checkpoint
hash to the receipt and frozen matrix.

| Checkpoint ID | Method | Training seed | Checkpoint SHA-256 |
| --- | --- | ---: | --- |
| `confirmatory-256k-seed3-fixed` | fixed sigmoid, `g=1.00` | 3 | `09a41e1e7c03dcdf5ffb93bb68687390278b4b190183dfff92bacc1bf79738d9` |
| `confirmatory-256k-seed3-global110` | global-only sigmoid, `g=1.10` | 3 | `24875430eea4679a416ae921c3e9ae16142f6416d2a0edf970764384ef964bed` |
| `confirmatory-256k-seed4-fixed` | fixed sigmoid, `g=1.00` | 4 | `ac94e7b07e5b7628e6b14b26155fb3de09e42373497183d39aba4fe9863663c9` |
| `confirmatory-256k-seed4-global110` | global-only sigmoid, `g=1.10` | 4 | `62a6122a7be523aeb12875d96e96312e9c90efde9eafb75d730c75ceea0e8862` |
| `confirmatory-256k-seed5-fixed` | fixed sigmoid, `g=1.00` | 5 | `21fab0e501bb27032c0e49a553b05a2800ea0fbe20a2a1d94a6bbf5276f2b72a` |
| `confirmatory-256k-seed5-global110` | global-only sigmoid, `g=1.10` | 5 | `491dc887990e6d9f6fde70b5d12775aaf4bfc6155b731682926b02061c253e9b` |

## Formal environment and settings

Every cell used FP32 on one NVIDIA A100-PCIE-40GB, Python 3.9.18, PyTorch
2.3.0, SciPy 1.13.1, and CUDA 12.1. The evaluator used the
`inception-2015-12-05` TorchScript detector in feature mode and the canonical
CIFAR-10 archive with SHA-256
`9fd64620e37bfc0c995535fa52701c9641bcd07635008bfda0c9fbddde1a4ed6`.

The formal pass used generation seeds `0-49999`, metric seed `20260730`, and
one evaluation per metric/cell. NFE=1 omitted `--mid_t` (`mid_t=[]` in the
manifest); NFE=2 used `--mid_t=0.821` (`mid_t=[0.821]`). It completed 12
checkpoint/NFE jobs and 24 finite KID-50k/FID-50k metric records.

## Completed formal results

Lower is better for KID and FID. Pairing is exactly
`training_seed + budget_kimg + nfe + metric`; the delta is `global_only -
fixed`, so a negative delta favors global-only. Sample SD is the paired-delta
sample standard deviation over the three training seeds.

| Metric | NFE | Fixed mean | Global-only mean | Mean paired delta | Paired SD | Global-only wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KID-50k | 1 | 0.332009663 | 0.306884229 | -0.025125434 | 0.010316682 | 3 / 3 |
| FID-50k | 1 | 311.586158925 | 297.402566815 | -14.183592109 | 3.296938320 | 3 / 3 |
| KID-50k | 2 | 0.205662306 | 0.058241853 | -0.147420454 | 0.129031614 | 3 / 3 |
| FID-50k | 2 | 197.310118996 | 70.393238523 | -126.916880474 | 110.560376043 | 3 / 3 |

All 12 individual paired metric comparisons favor global-only; there are no
fixed wins or ties. At NFE=2, seed 5 is nearly flat for both endpoints, so the
larger effects at seeds 3 and 4 should be reported as heterogeneous paired
outcomes rather than a uniform NFE effect.

## Main-text figures

![Figure 1: per-seed paired comparison](../results/q256_256k_formal/figures/figure1_per_seed_paired_comparison.png)

*Figure 1. Per-seed paired comparison at 256 kimg. Each line connects the
fixed and global-only checkpoint trained with the same seed; open markers are
fixed and filled markers are global-only. All panels use 50k samples per
checkpoint and lower values are better. Y-scales are panel-specific.*

![Figure 2: mean paired delta and between-seed variation](../results/q256_256k_formal/figures/figure2_mean_delta_seed_variation.png)

*Figure 2. Paired deltas are global-only minus fixed, so negative values favor
global-only. Colored points are the three independent training seeds; black
diamonds and whiskers are the mean and sample SD, respectively, not confidence
intervals. NFE=2 has visibly greater between-seed variation, driven by the
near-flat seed-5 comparison.*

## Appendix and machine-readable diagnostics

Exact sign-test descriptions, bootstrap sensitivity intervals,
leave-one-seed-out summaries, coefficient-of-variation diagnostics, rank
consistency, and geometric/median/worst-case summaries are retained outside
the main text in `paired_statistics.md`, `paired_statistics.json`, and
`analysis/q256_extended_statistics.json`. They remain descriptive appendix or
machine-readable material. There are only three independent training seeds;
bootstrap resampling of those seeds does not create additional independent
observations.

## Interpretation boundary and versioned package

This is formal evidence from a predeclared three-seed paired matrix. Report
the values as descriptive paired results; with only three independent training
seeds, do not make a significance claim from these sample standard deviations.

The portable result package is versioned at
`results/q256_256k_formal/`. It retains the machine-generated metric
values and provenance needed for review, but deliberately omits generated
samples, feature caches, and host-specific paths:

```text
results/q256_256k_formal/evaluation_results.csv
results/q256_256k_formal/paired_differences.csv
results/q256_256k_formal/paired_statistics.json
results/q256_256k_formal/paired_statistics.md
results/q256_256k_formal/environment_manifest.json
results/q256_256k_formal/README.md
```

The frozen logical matrix, immutable receipt digests, formal-promotion policy,
and evaluator implementation are retained in Git. Runtime paths are only host
bindings and do not define the experiment.

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
fixed wins or ties. The NFE=2 effect is directionally consistent but has
substantial between-seed variation, especially for FID.

## Paired robustness and NFE heterogeneity

For each seed, the relative improvement is
`100 × (fixed - global-only) / fixed`, so positive values favor global-only.
Arithmetic improvement averages these three percentages; geometric improvement
is `100 × (1 - geometric mean(global-only/fixed))`. Rank consistency is the
Spearman correlation between lower-is-better seed ranks for fixed and
global-only. The seed CV applies to the relative improvements, and the
worst-case column is the least favorable seed.

| Metric | NFE | Arithmetic improvement | Geometric improvement | Median paired delta | Worst-case improvement | Seed CV | Rank consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KID-50k | 1 | 7.501846% | 7.531510% | -0.023481160 | 5.023325% | 38.111785% | 1.000000 |
| FID-50k | 1 | 4.544298% | 4.547537% | -12.485467394 | 3.768644% | 21.164185% | 1.000000 |
| KID-50k | 2 | 50.599851% | 61.818842% | -0.201894470 | 0.249718% | 86.826602% | 0.500000 |
| FID-50k | 2 | 47.441515% | 55.593373% | -167.596988744 | 3.885378% | 80.519432% | 0.500000 |

The NFE=2-minus-NFE=1 relative-effect contrast has mean +43.098004 percentage
points for KID and +42.897217 points for FID. It favors a larger NFE=2 effect
in two of three seeds for each metric, but the third seed has a small negative
contrast (-4.773607 KID points; -0.358444 FID points). Thus this is effect
heterogeneity to report, not a claim that NFE modifies the effect generally.

For each metric/NFE stratum, the two-sided exact sign test for the 3/3
directional split is `p=0.25`. Its resolution is limited by the three
independent training seeds, so it is included as a descriptive directional
check rather than a significance result. Deterministic 10,000-resample
percentile bootstrap intervals and leave-one-seed-out summaries are available
in the result package as auxiliary sensitivity analyses only; the bootstrap
does not supply additional independent observations. Each leave-one-seed-out
subset still has 2/2 global-only wins, although NFE=2 magnitudes are notably
seed-sensitive.

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

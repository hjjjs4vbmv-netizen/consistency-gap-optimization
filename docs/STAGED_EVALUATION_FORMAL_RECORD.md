# Staged formal evaluation record

**Protocol:** `staged-checkpoint-evaluation-v1`

**Evidence class:** formal 50k benchmark

**Evaluation Git commit:** `125ece6d018e10a1c2cf13ea6e3beeda09667d23`

## Eligibility and frozen settings

All four 256 kimg training endpoints passed the training-integrity checker
before evaluation. Each receipt is bound to its checkpoint SHA-256, the
declared training seed and budget, a clean-exit log, consistent final progress,
and finite loss/state checks.

| Checkpoint ID | Method | Training seed | Checkpoint SHA-256 |
| --- | --- | ---: | --- |
| `seed4_fixed_256k` | fixed | 4 | `ac94e7b07e5b7628e6b14b26155fb3de09e42373497183d39aba4fe9863663c9` |
| `seed4_global110_256k` | global110 | 4 | `62a6122a7be523aeb12875d96e96312e9c90efde9eafb75d730c75ceea0e8862` |
| `seed5_fixed_256k` | fixed | 5 | `21fab0e501bb27032c0e49a553b05a2800ea0fbe20a2a1d94a6bbf5276f2b72a` |
| `seed5_global110_256k` | global110 | 5 | `491dc887990e6d9f6fde70b5d12775aaf4bfc6155b731682926b02061c253e9b` |

Every formal cell used FP32 on one GPU, the full canonical CIFAR-10 archive
(`9fd64620e37bfc0c995535fa52701c9641bcd07635008bfda0c9fbddde1a4ed6`),
generation seeds `0-49999`, metric seed `20260730`, and one metric evaluation.
NFE=1 uses `mid_t=[]`; NFE=2 uses `mid_t=[0.821]`.

## Completed formal matrix

The manifest completed all 8 checkpoint/NFE cells and all 16 KID-50k/FID-50k
records. Lower values are better for both metrics.

| Metric | NFE | Fixed mean | Global110 mean | Global110 - fixed | Paired sample SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| KID-50k | 1 | 0.326680064 | 0.300732493 | -0.025947571 | 0.014450342 |
| FID-50k | 1 | 307.084470367 | 291.850021214 | -15.234449153 | 3.887647286 |
| KID-50k | 2 | 0.160439545 | 0.059449721 | -0.100989824 | 0.142700718 |
| FID-50k | 2 | 155.516368430 | 70.828037549 | -84.688330881 | 117.250548388 |

Pairing is by `training_seed`; delta direction is `global110 - fixed`. The
underlying paired values are retained in the external formal summary alongside
the long-form 16-row result table. Checkpoint files, generated images, feature
caches, and server-local result trees are intentionally excluded from Git.

## Interpretation boundary

For seeds 4 and 5, the paired deltas are negative for both KID-50k and FID-50k
at NFE=1 and NFE=2. This is formal evidence under the frozen 50k protocol, but
there are only two paired training seeds. Report it as a descriptive paired
result and do not make a significance claim from this sample size alone.

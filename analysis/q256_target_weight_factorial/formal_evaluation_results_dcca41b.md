# q256 formal evaluation final record

Status: **PASS**. The formal seed3/4/5 matrix has 24/24 unique PASS job receipts across the immutable v3/v5/v6 continuation chain. The v4 and v5 failures remain preserved as fail-closed audit evidence and were not treated as results.

Frozen protocol: FP32, 50,000 samples per job, sample seeds `0..49999`, metric seed `20260730`, KID then FID from the same retained generated Inception features, primary-first NFE ordering, and `mid_t=0.821` for NFE2. Training source is `dcca41b19e7c45512b5fbe98776520396a1bf9ac`; formal GPU0 is `GPU-d79117bb-8d91-4f2e-d7bb-718e347ce859`.

## Seed5 results

| Arm | NFE1 FID | NFE1 KID | NFE2 FID | NFE2 KID |
|---|---:|---:|---:|---:|
| A | 318.638529753 | 0.329434699442 | 75.266429038 | 0.062150207000 |
| B | 308.854892082 | 0.326125451789 | 70.236021799 | 0.056118489655 |
| C | 308.414344315 | 0.325736081314 | 70.485902851 | 0.056507776707 |
| D | 315.200350185 | 0.329756152860 | 75.407868983 | 0.062109354377 |

All eight seed5 receipts are PASS; every GPU monitor is PASS with no foreign-process incident; within every job, FID and KID bind byte-identical generated-feature SHA-256 values. GPU0 was compute-idle after completion.

## Continuation audit

- v3 contributed 8 PASS jobs and then stopped for an audit-probe timeout.
- v4 contributed no PASS job and preserved a second audit-probe timeout receipt.
- v5 contributed 12 PASS jobs and stopped when a real foreign GPU0 process named `cudaCheck` appeared; it was not whitelisted or ignored.
- v6 hash-bound all 20 prior PASS jobs and both prior failure histories, then completed only the four remaining seed5 NFE2 jobs with terminal status PASS.

The machine-readable record is `formal_evaluation_results_dcca41b.json`. The authoritative server completion is `/data/raw/ECT/ect_runs/q256-target-weight-factorial-20260819/evaluation/frozen-eval-feature-reuse-primary-first-v6-seed5-nfe2-continuation/evaluation_completion.json` (SHA-256 `d827916918c4310d4b4075dd41259a78871c48882a601ab76a7d0123b9429103`).

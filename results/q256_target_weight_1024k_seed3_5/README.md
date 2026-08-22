# q256 target × weight factorial: seed 3–5 at 1024 kimg

This directory records the completed full-state continuation of the formal q256 target × weight factorial for seeds 3, 4, and 5. Each arm resumed from its immutable 256 kimg state and stopped at a total budget of exactly 1024 kimg (8000 attempted iterations); it did not add 1024 kimg on top of the source state.

## Arm definitions

| Arm | Target gap scale | Denominator gap scale |
|---|---:|---:|
| A | 1.0 | 1.0 |
| B | 1.1 | 1.1 |
| C | 1.1 | 1.0 |
| D | 1.0 | 1.1 |

## Completion status

All 12 arms passed the final full-state audit. Every final state has `cur_nimg=1024000`, `attempted_iteration=8000`, optimizer state, EMA, GradScaler state, and rank-local RNG/sampler state. Exact accepted-update counts, AMP skips, elapsed times, final-state SHA256 values, snapshot SHA256 values, and archive paths are in [training_results.csv](training_results.csv).

| Seed | Arms | Source kimg | Final kimg | Final attempts | Status |
|---:|---:|---:|---:|---:|:---:|
| 3 | 4 | 256 | 1024 | 8000 | PASS |
| 4 | 4 | 256 | 1024 | 8000 | PASS |
| 5 | 4 | 256 | 1024 | 8000 | PASS |

## Formal FID-50k and KID-50k

The formal v2 evaluation used 50,000 samples (`0-49999`), metric seed `20260730`, FP32 evaluation, NFE=1, and NFE=2 with frozen `mid_t=0.821`. Every KID/FID pair used byte-identical generated Inception features, verified by SHA256. No ranking or post-hoc arm selection is included here.

| Seed | Arm | FID, NFE=1 | KID, NFE=1 | FID, NFE=2 | KID, NFE=2 |
|---:|:---:|---:|---:|---:|---:|
| 3 | A | 9.6694058203 | 0.005999107735 | 2.8663959931 | 0.001053589207 |
| 3 | B | 8.6863514559 | 0.005436252860 | 2.8315304041 | 0.000989466236 |
| 3 | C | 10.6720017959 | 0.006814711532 | 3.0848048553 | 0.001354885483 |
| 3 | D | 8.8091992279 | 0.005311956116 | 2.8512938748 | 0.000955548589 |
| 4 | A | 8.2934221248 | 0.004999751504 | 2.7133167163 | 0.000885492127 |
| 4 | B | 8.5222345426 | 0.005156002970 | 2.7361111410 | 0.000923558609 |
| 4 | C | 9.2009753806 | 0.005765106644 | 2.7568436821 | 0.000928640075 |
| 4 | D | 8.5941061399 | 0.005125233418 | 2.7378431817 | 0.000917038511 |
| 5 | A | 8.3794531182 | 0.005054089152 | 2.8623330982 | 0.001030845165 |
| 5 | B | 7.7832140364 | 0.004753083584 | 2.7611816140 | 0.000938200345 |
| 5 | C | 7.6425691502 | 0.004610289454 | 2.7530926415 | 0.000907406071 |
| 5 | D | 7.7936884254 | 0.004595977693 | 2.7563193830 | 0.000892839905 |

Full-precision metric rows and generated-feature identities are in [evaluation_results.csv](evaluation_results.csv). The unchanged final collector outputs are [seed3-final-report-v2.json](seed3-final-report-v2.json) and [seed45-final-report-v2.json](seed45-final-report-v2.json).

## Reproducibility and archive

- Formal experiment base: `dcca41b19e7c45512b5fbe98776520396a1bf9ac`.
- Continuation implementation: `458205192722883df393a8d017c26e6fa46f48f7`.
- MatPool runtime extraction support and executed checkout: `12f905dff2bf474495abb186c215ca2ea959099e`.
- Runtime: Python 3.10.12, PyTorch 2.2.0a0+81ea7a4, CUDA 12.3, cuDNN 8907.
- GPU: NVIDIA A100-PCIE-40GB.
- Persistent archive root: `/mnt/ect_project/q256_target_weight_1024k`.
- Final states and snapshots: `runs/q256-target-weight-1024k/seed{3,4,5}/arm{A,B,C,D}`.
- Formal v2 metric records: `evaluation/q256-target-weight-1024k-fidkid50k-v2-shared-features/jobs`.
- Audit reports, queue logs, evaluation tools, code bundles, immutable 256 kimg source states, and the deterministic container image are stored under the same archive root.

The archive retains final full states, snapshots, configuration, telemetry, logs, and previews. Regenerable 50k sample arrays, duplicate Inception feature arrays, and per-job detector caches are intentionally excluded from persistent storage; their metric values and exact generated-feature SHA256 identities are retained in the reports.

## Included tools

- `scripts/run_q256_target_weight_1024k_seed_queue.sh`: sequential A→B→C→D full-state continuation.
- `scripts/run_q256_target_weight_1024k_fidkid_queue.sh`: formal NFE=1/2 FID/KID queue.
- `scripts/audit_q256_target_weight_1024k_final_states.py`: final training-state audit.
- `scripts/collect_q256_target_weight_1024k_final_report.py`: strict training/evaluation result collector.

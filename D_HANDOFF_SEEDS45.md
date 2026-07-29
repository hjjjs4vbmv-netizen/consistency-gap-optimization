# Role C -> Role D Handoff: Confirmatory 256k seeds 4 & 5

Frozen training commit: `ab03f9e03b7b82425282abc3bf661067ca45875a` (branch role-c/seeds-4-5-confirmatory; training code == 6d4bc7d / PR#24 merged; PR#25 added no training-code diff).

All 4 checkpoints passed integrity: kimg=256 reached, 0 NaN in loss history, expected schedule, gap scale held.

| Method | Seed | Run | Checkpoint | kimg | Commit | Integrity |
| --- | ---: | --- | --- | ---: | --- | --- |
| Fixed | 4 | seed4_fixed | `/root/ect_runs/confirmatory_256k/seed4_fixed/network-snapshot-latest.pkl` | 256 | ab03f9e | Passed |
| Global 1.10 | 4 | seed4_global110 | `/root/ect_runs/confirmatory_256k/seed4_global110/network-snapshot-latest.pkl` | 256 | ab03f9e | Passed |
| Fixed | 5 | seed5_fixed | `/root/ect_runs/confirmatory_256k/seed5_fixed/network-snapshot-latest.pkl` | 256 | ab03f9e | Passed |
| Global 1.10 | 5 | seed5_global110 | `/root/ect_runs/confirmatory_256k/seed5_global110/network-snapshot-latest.pkl` | 256 | ab03f9e | Passed |

## seed4_fixed (seed 4)
- method: Fixed (mapping=sigmoid, global_gap_scale=1.0)
- checkpoint: `/root/ect_runs/confirmatory_256k/seed4_fixed/network-snapshot-latest.pkl`
- checkpoint sha256: `ac94e7b07e5b7628e6b14b26155fb3de09e42373497183d39aba4fe9863663c9`
- training-state: `/root/ect_runs/confirmatory_256k/seed4_fixed/training-state-latest.pt` (cur_nimg=256000, optimizer+scaler)
- config: `/root/ect_runs/confirmatory_256k/seed4_fixed/training_options.json` (adj=sigmoid)
- log: `/root/ect_runs/confirmatory_256k/seed4_fixed/log.txt` ; summary: `/root/ect_runs/confirmatory_256k/seed4_fixed/train_summary.csv`
- loss: 2000 rows, 0 NaN, last=16.566 min=13.389 max=30.562
- gap_over_sigmoid_gap_mean: 1 -> 1 (target 1.0)

## seed4_global110 (seed 4)
- method: Global 1.10 (mapping=global_sigmoid, global_gap_scale=1.10)
- checkpoint: `/root/ect_runs/confirmatory_256k/seed4_global110/network-snapshot-latest.pkl`
- checkpoint sha256: `62a6122a7be523aeb12875d96e96312e9c90efde9eafb75d730c75ceea0e8862`
- training-state: `/root/ect_runs/confirmatory_256k/seed4_global110/training-state-latest.pt` (cur_nimg=256000, optimizer+scaler)
- config: `/root/ect_runs/confirmatory_256k/seed4_global110/training_options.json` (adj=global_sigmoid)
- log: `/root/ect_runs/confirmatory_256k/seed4_global110/log.txt` ; summary: `/root/ect_runs/confirmatory_256k/seed4_global110/train_summary.csv`
- loss: 2000 rows, 0 NaN, last=16.555 min=13.345 max=30.100
- gap_over_sigmoid_gap_mean: 1.10000001913 -> 1.10000038269 (target 1.10)

## seed5_fixed (seed 5)
- method: Fixed (mapping=sigmoid, global_gap_scale=1.0)
- checkpoint: `/root/ect_runs/confirmatory_256k/seed5_fixed/network-snapshot-latest.pkl`
- checkpoint sha256: `21fab0e501bb27032c0e49a553b05a2800ea0fbe20a2a1d94a6bbf5276f2b72a`
- training-state: `/root/ect_runs/confirmatory_256k/seed5_fixed/training-state-latest.pt` (cur_nimg=256000, optimizer+scaler)
- config: `/root/ect_runs/confirmatory_256k/seed5_fixed/training_options.json` (adj=sigmoid)
- log: `/root/ect_runs/confirmatory_256k/seed5_fixed/log.txt` ; summary: `/root/ect_runs/confirmatory_256k/seed5_fixed/train_summary.csv`
- loss: 2000 rows, 0 NaN, last=15.188 min=13.347 max=31.666
- gap_over_sigmoid_gap_mean: 1 -> 1 (target 1.0)

## seed5_global110 (seed 5)
- method: Global 1.10 (mapping=global_sigmoid, global_gap_scale=1.10)
- checkpoint: `/root/ect_runs/confirmatory_256k/seed5_global110/network-snapshot-latest.pkl`
- checkpoint sha256: `491dc887990e6d9f6fde70b5d12775aaf4bfc6155b731682926b02061c253e9b`
- training-state: `/root/ect_runs/confirmatory_256k/seed5_global110/training-state-latest.pt` (cur_nimg=256000, optimizer+scaler)
- config: `/root/ect_runs/confirmatory_256k/seed5_global110/training_options.json` (adj=global_sigmoid)
- log: `/root/ect_runs/confirmatory_256k/seed5_global110/log.txt` ; summary: `/root/ect_runs/confirmatory_256k/seed5_global110/train_summary.csv`
- loss: 2000 rows, 0 NaN, last=15.215 min=13.409 max=25.291
- gap_over_sigmoid_gap_mean: 1.1000003469 -> 1.10000058239 (target 1.10)

## Recommended evaluation

All four `network-snapshot-latest.pkl` are authoritative 256-kimg endpoints (EMA inside `ema` key). Role D evaluates NFE=1, NFE=2, KID-5k, FID-5k. fixed vs global differ ONLY by mapping/gap_scale; all other settings identical (verified by resolved-config diff for seed 3).

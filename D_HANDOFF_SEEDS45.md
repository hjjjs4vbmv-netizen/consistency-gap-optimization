# Role C -> Role D Handoff: Confirmatory 256k seeds 4 & 5

## Provenance

- **Executed training source commit:** `ab03f9e03b7b82425282abc3bf661067ca45875a`
- **Training-code baseline:** `6d4bc7d`
- **PR head / handoff-document commit:** recorded by the Git commit containing this document.

The four training jobs were launched from `ab03f9e03b7b82425282abc3bf661067ca45875a`. Its training code is equivalent to baseline `6d4bc7d`; the PR/handoff-document commit is distinct and must not be described as the executed training source.

## Training health

Each run had one initialization-time AMP-skipped optimizer step, but no NaN values were recorded in the 2000-row loss history. Finite losses were recorded from the first completed training iteration onward.

This is training-health evidence only. Loss is not an evaluation metric and must not be used to infer KID or FID; Role D's independent generation-quality evaluation remains authoritative.

## Checkpoint transfer and verification

The `/root/ect_runs/...` paths below are node-local provenance references, not a shared Role D handoff location. Role D does **not** evaluate by assuming access to the training node or its filesystem.

Transfer archive (staged outside the training node):

- archive: `D:\\seeds45_ckpts_full.tar.gz`
- size: `3,291,447,686` bytes
- archive SHA256: `1bdb147e535fe0b4f069f4106e28a7a6b065b317ca9df38cfbe9643772937608`
- extract: `tar xzf D:\\seeds45_ckpts_full.tar.gz` -> `seeds45_package_full/`

The archive contains all four authoritative `network-snapshot-latest.pkl` files (including EMA weights), their configurations, logs, loss histories, and training states.

**Required Role D acceptance check before evaluation:** obtain the archive through the agreed transfer channel, recompute its SHA256, extract it, recompute the SHA256 of each `network-snapshot-latest.pkl`, and reply on the PR confirming all five values match. Until that confirmation is posted, the archive's accessibility to Role D is not assumed.

| Method | Seed | Run | Training-node checkpoint reference | Checkpoint SHA256 | kimg | Executed source | Integrity |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| Fixed | 4 | seed4_fixed | `/root/ect_runs/confirmatory_256k/seed4_fixed/network-snapshot-latest.pkl` | `ac94e7b07e5b7628e6b14b26155fb3de09e42373497183d39aba4fe9863663c9` | 256 | `ab03f9e` | Passed on training node |
| Global 1.10 | 4 | seed4_global110 | `/root/ect_runs/confirmatory_256k/seed4_global110/network-snapshot-latest.pkl` | `62a6122a7be523aeb12875d96e96312e9c90efde9eafb75d730c75ceea0e8862` | 256 | `ab03f9e` | Passed on training node |
| Fixed | 5 | seed5_fixed | `/root/ect_runs/confirmatory_256k/seed5_fixed/network-snapshot-latest.pkl` | `21fab0e501bb27032c0e49a553b05a2800ea0fbe20a2a1d94a6bbf5276f2b72a` | 256 | `ab03f9e` | Passed on training node |
| Global 1.10 | 5 | seed5_global110 | `/root/ect_runs/confirmatory_256k/seed5_global110/network-snapshot-latest.pkl` | `491dc887990e6d9f6fde70b5d12775aaf4bfc6155b731682926b02061c253e9b` | 256 | `ab03f9e` | Passed on training node |

## Per-run evidence

### seed4_fixed (seed 4)

- method: Fixed (`mapping=sigmoid`, `global_gap_scale=1.0`)
- training-state: `/root/ect_runs/confirmatory_256k/seed4_fixed/training-state-latest.pt` (`cur_nimg=256000`)
- config/log: `training_options.json` / `log.txt`
- loss: 2000 rows; no recorded NaN; last=16.566, min=13.389, max=30.562
- `gap_over_sigmoid_gap_mean`: 1 -> 1 (target 1.0)

### seed4_global110 (seed 4)

- method: Global 1.10 (`mapping=global_sigmoid`, `global_gap_scale=1.10`)
- training-state: `/root/ect_runs/confirmatory_256k/seed4_global110/training-state-latest.pt` (`cur_nimg=256000`)
- config/log: `training_options.json` / `log.txt`
- loss: 2000 rows; no recorded NaN; last=16.555, min=13.345, max=30.100
- `gap_over_sigmoid_gap_mean`: 1.10000001913 -> 1.10000038269 (target 1.10)

### seed5_fixed (seed 5)

- method: Fixed (`mapping=sigmoid`, `global_gap_scale=1.0`)
- training-state: `/root/ect_runs/confirmatory_256k/seed5_fixed/training-state-latest.pt` (`cur_nimg=256000`)
- config/log: `training_options.json` / `log.txt`
- loss: 2000 rows; no recorded NaN; last=15.188, min=13.347, max=31.666
- `gap_over_sigmoid_gap_mean`: 1 -> 1 (target 1.0)

### seed5_global110 (seed 5)

- method: Global 1.10 (`mapping=global_sigmoid`, `global_gap_scale=1.10`)
- training-state: `/root/ect_runs/confirmatory_256k/seed5_global110/training-state-latest.pt` (`cur_nimg=256000`)
- config/log: `training_options.json` / `log.txt`
- loss: 2000 rows; no recorded NaN; last=15.215, min=13.409, max=25.291
- `gap_over_sigmoid_gap_mean`: 1.1000003469 -> 1.10000058239 (target 1.10)

## Recommended evaluation

After the required transfer verification, Role D evaluates NFE=1, NFE=2, KID-5k, and FID-5k. Fixed and global110 differ only by mapping/gap scale; the remaining settings are identical per seed.
# Role C → Role D Handoff: Confirmatory 256k (seed 3)

Frozen training commit: `3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43` (branch role-c/confirmatory-gap-g110, PR #23 merged into main)
All checkpoints below passed integrity: loadable snapshot + training-state, kimg=256 reached, finite loss history, expected schedule, gap scale held.

| Method | Seed | Checkpoint | kimg | Commit | Integrity |
| --- | ---: | --- | ---: | --- | --- |
| Fixed | 3 | /root/ect_runs/confirmatory_256k/seed3_fixed/network-snapshot-latest.pkl | 256 | 3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43 | Passed |
| Global 1.10 | 3 | /root/ect_runs/confirmatory_256k/seed3_global110/network-snapshot-latest.pkl | 256 | 3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43 | Passed |

## Per-run detail

### Fixed (sigmoid, g=1.0)
- run dir: `/root/ect_runs/confirmatory_256k/seed3_fixed`
- snapshot sha256: `09a41e1e7c03dcdf5ffb93bb68687390278b4b190183dfff92bacc1bf79738d9`
- training-state: `training-state-latest.pt` (cur_nimg=256000, cur_tick=27, optimizer+scaler state)
- schedule: sigmoid; gap_over_sigmoid_gap_mean held at 1.000000 throughout
- loss: 2000 finite rows, min 13.193 / max 33.129 / final 17.063 (no NaN/Inf)
- config: `training_options.json` (adj=sigmoid, global_gap_scale=1.0)

### Global 1.10 (global_sigmoid, g=1.10)
- run dir: `/root/ect_runs/confirmatory_256k/seed3_global110`
- snapshot sha256: `24875430eea4679a416ae921c3e9ae16142f6416d2a0edf970764384ef964bed`
- training-state: `training-state-latest.pt` (cur_nimg=256000, cur_tick=27, optimizer+scaler state)
- schedule: global_sigmoid; gap_over_sigmoid_gap_mean held at 1.100000 throughout
- loss: 2000 finite rows, min 13.277 / max 31.419 / final 16.448 (no NaN/Inf)
- config: `training_options.json` (adj=global_sigmoid, global_gap_scale=1.1)

## Recommended evaluation checkpoints
Both `network-snapshot-latest.pkl` are the authoritative 256-kimg endpoints and
are recommended for Role D evaluation. The two arms differ ONLY by
`mapping`/`global_gap_scale`; all other training settings are identical
(verified by resolved-config diff). EMA weights are inside the snapshot (`ema` key).

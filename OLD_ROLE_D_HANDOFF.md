# Historical Role D handoff — Confirmatory 256k (seed 3)

> **Superseded for current paper-mainline work by
> [`ROLE_D_HANDOFF_COMPUTE_TO_QUALITY_20260822.md`](ROLE_D_HANDOFF_COMPUTE_TO_QUALITY_20260822.md); retained for historical provenance.**

This archival copy records the seed-3 handoff before the paper mainline moved
from optimizer-oriented reporting to compute-to-quality evidence.

## Provenance

- Training-code SHA: `3a0d603da97dd93ddbb6c7ce49e4a7351d54bb43`.
- PR-head SHA: `79143c685e5588948972c17457b1c51c7a77bb49`.
- Output root: `/root/ect_runs/confirmatory_256k/`.

| Method | Seed | Checkpoint | kimg | Integrity |
| --- | ---: | --- | ---: | --- |
| Fixed | 3 | `seed3_fixed/network-snapshot-latest.pkl` | 256 | Passed |
| Global 1.10 | 3 | `seed3_global110/network-snapshot-latest.pkl` | 256 | Passed |

## Per-run detail

### Fixed (sigmoid, g=1.0)

- Training-node run directory: `/root/ect_runs/confirmatory_256k/seed3_fixed`.
- Snapshot SHA-256:
  `09a41e1e7c03dcdf5ffb93bb68687390278b4b190183dfff92bacc1bf79738d9`.
- Training state: `training-state-latest.pt` at `cur_nimg=256000`,
  `cur_tick=27`, including optimizer and scaler state.
- Schedule: sigmoid; `gap_over_sigmoid_gap_mean` was held at `1.000000`.
- Loss: 2,000 finite rows; min 13.193, max 33.129, final 17.063.
- Configuration: `training_options.json` with `adj=sigmoid`,
  `global_gap_scale=1.0`.

### Global 1.10 (global_sigmoid, g=1.1)

- Training-node run directory:
  `/root/ect_runs/confirmatory_256k/seed3_global110`.
- Snapshot SHA-256:
  `24875430eea4679a416ae921c3e9ae16142f6416d2a0edf970764384ef964bed`.
- Training state: `training-state-latest.pt` at `cur_nimg=256000`,
  `cur_tick=27`, including optimizer and scaler state.
- Schedule: global sigmoid; `gap_over_sigmoid_gap_mean` was held at
  `1.100000`.
- Loss: 2,000 finite rows; min 13.277, max 31.419, final 16.448.
- Configuration: `training_options.json` with `adj=global_sigmoid`,
  `global_gap_scale=1.1`.

Both checkpoints contain EMA weights. The two arms differ only in mapping and
global gap scale; all other resolved training settings matched. The
authoritative current presentation and any new analysis must use the
compute-to-quality handoff above.

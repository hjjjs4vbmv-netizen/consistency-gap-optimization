# Collaborator formal-run handoff

## Outcome

The preregistered A/B/C training matrix completed successfully on ECT002.
The detached launcher reached `ALL FORMAL ARMS COMPLETE`; all three trainers
exited normally after 256 kimg.

| Arm | Gap | RAdam LR | Attempted iterations | Successful optimizer steps | Final GradScaler |
|---|---:|---:|---:|---:|---:|
| A | 1.0 | 1.0e-4 | 2000 | 1991 | 128 |
| B | 1.3 | 1.0e-4 | 2000 | 1992 | 256 |
| C | 1.3 | 1.2963523762588692e-4 | 2000 | 1992 | 256 |

FID/KID evaluation is intentionally not part of this handoff and remains a
Role E downstream task.

## Frozen provenance

- Protocol commit: `2f1005b1a14446c0efdf86a95f20a2d7fb172121`
- Training-code commit: `2357bb1d2531a343bdb4397f5a08f4d42a2d135b`
- Final authorization receipt SHA256:
  `6487fbcc5f63817c8e3a91968f45fb13437d1c580afa73966bdf0ad8061bb9fa`
- Internal post-run receipt SHA256:
  `3b715ba8b1bd1ce5b4109ea7968235d5547adf01f45ee7d6e3594521d625ebe9`
- GitHub-sanitized post-run receipt SHA256:
  `5e57adfa8232a12f67c3300d4efe888bbea9f5f4f5acacceb8875e8e479b5d00`
- Dataset SHA256:
  `a469a9f1b89d43a4a5a0fea42a351b6f107800fc32712881ea3d0ee8cc3a88c1`
- Transfer checkpoint SHA256:
  `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`
- Server-side validator: exit 0; validated
  `c0_star=1.2963523762588691`

The three `training_options.json` files are identical after removing only the
three preregistered differences (`global_gap_scale`, optimizer LR, and
`run_dir`). The three initialization-image hashes match, and the three
data-image hashes match.

## Fork-state index

The requested K points map to numbered files as follows. The slight +0.128
kimg offset at the first three points is caused by the first 128-image AMP
attempt and is permitted by the protocol; `train_summary.csv:processed_kimg`
is authoritative.

| Requested K | Actual kimg | State ID | Paired EMA/network snapshot |
|---:|---:|---|---|
| 32 | 32.128 | `training-state-000001.pt` | `network-snapshot-000001.pkl` |
| 64 | 64.128 | `training-state-000002.pt` | `network-snapshot-000002.pkl` |
| 128 | 128.128 | `training-state-000004.pt` | `network-snapshot-000004.pkl` |
| 256 | 256.000 | `training-state-000008.pt` | `network-snapshot-000008.pkl` |

The arm directories under the ECT002 run root are:

- `arm_a_g1_0_lr_fixed_s3`
- `arm_b_g1_3_lr_fixed_s3`
- `arm_c_g1_3_lr_matched_s3`

All 12 key states exist, are 669,421,866 bytes each, have recorded SHA256
digests in `collaborator_training_state_receipt.json`, and were successfully
deserialized after training. Each contains the current network, all 416 RAdam
parameter states (`step`, `exp_avg`, `exp_avg_sq`), attempted/successful step
counters, loss state, progress fields, and GradScaler state.

## Role D handoff boundary

These files are ready for the preregistered **same saved state + same paired
audit minibatch** current-gap counterfactual. They are not sufficient for
bitwise continuation of the uninterrupted trajectory because CPU/CUDA/NumPy/
Python RNG, sampler/DataLoader position, and worker prefetch state are not
stored. EMA is stored in the paired numbered network snapshot rather than the
training-state file.

The current `analysis/radam_stateful_update_audit.py` loader accepts
`sigmoid` but rejects the formal snapshots' `global_sigmoid` schedule name.
Role D must make a downstream loader-compatibility change before running the
longitudinal audit; the frozen launcher and training code must remain
unchanged.

## Validation notes

- No fatal error, traceback, or OOM was found in A/B/C logs.
- Each arm emitted the same non-fatal PyTorch `destroy_process_group()` cleanup
  warning after normal completion.
- The frozen receipt unit-test suite reports 5/6 because one mock contains the
  literal characters `\\n`; the real validator helper and the server-side
  receipt validation both passed.
- The machine-readable post-run receipt is the authoritative hash/state index.

## Artifacts in this delivery

- `final_gap_lr_audit_receipt.json`: Role E launch authorization.
- `collaborator_training_state_receipt.json`: post-run state/provenance receipt.
- `run_status.csv`: current three-arm training status.
- `job_manifest.json`: updated runtime provenance and downstream status.
